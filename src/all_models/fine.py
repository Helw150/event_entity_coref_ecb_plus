import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from transformers import LongformerModel
from coarse import EncoderCosineRanker


def get_raw_strings(sentences, mention_sentence, mapping=None):
    raw_strings = []
    offset = 0
    if mapping:
        offset = max(mapping.keys()) + 1
    else:
        mapping = {}
    for i, sentence in enumerate(sentences):
        raw_strings.append("<s>" + ' '.join(
            [tok.replace(" ", "")
             for tok in sentence.get_tokens_strings()]) + "</s>")
        if i == mention_sentence:
            mention_offset = offset
        for _ in sentence.get_tokens_strings():
            mapping[offset] = []
            offset += 1
    return raw_strings, mention_offset, mapping


def tokenize_and_map_pair(sentences_1, sentences_2, mention_sentence_1,
                          mention_sentence_2, tokenizer):
    max_seq_length = 512
    raw_strings_1, mention_offset_1, mapping = get_raw_strings(
        sentences_1, mention_sentence_1)
    raw_strings_2, mention_offset_2, mapping = get_raw_strings(
        sentences_2, mention_sentence_2, mapping)
    embeddings = tokenizer(' '.join(raw_strings_1),
                           ' '.join(raw_strings_2),
                           max_length=max_seq_length,
                           truncation=True,
                           padding="max_length")["input_ids"]
    counter = 0
    new_tokens = tokenizer.convert_ids_to_tokens(embeddings)
    if new_tokens[-1] != "<pad>":
        print("trunc")
    for i, token in enumerate(new_tokens):
        if token == "<s>" or token == "</s>" or token == "<pad>" or token == "<doc-s>" or token == "</doc-s>":
            continue
        elif token[0] == "Ġ" or new_tokens[i - 2] == "</doc-s>":
            counter += 1
            mapping[counter].append(i)
        else:
            mapping[counter].append(i)
            continue
    return embeddings, mapping, mention_offset_1, mention_offset_2


class CoreferenceMetricLearner(nn.Module):
    def __init__(self, device, mention_model_params):
        super(CoreferenceMetricLearner, self).__init__()
        self.device = device
        self.pos_weight = torch.tensor([0.1]).to(device)
        self.model_type = 'CoreferenceMetricLearner'
        self.event_encoder = EncoderCosineRanker(device)
        self.event_encoder.load_state_dict(mention_model_params)
        self.mention_dim = 1536
        self.input_dim = self.mention_dim * 3
        self.out_dim = 1

        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer_1 = nn.Linear(self.input_dim, self.mention_dim)
        self.hidden_layer_2 = nn.Linear(self.mention_dim, self.mention_dim)
        self.out_layer = nn.Linear(self.mention_dim, self.out_dim)

    def forward(self,
                sentence_1,
                sentence_2,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                labels=None):
        transformer_output_1 = self.event_encoder.get_sentence_vecs(sentence_1)
        transformer_output_2 = self.event_encoder.get_sentence_vecs(sentence_2)
        mention_reps_1 = self.event_encoder.get_mention_rep(
            transformer_output_1, start_pieces_1, end_pieces_1).squeeze(1)
        mention_reps_2 = self.event_encoder.get_mention_rep(
            transformer_output_2, start_pieces_2, end_pieces_2).squeeze(1)
        combined_rep = torch.cat(
            [mention_reps_1, mention_reps_2, mention_reps_1 * mention_reps_2],
            dim=1)
        combined_rep = self.dropout(combined_rep)
        first_hidden = F.relu(self.hidden_layer_1(combined_rep))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = self.out_layer(second_hidden)
        probs = F.sigmoid(out)
        predictions = torch.where(probs > 0.5, 1.0, 0.0)
        output_dict = {"probabilities": probs, "predictions": predictions}
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            correct = torch.sum(predictions == labels)
            total = float(predictions.shape[0])
            acc = correct / total
            if torch.sum(predictions).item() != 0:
                precision = torch.sum(
                    (predictions == labels).float() * predictions == 1) / (
                        torch.sum(predictions) + sys.float_info.epsilon)
            else:
                precision = torch.tensor(1.0).to(self.device)
            output_dict["accuracy"] = acc
            output_dict["precision"] = precision
            loss = loss_fct(out, labels)
            output_dict["loss"] = loss
        return output_dict


class CoreferenceCrossEncoder(nn.Module):
    def __init__(self, device):
        super(CoreferenceCrossEncoder, self).__init__()
        self.device = device
        self.pos_weight = torch.tensor([0.1]).to(device)
        self.model_type = 'CoreferenceCrossEncoder'
        self.mention_model = LongformerModel.from_pretrained('CDLM',
                                                             return_dict=True)
        self.mention_dim = 1536
        self.input_dim = self.mention_dim * 3
        self.out_dim = 1

        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer_1 = nn.Linear(self.input_dim, self.mention_dim)
        self.hidden_layer_2 = nn.Linear(self.mention_dim, self.mention_dim)
        self.out_layer = nn.Linear(self.mention_dim, self.out_dim)

    def get_sentence_vecs(self, sentences, start_pieces_1, end_pieces_1,
                          start_pieces_2, end_pieces_2):
        expected_transformer_input = self.to_transformer_input(
            sentences, start_pieces_1, end_pieces_1, start_pieces_2,
            end_pieces_2)
        transformer_output = self.mention_model(
            **expected_transformer_input).last_hidden_state
        return transformer_output

    def get_mention_rep(self, transformer_output, start_pieces, end_pieces):
        start_pieces = start_pieces.repeat(1, 768).view(-1, 1, 768)
        start_piece_vec = torch.gather(transformer_output, 1, start_pieces)
        end_piece_vec = torch.gather(
            transformer_output, 1,
            end_pieces.repeat(1, 768).view(-1, 1, 768))
        mention_rep = torch.cat([start_piece_vec, end_piece_vec],
                                dim=2).squeeze(1)
        return mention_rep

    def to_transformer_input(self, sentence_tokens, start_pieces_1,
                             end_pieces_1, start_pieces_2, end_pieces_2):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        start_1_attention_mask = (torch.arange(
            sentence_tokens.shape[1]).repeat(sentence_tokens.shape[0], 1).to(
                self.device) == start_pieces_1)
        end_1_attention_mask = (torch.arange(sentence_tokens.shape[1]).repeat(
            sentence_tokens.shape[0], 1).to(self.device) == end_pieces_1)
        start_2_attention_mask = (torch.arange(
            sentence_tokens.shape[1]).repeat(sentence_tokens.shape[0], 1).to(
                self.device) == start_pieces_2)
        end_2_attention_mask = (torch.arange(sentence_tokens.shape[1]).repeat(
            sentence_tokens.shape[0], 1).to(self.device) == end_pieces_2)
        global_attention_mask = start_1_attention_mask | end_1_attention_mask
        global_attention_mask = global_attention_mask | (
            start_2_attention_mask | end_2_attention_mask)
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask,
            "global_attention_mask": global_attention_mask
        }

    def forward(self,
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                labels=None):
        transformer_output = self.get_sentence_vecs(sentences, start_pieces_1,
                                                    end_pieces_1,
                                                    start_pieces_2,
                                                    end_pieces_2)
        mention_reps_1 = self.get_mention_rep(transformer_output,
                                              start_pieces_1, end_pieces_1)
        mention_reps_2 = self.get_mention_rep(transformer_output,
                                              start_pieces_2, end_pieces_2)
        combined_rep = torch.cat(
            [mention_reps_1, mention_reps_2, mention_reps_1 * mention_reps_2],
            dim=1)
        combined_rep = combined_rep
        first_hidden = F.relu(self.hidden_layer_1(combined_rep))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = self.out_layer(second_hidden)
        probs = F.sigmoid(out)
        predictions = torch.where(probs > 0.5, 1.0, 0.0)
        output_dict = {"probabilities": probs, "predictions": predictions}
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            correct = torch.sum(predictions == labels)
            total = float(predictions.shape[0])
            acc = correct / total
            if torch.sum(predictions).item() != 0:
                precision = torch.sum(
                    (predictions == labels).float() * predictions == 1) / (
                        torch.sum(predictions) + sys.float_info.epsilon)
            else:
                precision = torch.tensor(1.0).to(self.device)
            output_dict["accuracy"] = acc
            output_dict["precision"] = precision
            loss = loss_fct(out, labels)
            output_dict["loss"] = loss
        return output_dict
