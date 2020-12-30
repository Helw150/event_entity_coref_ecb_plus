import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from transformers import RobertaModel


def tokenize_and_map_pair(sentence_1, sentence_2, tokenizer):
    max_seq_length = 512
    embeddings = tokenizer(sentence_1.get_raw_sentence(),
                           sentence_2.get_raw_sentence(),
                           max_length=256,
                           truncation=True,
                           padding="max_length")["input_ids"]
    counter = 0
    mapping = {k: [] for k, _ in enumerate(sentence_1.get_tokens_strings())}
    offset = len(sentence_1.get_tokens_strings())
    for k, _ in enumerate(sentence_2.get_tokens_strings()):
        mapping[offset + k] = []
    new_tokens = tokenizer.convert_ids_to_tokens(embeddings)
    for i, token in enumerate(new_tokens):
        if token == "<s>" or token == "</s>" or token == "<pad>":
            continue
        elif token[0] == "Ġ" or new_tokens[i - 1] == "</s>":
            counter += 1
            mapping[counter].append(i)
        else:
            mapping[counter].append(i)
            continue
    return embeddings, mapping, offset


class CoreferenceCrossEncoder(nn.Module):
    def __init__(self, device):
        super(CoreferenceCrossEncoder, self).__init__()
        self.device = device
        self.pos_weight = torch.tensor([0.001]).to(device)
        self.model_type = 'CoreferenceCrossEncoder'
        self.mention_model = RobertaModel.from_pretrained('roberta-base',
                                                          return_dict=True)
        self.mention_dim = 1536
        self.input_dim = self.mention_dim * 3
        self.out_dim = 1

        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer_1 = nn.Linear(self.input_dim, self.mention_dim)
        self.hidden_layer_2 = nn.Linear(self.mention_dim, self.mention_dim)
        self.out_layer = nn.Linear(self.mention_dim, self.out_dim)

    def get_sentence_vecs(self, sentences):
        expected_transformer_input = self.to_transformer_input(sentences)
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

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask
        }

    def forward(self,
                sentences,
                start_pieces_1,
                end_pieces_1,
                start_pieces_2,
                end_pieces_2,
                labels=None):
        transformer_output = self.get_sentence_vecs(sentences)
        mention_reps_1 = self.get_mention_rep(transformer_output,
                                              start_pieces_1, end_pieces_1)
        mention_reps_2 = self.get_mention_rep(transformer_output,
                                              start_pieces_2, end_pieces_2)
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
