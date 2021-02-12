import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import faiss
from transformers import RobertaModel
from tqdm import tqdm, trange


def tokenize_and_map(sentences, tokenizer, mention_sentence=0):
    max_seq_length = 512
    mapping = {}
    raw_strings = []
    offset = 0
    for i, sentence in enumerate(sentences):
        raw_strings.append(' '.join(
            [tok.replace(" ", "") for tok in sentence.get_tokens_strings()]))
        if i == mention_sentence:
            mention_offset = offset
        for _ in sentence.get_tokens_strings():
            mapping[offset] = []
            offset += 1
    embeddings = tokenizer(' '.join(raw_strings),
                           max_length=max_seq_length,
                           truncation=True,
                           padding="max_length")["input_ids"]
    counter = 0
    for i, token in enumerate(tokenizer.convert_ids_to_tokens(embeddings)):
        if token == "<s>" or token == "</s>" or token == "<pad>":
            continue
        elif token[0] == "Ġ":
            counter += 1
            mapping[counter].append(i)
        else:
            mapping[counter].append(i)
            continue
    return embeddings, mapping, mention_offset


class EncoderCosineRanker(nn.Module):
    def __init__(self, device):
        super(EncoderCosineRanker, self).__init__()
        self.device = device
        self.model_type = 'EncoderCosineRanker'
        self.mention_model = RobertaModel.from_pretrained('roberta-large',
                                                          return_dict=True)
        self.cluster_lookup = {}
        self.faiss_index = {}

    def update_cluster_lookup(self, label_sets, dev=False):
        cluster_lookup = {}
        del self.cluster_lookup
        self.cluster_lookup = {}
        index = faiss.IndexFlatIP(1536)
        with torch.no_grad():
            for label_id, label_records in tqdm(label_sets.items(),
                                                desc="Exemplar Reps"):
                label_records = label_records[:50]
                assert (label_records[0]["label"] == [label_id])
                sentences = torch.tensor(
                    [record["sentence"] for record in label_records])
                start_pieces = torch.tensor(
                    [record["start_piece"] for record in label_records])
                end_pieces = torch.tensor(
                    [record["end_piece"] for record in label_records])
                sentences = torch.tensor(sentences).to(self.device)
                transformer_output = self.get_sentence_vecs(sentences)
                start = torch.tensor(start_pieces).to(self.device)
                end = torch.tensor(end_pieces).to(self.device)
                mention_rep = self.get_mention_rep(transformer_output, start,
                                                   end)
                mention_rep = mention_rep.mean(dim=0)
                cluster_lookup[label_id] = mention_rep
                index.add(mention_rep.cpu().numpy())
        self.cluster_lookup = cluster_lookup
        self.faiss_index = index

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

        mention_rep = torch.cat([start_piece_vec, end_piece_vec], dim=2)
        return mention_rep

    def to_transformer_input(self, sentence_tokens):
        segment_idx = sentence_tokens * 0
        mask = sentence_tokens != 1
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask
        }

    def convert_labels_to_reps(self, label):
        return self.cluster_lookup[label]

    def get_hard_cases(self, mention_reps, labels):
        mention_reps = mention_reps.squeeze(1).detach().cpu().numpy()
        _, hard_case_lists = self.faiss_index.search(mention_reps, 10)
        hard_cases = []
        for i, h_list in enumerate(hard_case_lists):
            for j, hard_case in enumerate(h_list):
                #if labels[i] == hard_case:
                #    tqdm.write(str(j))
                #    break
                hard_cases.append(hard_case)
        return torch.tensor(hard_cases).to(self.device).unsqueeze(1)

    def forward(self, sentences, start_pieces, end_pieces, labels, dev=False):
        transformer_output = self.get_sentence_vecs(sentences)
        mention_reps = self.get_mention_rep(transformer_output, start_pieces,
                                            end_pieces)
        hard_cases = self.get_hard_cases(mention_reps, labels)
        labels_with_hard_neg = torch.cat([labels, hard_cases], dim=0)
        unique_clusters, local_labels = labels_with_hard_neg.unique(
            return_inverse=True)
        local_labels = local_labels[:len(labels)].squeeze(1)
        exemplars = list(
            map(self.convert_labels_to_reps,
                unique_clusters.cpu().tolist()))
        exemplars = torch.cat(exemplars, dim=0)
        exemplars = exemplars
        mention_reps = mention_reps.squeeze(1)
        scores = torch.mm(mention_reps, exemplars.t())
        if not self.training:
            return {"logits": logits}

        predictions = scores.argmax(dim=1)
        correct = torch.sum(predictions == local_labels)
        total = float(predictions.shape[0])
        acc = correct / total
        loss_fct = nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(scores, local_labels)
        return {"logits": scores, "loss": loss, "accuracy": acc}
