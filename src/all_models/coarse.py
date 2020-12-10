import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from transformers import RobertaModel
from tqdm import tqdm, trange


class EncoderCosineRanker(nn.Module):
    def __init__(self, device):
        super(EncoderCosineRanker, self).__init__()
        self.device = device
        self.model_type = 'EncoderCosineRanker'
        self.mention_model = RobertaModel.from_pretrained('roberta-base',
                                                          return_dict=True)
        self.cluster_lookup = {}

    def update_cluster_lookup(self, label_sets):
        self.cluster_lookup = {}
        with torch.no_grad():
            for label_id, label_exemplar in tqdm(label_sets.items(),
                                                 desc="Exemplar Reps"):
                assert (label_exemplar["label"] == [label_id])
                sentences = torch.tensor(
                    label_exemplar["sentence"]).unsqueeze(1).permute(1, 0).to(
                        self.device)
                transformer_output = self.get_sentence_vecs(sentences)
                start = torch.tensor(
                    label_exemplar["start_piece"]).unsqueeze(1).to(self.device)
                end = torch.tensor(
                    label_exemplar["end_piece"]).unsqueeze(1).to(self.device)
                mention_rep = self.get_mention_rep(transformer_output, start,
                                                   end)
                self.cluster_lookup[label_id] = mention_rep

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
        mask = sentence_tokens != 0
        return {
            "input_ids": sentence_tokens,
            "token_type_ids": segment_idx,
            "attention_mask": mask
        }

    def convert_labels_to_reps(self, label):
        return self.cluster_lookup[label[0]]

    def forward(self, sentences, start_pieces, end_pieces, labels):
        exemplars = list(
            map(self.convert_labels_to_reps,
                labels.cpu().tolist()))
        exemplars = torch.cat(exemplars, dim=0)
        transformer_output = self.get_sentence_vecs(sentences)
        mention_reps = self.get_mention_rep(transformer_output, start_pieces,
                                            end_pieces)
        exemplars = exemplars.squeeze(1)
        mention_reps = mention_reps.squeeze(1)
        scores = torch.mm(exemplars, mention_reps.t()).view(-1, 1)

        pairwise_labels = (labels == labels.t()).view(-1, 1).type(
            torch.FloatTensor).to(self.device)

        loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
        loss = loss_fct(scores, pairwise_labels)

        return loss, scores
