import math
import torch
import torch.nn as nn
from model_utils import *
import torch.nn.functional as F
import torch.autograd as autograd


class TransformerCorefScorer(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, device, dropout=0.5):
        super(TransformerCorefScorer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.device = device
        self.max_len = 30
        self.nhid = nhid
        self.arg2_embedding = nn.Linear(ninp, ninp, bias=False)
        self.arg1_embedding = nn.Linear(ninp, ninp, bias=False)
        self.loc_embedding = nn.Linear(ninp, ninp, bias=False)
        self.tmp_embedding = nn.Linear(ninp, ninp, bias=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.seperator = nn.Parameter(torch.rand(nhid).view(1, 1, -1))
        self.seq_1 = nn.Parameter(torch.rand(nhid).view(1, 1, -1))
        self.seq_2 = nn.Parameter(torch.rand(nhid).view(1, 1, -1))
        self.hidden_dim_1 = int(nhid / 2)
        self.hidden_dim_2 = int(nhid / 2)
        self.hidden_layer_1 = nn.Linear(nhid, self.hidden_dim_1)
        self.hidden_layer_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.out_layer = nn.Linear(self.hidden_dim_2, 1)
        self.ninp = ninp

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def _transform_and_pack(self, mention, arg1, arg2, loc, tmp):
        arg1 = self.arg1_embedding(arg1)
        arg2 = self.arg2_embedding(arg2)
        loc = self.loc_embedding(loc)
        tmp = self.tmp_embedding(tmp)
        src = torch.cat([arg1, mention, arg2, loc, tmp], dim=1)
        _, re_order_index = ((src == 0).sum(2) == 0).type(torch.IntTensor).to(
            self.device).sort(1, descending=True)
        gather_map = re_order_index.view(-1, (self.max_len * 5),
                                         1).expand(-1, -1, self.nhid)
        src = torch.gather(src, 1, gather_map)
        return src

    def _pack_final(self, final_tensor):
        _, reorder_index = ((final_tensor == 0).sum(2) == 0).type(
            torch.IntTensor).to(self.device).sort(1, descending=True)
        gather_map = reorder_index.view(-1, (self.max_len * 10) + 1,
                                        1).expand(-1, -1, self.nhid)
        final_tensor = torch.gather(final_tensor, 1, gather_map)
        map_to_seperators = (reorder_index == (
            5 * self.max_len)).nonzero()[:, 1].view(-1, 1, 1).expand(
                -1, -1, self.nhid)
        return final_tensor, map_to_seperators

    def forward(self, src_mention_1, src_arg1_1, src_arg2_1, src_loc_1,
                src_tmp_1, src_mention_2, src_arg1_2, src_arg2_2, src_loc_2,
                src_tmp_2):

        src_1 = self._transform_and_pack(src_mention_1, src_arg1_1, src_arg2_1,
                                         src_loc_1, src_tmp_1) + self.seq_1

        src_2 = self._transform_and_pack(src_mention_2, src_arg1_2, src_arg2_2,
                                         src_loc_2, src_tmp_2) + self.seq_2

        seps = self.seperator.repeat(src_1.shape[0], 1, 1)
        src = torch.cat([src_1, seps, src_2], dim=1)
        src, map_to_seperators = self._pack_final(src)
        max_seq_in_batch = ((src == 0).sum(2) == 0).sum(1).max().detach()
        src = src[:, :max_seq_in_batch, :]
        output = self.transformer_encoder(src)
        sep_store = output.gather(1, map_to_seperators)
        sep_store = sep_store.view(-1, 768)
        first_hidden = F.relu(self.hidden_layer_1(sep_store))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = F.sigmoid(self.out_layer(second_hidden))
        return out


class CDCorefScorer(nn.Module):
    '''
    An abstract class represents a coreference pairwise scorer.
    Inherits Pytorch's Module class.
    '''
    def __init__(self, word_embeds, word_to_ix, vocab_size, char_embedding,
                 char_to_ix, char_rep_size, dims, use_mult, use_diff,
                 feature_size):
        '''
        C'tor for CorefScorer object
        :param word_embeds: pre-trained word embeddings
        :param word_to_ix: a mapping between a word (string) to
        its index in the word embeddings' lookup table
        :param vocab_size:  the vocabulary size
        :param char_embedding: initial character embeddings
        :param char_to_ix:  mapping between a character to
        its index in the character embeddings' lookup table
        :param char_rep_size: hidden size of the character LSTM
        :param dims: list holds the layer dimensions
        :param use_mult: a boolean indicates whether to use element-wise multiplication in the
        input layer
        :param use_diff: a boolean indicates whether to use element-wise differentiation in the
        input layer
        :param feature_size: embeddings size of binary features


        '''
        super(CDCorefScorer, self).__init__()
        self.embed = nn.Embedding(vocab_size, word_embeds.shape[1])

        self.embed.weight.data.copy_(torch.from_numpy(word_embeds))
        self.embed.weight.requires_grad = False  # pre-trained word embeddings are fixed
        self.word_to_ix = word_to_ix

        self.char_embeddings = nn.Embedding(len(char_to_ix.keys()),
                                            char_embedding.shape[1])
        self.char_embeddings.weight.data.copy_(
            torch.from_numpy(char_embedding))
        self.char_embeddings.weight.requires_grad = True
        self.char_to_ix = char_to_ix
        self.embedding_dim = word_embeds.shape[1]
        self.char_hidden_dim = char_rep_size

        self.char_lstm = nn.LSTM(input_size=char_embedding.shape[1],
                                 hidden_size=self.char_hidden_dim,
                                 num_layers=1,
                                 bidirectional=False)

        # binary features for coreferring arguments/predicates
        self.coref_role_embeds = nn.Embedding(2, feature_size)

        self.use_mult = use_mult
        self.use_diff = use_diff
        self.input_dim = dims[0]
        self.hidden_dim_1 = dims[1]
        self.hidden_dim_2 = dims[2]
        self.out_dim = 1

        self.hidden_layer_1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.hidden_layer_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.out_layer = nn.Linear(self.hidden_dim_2, self.out_dim)

        self.model_type = 'CD_scorer'

    def forward(self, clusters_pair_tensor):
        '''
        The forward method - pass the input tensor through a feed-forward neural network
        :param clusters_pair_tensor: an input tensor consists of a concatenation between
        two mention representations, their element-wise multiplication and a vector of binary features
        (each feature embedded as 50 dimensional embeddings)
        :return: a predicted confidence score (between 0 to 1) of the mention pair to be in the
        same coreference chain (aka cluster).
        '''
        first_hidden = F.relu(self.hidden_layer_1(clusters_pair_tensor))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = F.sigmoid(self.out_layer(second_hidden))

        return out

    def init_char_hidden(self, device):
        '''
        initializes hidden states the character LSTM
        :param device: gpu/cpu Pytorch device
        :return: initialized hidden states (tensors)
        '''
        return (torch.randn((1, 1, self.char_hidden_dim),
                            requires_grad=True).to(device),
                torch.randn((1, 1, self.char_hidden_dim),
                            requires_grad=True).to(device))

    def get_char_embeds(self, seq, device):
        '''
        Runs a LSTM on a list of character embeddings and returns the last output state
        :param seq: a list of character embeddings
        :param device:  gpu/cpu Pytorch device
        :return: the LSTM's last output state
        '''
        char_hidden = self.init_char_hidden(device)
        input_char_seq = self.prepare_chars_seq(seq, device)
        char_embeds = self.char_embeddings(input_char_seq).view(
            len(seq), 1, -1)
        char_lstm_out, char_hidden = self.char_lstm(char_embeds, char_hidden)
        char_vec = char_lstm_out[-1]

        return char_vec

    def prepare_chars_seq(self, seq, device):
        '''
        Given a string represents a word or a phrase, this method converts the sequence
        to a list of character embeddings
        :param seq: a string represents a word or a phrase
        :param device: device:  gpu/cpu Pytorch device
        :return: a list of character embeddings
        '''
        idxs = []
        for w in seq:
            if w in self.char_to_ix:
                idxs.append(self.char_to_ix[w])
            else:
                lower_w = w.lower()
                if lower_w in self.char_to_ix:
                    idxs.append(self.char_to_ix[lower_w])
                else:
                    idxs.append(self.char_to_ix['<UNK>'])
                    print('can find char {}'.format(w))
        tensor = torch.tensor(idxs, dtype=torch.long).to(device)

        return tensor
