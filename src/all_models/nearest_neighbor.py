import argparse
from coarse import *
import faiss
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import _pickle as cPickle
from transformers import RobertaTokenizer

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))
from classes import *

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def build_mention_reps(dataset, model, events=True):
    processed_dataset = []
    labels = []
    label_vocab_size = 0
    docs = [
        document for topic in dataset.topics.values()
        for document in topic.docs.values()
    ]
    for doc in docs:
        for sentence in doc.get_sentences().values():
            tokenized_sentence, tokenization_mapping = tokenize_and_map(
                sentence, tokenizer)
            sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
            sentence_vec = model.get_sentence_vecs(
                torch.tensor([tokenized_sentence]).to(model.device))
            for mention in sentence_mentions:
                start_piece = torch.tensor(
                    [[tokenization_mapping[mention.start_offset][0]]])
                end_piece = torch.tensor(
                    [[tokenization_mapping[mention.end_offset][-1]]])
                mention_rep = model.get_mention_rep(
                    sentence_vec, start_piece.to(model.device),
                    end_piece.to(model.device))
                processed_dataset.append(mention_rep.detach().cpu().numpy()[0])
                labels.append((mention.mention_str, mention.gold_tag))

    return np.concatenate(processed_dataset, axis=0), labels


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def nn_eval(eval_data, model, k=200):
    vectors, labels = build_mention_reps(eval_data, model)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    # Add 1 since the first will be identity
    D, I = index.search(vectors, k + 1)
    relevance_matrix = []
    tp = 0
    precision = 0
    singletons = 0
    for results in [[labels[i] for i in row] for row in I]:
        original_str, true_label = results[0]
        if "Singleton" in true_label:
            singletons += 1
            continue
        matches = results[1:]
        relevance = [label == true_label for _, label in matches]
        num_correct = np.sum(relevance)
        precision += num_correct / k
        if num_correct >= 1:
            tp += 1
        relevance_matrix.append(relevance)
    return (tp / float(len(I) - singletons),
            mean_reciprocal_rank(relevance_matrix),
            mean_average_precision(relevance_matrix),
            precision / float(len(I) - singletons))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Doing a Nearest Neighbor Eval of Dense Space Encoder')

    parser.add_argument('--dataset', type=str, help='Dataset to Evaluate on')
    parser.add_argument('--model', type=str, help='Model to Evaluate')
    parser.add_argument('--cuda',
                        dest='use_cuda',
                        action='store_true',
                        help='Use CUDA/GPU for prediction')
    parser.set_defaults(use_cuda=False)

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    with open(args.dataset, 'rb') as f:
        eval_data = cPickle.load(f)
    with open(args.model, 'rb') as f:
        params = torch.load(f)
        model = EncoderCosineRanker(params)
    model.device = torch.device("cuda:0" if args.use_cuda else "cpu")
    model = model.to(model.device)
    recall, mrr, maP, mean_precision_k = nn_eval(eval_data, model)
    tqdm.write(
        "Recall: {:.6f} - MRR: {:.6f} - MAP: {:.6f} - Mean Precision @ K: {:.6f}"
        .format(recall, mrr, maP, mean_precision_k))
