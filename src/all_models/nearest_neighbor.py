import argparse
from coarse import *
import faiss
import numpy as np
from scipy import stats
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


def dataset_to_docs(dataset):
    docs = [
        document for topic in dataset.topics.values()
        for document in topic.docs.values()
    ]
    return docs


def generate_singleton_set(docs):
    clusters = {}
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            for mention in sentences[sentence_id].gold_event_mentions:
                if mention.gold_tag not in clusters:
                    clusters[mention.gold_tag] = [mention.mention_id]
                else:
                    clusters[mention.gold_tag].append(mention.mention_id)
    singletons = [
        mention for cluster_id in clusters for mention in clusters[cluster_id]
        if len(clusters[cluster_id]) == 1
    ]
    return set(singletons)


def build_mention_reps(docs, model, events=True, remove_singletons=False):
    processed_dataset = []
    labels = []
    mentions = []
    label_vocab_size = 0
    singleton_set = set()
    if remove_singletons:
        singleton_set = generate_singleton_set(docs)
    print(len(singleton_set))
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            sentence = sentences[sentence_id]
            sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
            if len(sentence_mentions) == 0:
                continue
            lookback = max(0, sentence_id - 5)
            lookforward = min(sentence_id + 5, max(sentences.keys())) + 1
            tokenization_input = ([
                sentences[_id] for _id in range(lookback, lookforward)
            ], sentence_id - lookback)
            tokenized_sentence, tokenization_mapping, sent_offset = tokenize_and_map(
                tokenization_input[0], tokenizer, tokenization_input[1])
            sentence_vec = model.get_sentence_vecs(
                torch.tensor([tokenized_sentence]).to(model.device))
            for mention in sentence_mentions:
                if remove_singletons and mention in singleton_set:
                    continue
                start_piece = torch.tensor([[
                    tokenization_mapping[sent_offset + mention.start_offset][0]
                ]])
                end_piece = torch.tensor([[
                    tokenization_mapping[sent_offset + mention.end_offset][-1]
                ]])
                mention_rep = model.get_mention_rep(
                    sentence_vec, start_piece.to(model.device),
                    end_piece.to(model.device))
                processed_dataset.append(mention_rep.detach().cpu().numpy()[0])
                labels.append((mention.mention_str, mention.gold_tag))
                mentions.append(mention)

    return np.concatenate(processed_dataset, axis=0), labels, mentions


def build_cluster_rep(cluster, model, docs):
    cluster_rep = []
    for mention in cluster.mentions.values():
        sentence = docs[mention.doc_id].get_sentences()[mention.sent_id]
        tokenized_sentence, tokenization_mapping, sent_offset = tokenize_and_map(
            [sentence], tokenizer, 0)
        sent_rep = model.get_sentence_vecs(
            torch.tensor([tokenized_sentence]).to(model.device))
        start_piece = torch.tensor(
            [[tokenization_mapping[mention.start_offset][0]]])
        end_piece = torch.tensor(
            [[tokenization_mapping[mention.end_offset][-1]]])
        mention_rep = model.get_mention_rep(sent_rep,
                                            start_piece.to(model.device),
                                            end_piece.to(model.device))
        cluster_rep.append(mention_rep)
    return torch.cat(cluster_rep, dim=0).mean(dim=0).detach().cpu().numpy()


def build_cluster_reps(clusters, model, docs):
    cluster_reps = []
    sent_reps = {}
    for cluster in clusters:
        cluster_rep = build_cluster_rep(cluster, model, docs)
        cluster_reps.append(cluster_rep)
    return np.concatenate(cluster_reps, axis=0)


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


def nn_cluster_pairs(clusters, model, docs, k=10):
    with torch.no_grad():
        vectors = build_cluster_reps(clusters, model, docs)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    D, I = index.search(vectors, k + 1)
    pairs = []
    for i, cluster in enumerate(clusters):
        nearest_neighbor_indexes = I[i][1:]
        nearest_neighbors = [(cluster, clusters[j])
                             for j in nearest_neighbor_indexes]
        pairs.extend(nearest_neighbors)
    return pairs


def create_cluster_index(clusters, model, docs):
    with torch.no_grad():
        vectors = build_cluster_reps(clusters, model, docs)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    return index


def create_mention_index(docs, model):
    with torch.no_grad():
        vectors = build_mention_reps(docs, model)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    return index


def nn_generate_pairs(data, model, k=10, events=True, remove_singletons=False):
    vectors, labels, mentions = build_mention_reps(
        data, model, events, remove_singletons=remove_singletons)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    D, I = index.search(vectors, k + 1)
    pairs = set()
    for i, mention in enumerate(mentions):
        nearest_neighbor_indexes = I[i]
        nearest_neighbors = set()
        for nn_index, j in enumerate(nearest_neighbor_indexes):
            if mention.mention_id != mentions[j].mention_id:
                nearest_neighbors.add(frozenset([mention, mentions[j]]))
        pairs = pairs | nearest_neighbors
    return pairs


def nn_eval(eval_data, model, k=5):
    vectors, labels, _ = build_mention_reps(dataset_to_docs(eval_data), model)
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
        model = EncoderCosineRanker("cuda:0")
        model.load_state_dict(params)
        model.eval()
    model.device = torch.device("cuda:0" if args.use_cuda else "cpu")
    model = model.to(model.device)
    recall, mrr, maP, mean_precision_k = nn_eval(eval_data, model)
    tqdm.write(
        "Recall: {:.6f} - MRR: {:.6f} - MAP: {:.6f} - Mean Precision @ K: {:.6f}"
        .format(recall, mrr, maP, mean_precision_k))
