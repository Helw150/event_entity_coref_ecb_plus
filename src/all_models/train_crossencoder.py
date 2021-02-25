import os
import sys
import json
import random
import logging
import argparse
import traceback
from collections import defaultdict
import numpy as np
from tqdm import tqdm, trange
from scorer import *
from nearest_neighbor import nn_generate_pairs, dataset_to_docs
import _pickle as cPickle
from graphviz import Graph
import networkx as nx

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

parser = argparse.ArgumentParser(description='Training a regressor')

parser.add_argument('--config_path',
                    type=str,
                    help=' The path configuration json file')
parser.add_argument('--out_dir',
                    type=str,
                    help=' The directory to the output folder')
parser.add_argument('--eval',
                    dest='evaluate_dev',
                    action='store_true',
                    help='evaluate_dev')
parser.add_argument('--cont',
                    dest='continue_training',
                    action='store_true',
                    help='Continue Training From Checkpoint')

args = parser.parse_args()

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logging.basicConfig(filename=os.path.join(args.out_dir, "train_log.txt"),
                    level=logging.DEBUG,
                    filemode='w')

# Load json config file
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

with open(os.path.join(args.out_dir, 'train_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

if config_dict["gpu_num"] != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict["gpu_num"])
    args.use_cuda = True
else:
    args.use_cuda = False

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.optim as optim

args.use_cuda = args.use_cuda and torch.cuda.is_available()

from classes import *
from model_utils import load_entity_wd_clusters
from bcubed_scorer import *
from coarse import *
from fine import *
from transformers import DebertaTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

# Fix the random seeds
seed = config_dict["random_seed"]
random.seed(seed)
np.random.seed(seed)
if args.use_cuda:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Training with CUDA')

tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-large")
best_score = None
patience = 0
comparison_set = set()


def all_positives(docs, events):
    gold_clusters = defaultdict(lambda: [])
    for doc in docs:
        for sentence in doc.get_sentences().values():
            sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
            for mention in sentence_mentions:
                gold_clusters[mention.gold_tag].append(mention)
    pairs = set()
    for value in gold_clusters.values():
        for i, mention_1 in enumerate(value):
            for mention_2 in value[i + 1:]:
                pairs.add(frozenset([mention_1, mention_2]))
    return pairs


def wd_comparisons(docs, events):
    pairs = set()
    for doc in docs:
        doc_mentions = []
        for sentence in doc.get_sentences().values():
            sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
            doc_mentions.extend(sentence_mentions)
        for mention_1 in doc_mentions:
            for mention_2 in doc_mentions:
                if not mention_1.mention_id == mention_2.mention_id:
                    pairs.add(frozenset([mention_1, mention_2]))
    return pairs


def get_sents(sentences, sentence_id, window=config_dict["window_size"]):
    lookback = max(0, sentence_id - window)
    lookforward = min(sentence_id + window, max(sentences.keys())) + 1
    return ([sentences[_id]
             for _id in range(lookback, lookforward)], sentence_id - lookback)


def structure_pair(mention_1,
                   mention_2,
                   doc_dict,
                   window=config_dict["window_size"]):
    try:
        sents_1, sent_id_1 = get_sents(doc_dict[mention_1.doc_id].sentences,
                                       mention_1.sent_id, window)
        sents_2, sent_id_2 = get_sents(doc_dict[mention_2.doc_id].sentences,
                                       mention_2.sent_id, window)
        tokens, segment_idx, token_map, offset_1, offset_2 = tokenize_and_map_pair(
            sents_1, sents_2, sent_id_1, sent_id_2, tokenizer)
        start_piece_1 = token_map[offset_1 + mention_1.start_offset][0]
        if offset_1 + mention_1.end_offset in token_map:
            end_piece_1 = token_map[offset_1 + mention_1.end_offset][-1]
        else:
            end_piece_1 = token_map[offset_1 + mention_1.start_offset][-1]
        start_piece_2 = token_map[offset_2 + mention_2.start_offset][0]
        if offset_2 + mention_2.end_offset in token_map:
            end_piece_2 = token_map[offset_2 + mention_2.end_offset][-1]
        else:
            end_piece_2 = token_map[offset_2 + mention_2.start_offset][-1]
        label = [1.0] if mention_1.gold_tag == mention_2.gold_tag else [0.0]
        record = {
            "sentence": tokens,
            "label": label,
            "start_piece_1": [start_piece_1],
            "end_piece_1": [end_piece_1],
            "start_piece_2": [start_piece_2],
            "end_piece_2": [end_piece_2],
            "segment_idx": [segment_idx]
        }
    except:
        if window > 0:
            return structure_pair(mention_1, mention_2, doc_dict, window - 1)
        else:
            traceback.print_exc()
            sys.exit()
    return record


def structure_dataset(data_set,
                      encoder_model,
                      events=True,
                      k=10,
                      is_train=False):
    processed_dataset = []
    doc_dict = {
        key: document
        for topic in data_set.topics.values()
        for key, document in topic.docs.items()
    }
    docs = dataset_to_docs(data_set)
    if is_train:
        docs = docs[:int(len(docs) * 1)]
    pairs = nn_generate_pairs(
        docs,
        encoder_model,
        k=k,
        events=events,
        remove_singletons=config_dict["remove_singletons"])
    if config_dict["all_positives"] and is_train:
        pairs = pairs | all_positives(docs, events)
    if config_dict["add_wd_pairs"]:
        pairs = pairs | wd_comparisons(docs, events)
    pairs = list(pairs)
    for mention_1, mention_2 in pairs:
        record = structure_pair(mention_1, mention_2, doc_dict)
        processed_dataset.append(record)
    sentences = torch.tensor(
        [record["sentence"] for record in processed_dataset])
    labels = torch.tensor([record["label"] for record in processed_dataset])
    start_pieces_1 = torch.tensor(
        [record["start_piece_1"] for record in processed_dataset])
    end_pieces_1 = torch.tensor(
        [record["end_piece_1"] for record in processed_dataset])
    start_pieces_2 = torch.tensor(
        [record["start_piece_2"] for record in processed_dataset])
    end_pieces_2 = torch.tensor(
        [record["end_piece_2"] for record in processed_dataset])
    segment_idxs = torch.tensor(
        [record["segment_idx"] for record in processed_dataset])
    print(labels.sum() / float(labels.shape[0]))
    return TensorDataset(sentences, segment_idxs, start_pieces_1, end_pieces_1,
                         start_pieces_2, end_pieces_2, labels), pairs, doc_dict


def get_optimizer(model):
    lr = config_dict["lr"]
    optimizer = None
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if config_dict["optimizer"] == 'adadelta':
        optimizer = optim.Adadelta(parameters,
                                   lr=lr,
                                   weight_decay=config_dict["weight_decay"])
    elif config_dict["optimizer"] == 'adam':
        optimizer = optim.Adam(parameters,
                               lr=lr,
                               weight_decay=config_dict["weight_decay"])
    elif config_dict["optimizer"] == 'sgd':
        optimizer = optim.SGD(parameters,
                              lr=lr,
                              momentum=config_dict["momentum"],
                              nesterov=True)

    assert (optimizer is not None), "Config error, check the optimizer field"

    return optimizer


def get_scheduler(optimizer, len_train_data):
    batch_size = config_dict["accumulated_batch_size"]
    epochs = config_dict["epochs"]

    num_train_steps = int(len_train_data / batch_size) * epochs
    num_warmup_steps = int(num_train_steps * config_dict["warmup_proportion"])

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                                num_train_steps)
    return scheduler


def find_cluster_key(node, clusters):
    if node in clusters:
        return node
    for key, value in clusters.items():
        if node in value:
            return key
    return None


def is_cluster_merge(cluster_1, cluster_2, mentions, model, doc_dict):
    if config_dict["oracle"]:
        return True
    score = 0.0
    sample_size = 20
    global comparison_set
    if len(cluster_1) > sample_size:
        c_1 = random.sample(cluster_1, sample_size)
    else:
        c_1 = cluster_1
    if len(cluster_2) > sample_size:
        c_2 = random.sample(cluster_2, sample_size)
    else:
        c_2 = cluster_2
    for mention_id_1 in c_1:
        records = []
        mention_1 = mentions[mention_id_1]
        for mention_id_2 in c_2:
            comparison_set = comparison_set | set(
                [frozenset([mention_id_1, mention_id_2])])
            mention_2 = mentions[mention_id_2]
            record = structure_pair(mention_1, mention_2, doc_dict)
            records.append(record)
        sentences = torch.tensor([record["sentence"]
                                  for record in records]).to(model.device)
        labels = torch.tensor([record["label"]
                               for record in records]).to(model.device)
        start_pieces_1 = torch.tensor(
            [record["start_piece_1"] for record in records]).to(model.device)
        end_pieces_1 = torch.tensor(
            [record["end_piece_1"] for record in records]).to(model.device)
        start_pieces_2 = torch.tensor(
            [record["start_piece_2"] for record in records]).to(model.device)
        end_pieces_2 = torch.tensor(
            [record["end_piece_2"] for record in records]).to(model.device)
        with torch.no_grad():
            out_dict = model(sentences, start_pieces_1, end_pieces_1,
                             start_pieces_2, end_pieces_2, labels)
            mean_prob = torch.mean(out_dict["probabilities"]).item()
            score += mean_prob
    return (score / len(c_1)) >= 0.5


def transitive_closure_merge(edges, mentions, model, doc_dict, graph,
                             graph_render):
    clusters = {}
    inv_clusters = {}
    mentions = {mention.mention_id: mention for mention in mentions}
    for edge in tqdm(edges):
        cluster_key = find_cluster_key(edge[0], clusters)
        alt_key = find_cluster_key(edge[1], clusters)
        if cluster_key == None and alt_key == None:
            cluster_key = edge[0]
            clusters[cluster_key] = set()
        elif cluster_key == None and alt_key != None:
            cluster_key = alt_key
            alt_key = None
        elif cluster_key == alt_key:
            alt_key = None
        # If alt_key exists, merge clusters
        perform_merge = True
        if alt_key:
            perform_merge = is_cluster_merge(clusters[cluster_key],
                                             clusters[alt_key], mentions,
                                             model, doc_dict)
        elif clusters[cluster_key] != set():
            new_elements = set([edge[0], edge[1]]) - clusters[cluster_key]
            if len(new_elements) > 0:
                perform_merge = is_cluster_merge(clusters[cluster_key],
                                                 new_elements, mentions, model,
                                                 doc_dict)
        if alt_key and perform_merge:
            clusters[cluster_key] = clusters[cluster_key] | clusters[alt_key]
            for node in clusters[alt_key]:
                inv_clusters[node] = cluster_key
            del clusters[alt_key]
        if perform_merge:
            if not (graph.has_edge(edge[0], edge[1])
                    or graph.has_edge(edge[1], edge[0])):
                graph.add_edge(edge[0], edge[1])
                color = 'black'
                if edge[2] != 1.0:
                    color = 'red'
                graph_render.edge(edge[0],
                                  edge[1],
                                  color=color,
                                  label=str(edge[3]))
            cluster = clusters[cluster_key]
            cluster.add(edge[0])
            cluster.add(edge[1])
            inv_clusters[edge[0]] = cluster_key
            inv_clusters[edge[1]] = cluster_key
    print(len(comparison_set))
    return clusters, inv_clusters


def evaluate(model, encoder_model, dev_dataloader, dev_pairs, doc_dict,
             epoch_num):
    global best_score, comparison_set
    model = model.eval()
    offset = 0
    edges = set()
    best_edges = {}
    mentions = set()
    acc_sum = 0.0
    all_probs = []
    for step, batch in enumerate(tqdm(dev_dataloader, desc="Test Batch")):
        batch = tuple(t.to(model.device) for t in batch)
        sentences, segment_idxs, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = batch
        if not config_dict["oracle"]:
            with torch.no_grad():
                out_dict = model(sentences, start_pieces_1, end_pieces_1,
                                 start_pieces_2, end_pieces_2, segment_idxs,
                                 labels)
        else:
            out_dict = {
                "accuracy": 1.0,
                "predictions": labels,
                "probabilities": labels
            }
        acc_sum += out_dict["accuracy"]
        predictions = out_dict["predictions"].detach().cpu().tolist()
        probs = out_dict["probabilities"].detach().cpu().tolist()
        for p_index in range(len(predictions)):
            pair_0, pair_1 = dev_pairs[offset + p_index]
            prediction = predictions[p_index]
            mentions.add(pair_0)
            mentions.add(pair_1)
            comparison_set = comparison_set | set(
                [frozenset([pair_0.mention_id, pair_1.mention_id])])
            if probs[p_index][0] > 0.5:
                if pair_0.mention_id not in best_edges or (
                        probs[p_index][0] > best_edges[pair_0.mention_id][3]):
                    best_edges[pair_0.mention_id] = (pair_0.mention_id,
                                                     pair_1.mention_id,
                                                     labels[p_index][0],
                                                     probs[p_index][0])
                edges.add((pair_0.mention_id, pair_1.mention_id,
                           labels[p_index][0], probs[p_index][0]))
        #for item in best_edges:
        #    edges.add(best_edges[item])

        offset += len(predictions)

    tqdm.write("Pairwise Accuracy: {:.6f}".format(acc_sum /
                                                  float(len(dev_dataloader))))
    eval_edges(edges, mentions, model, doc_dict)


def eval_edges(edges, mentions, model, doc_dict):
    print(len(mentions))
    global best_score, patience
    dot = Graph(comment='Cross Doc Co-ref')
    G = nx.Graph()
    edges = sorted(edges, key=lambda x: -1 * x[3])
    for mention in mentions:
        G.add_node(mention.mention_id)
        dot.node(mention.mention_id,
                 label=str((str(mention), doc_dict[mention.doc_id].sentences[
                     mention.sent_id].get_raw_sentence())))
    bridges = list(nx.bridges(G))
    articulation_points = list(nx.articulation_points(G))
    #edges = [edge for edge in edges if edge not in bridges]
    clusters, inv_clusters = transitive_closure_merge(edges, mentions, model,
                                                      doc_dict, G, dot)

    # Find Transitive Closure Clusters
    gold_sets = []
    model_sets = []
    for mention in mentions:
        gold_sets.append(mention.gold_tag)
        if mention.mention_id in inv_clusters:
            model_sets.append(inv_clusters[mention.mention_id])
        else:
            model_sets.append(mention.mention_id)
    p, r, f1 = bcubed(gold_sets, model_sets)
    tqdm.write("Recall: {:.6f} Precision: {:.6f} F1: {:.6f}".format(p, r, f1))
    if best_score == None or f1 > best_score:
        tqdm.write("F1 Improved Saving Model")
        best_score = f1
        patience = 0
        if not args.evaluate_dev:
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, 'best_model'))
        else:
            dot.render('/home/ubuntu/clustering')
    else:
        patience += 1
        if patience > config_dict["early_stop_patience"]:
            print("Early Stopping")
            sys.exit()


def train_model(train_set, dev_set):
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    with open(config_dict["event_encoder_model"], 'rb') as f:
        params = torch.load(f)
        event_encoder = EncoderCosineRanker(device)
        event_encoder.load_state_dict(params)
        event_encoder = event_encoder.to(device).eval()
        event_encoder.requires_grad = False
    model = CoreferenceCrossEncoder(device).to(device)
    if args.continue_training:
        print("Loading Model from Checkpoint...")
        with open(os.path.join(args.out_dir, 'best_model'), 'rb') as f:
            params = torch.load(f)
            model.load_state_dict(params)
    train_event_pairs, _, _ = structure_dataset(train_set,
                                                event_encoder,
                                                events=config_dict["events"],
                                                k=5,
                                                is_train=True)
    dev_event_pairs, dev_pairs, dev_docs = structure_dataset(
        dev_set, event_encoder, events=config_dict["events"], k=5)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, len(train_event_pairs))
    train_sampler = SequentialSampler(train_event_pairs)
    train_dataloader = DataLoader(train_event_pairs,
                                  sampler=train_sampler,
                                  batch_size=config_dict["batch_size"])
    dev_sampler = SequentialSampler(dev_event_pairs)
    dev_dataloader = DataLoader(dev_event_pairs,
                                sampler=dev_sampler,
                                batch_size=config_dict["batch_size"])

    for epoch_idx in trange(int(config_dict["epochs"]),
                            desc="Epoch",
                            leave=True):
        model = model.train()
        tr_loss = 0.0
        tr_p = 0.0
        tr_a = 0.0
        batcher = tqdm(train_dataloader, desc="Batch")
        for step, batch in enumerate(batcher):
            batch = tuple(t.to(device) for t in batch)
            sentences, segment_idxs, start_pieces_1, end_pieces_1, start_pieces_2, end_pieces_2, labels = batch
            out_dict = model(sentences, start_pieces_1, end_pieces_1,
                             start_pieces_2, end_pieces_2, segment_idxs,
                             labels)
            loss = out_dict["loss"]
            precision = out_dict["precision"]
            accuracy = out_dict["accuracy"]
            loss.backward()
            tr_loss += loss.item()
            tr_p += precision.item()
            tr_a += accuracy.item()

            if ((step + 1) * config_dict["batch_size"]
                ) % config_dict["accumulated_batch_size"] == 0:
                batcher.set_description(
                    "Batch (average loss: {:.6f} precision: {:.6f} accuracy: {:.6f})"
                    .format(
                        tr_loss / float(step + 1),
                        tr_p / float(step + 1),
                        tr_a / float(step + 1),
                    ))
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config_dict["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        evaluate(model, event_encoder, dev_dataloader, dev_pairs, dev_docs,
                 epoch_idx)


def main():
    logging.info('Loading training and dev data...')

    logging.info('Training and dev data have been loaded.')
    if not args.evaluate_dev:
        with open(config_dict["train_path"], 'rb') as f:
            training_data = cPickle.load(f)
        with open(config_dict["dev_path"], 'rb') as f:
            dev_data = cPickle.load(f)
        train_model(training_data, dev_data)
    else:
        with open(config_dict["test_path"], 'rb') as f:
            dev_data = cPickle.load(f)
        topic_sizes = [
            len([
                mention for key, doc in topic.docs.items()
                for sent_id, sent in doc.get_sentences().items()
                for mention in sent.gold_event_mentions
            ]) for topic in dev_data.topics.values()
        ]
        print(topic_sizes)
        print(sum(topic_sizes))
        print(sum([size * size for size in topic_sizes]))
        device = torch.device("cuda:0" if args.use_cuda else "cpu")
        with open(config_dict["event_encoder_model"], 'rb') as f:
            params = torch.load(f)
            event_encoder = EncoderCosineRanker(device)
            event_encoder.load_state_dict(params)
            event_encoder = event_encoder.to(device).eval()
            event_encoder.requires_grad = False
        with open(os.path.join(args.out_dir, 'best_model'), 'rb') as f:
            params = torch.load(f)
            model = CoreferenceCrossEncoder(device)
            model.load_state_dict(params)
            model = model.to(device).eval()
            model.requires_grad = False
        dev_event_pairs, dev_pairs, dev_docs = structure_dataset(
            dev_data, event_encoder, events=config_dict["events"], k=5)
        dev_sampler = SequentialSampler(dev_event_pairs)
        dev_dataloader = DataLoader(dev_event_pairs,
                                    sampler=dev_sampler,
                                    batch_size=config_dict["batch_size"])
        evaluate(model, event_encoder, dev_dataloader, dev_pairs, dev_docs, 0)


if __name__ == '__main__':
    main()
