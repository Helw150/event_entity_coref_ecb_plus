import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange
from scorer import *
from nearest_neighbor import nn_eval
import _pickle as cPickle

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
from eval_utils import *
from coarse import *
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

# Fix the random seeds
seed = config_dict["random_seed"]
random.seed(seed)
np.random.seed(seed)
if args.use_cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Training with CUDA')

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
best_loss = None


def structure_dataset(data_set, events=True):
    processed_dataset = []
    labels_to_ids = {}
    label_sets = {}
    label_vocab_size = 0
    docs = [
        document for topic in data_set.topics.values()
        for document in topic.docs.values()
    ]
    for doc in docs:
        for sentence in doc.get_sentences().values():
            tokenized_sentence, tokenization_mapping = tokenize_and_map(
                sentence, tokenizer)
            sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
            for mention in sentence_mentions:
                if mention.gold_tag not in labels_to_ids:
                    labels_to_ids[mention.gold_tag] = label_vocab_size
                    label_sets[label_vocab_size] = []
                    label_vocab_size += 1
                label_id = labels_to_ids[mention.gold_tag]
                start_piece = tokenization_mapping[mention.start_offset][0]
                end_piece = tokenization_mapping[mention.end_offset][-1]
                record = {
                    "sentence": tokenized_sentence,
                    "label": [label_id],
                    "start_piece": [start_piece],
                    "end_piece": [end_piece]
                }
                processed_dataset.append(record)
                label_sets[label_id].append(record)

    sentences = torch.tensor(
        [record["sentence"] for record in processed_dataset])
    labels = torch.tensor([record["label"] for record in processed_dataset])
    start_pieces = torch.tensor(
        [record["start_piece"] for record in processed_dataset])
    end_pieces = torch.tensor(
        [record["end_piece"] for record in processed_dataset])
    return TensorDataset(sentences, start_pieces, end_pieces,
                         labels), label_sets


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


def evaluate(model, dev_data, dev_raw, dev_event_gold, epoch_num):
    global best_loss
    recall, mrr, maP, p_at_k = nn_eval(dev_raw, model)
    loss_based = False
    tqdm.write("Recall: {:.6f} - MRR: {:.6f} - MAP: {:.6f}".format(
        recall, mrr, maP))
    if not loss_based:
        if best_loss == None or recall > best_loss:
            tqdm.write("Recall Improved Saving Model")
            best_loss = recall
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, 'best_model'))
        return
    with torch.no_grad():
        model.update_cluster_lookup(dev_event_gold, dev=True)
        tr_loss = 0
        tr_accuracy = 0
        examples = 0
        for step, batch in enumerate(tqdm(dev_data, desc="Evaluation")):
            batch = tuple(t.to(model.device) for t in batch)
            sentences, start_pieces, end_pieces, labels = batch
            out_dict = model(sentences,
                             start_pieces,
                             end_pieces,
                             labels,
                             dev=True)
            loss = out_dict["loss"]
            accuracy = out_dict["accuracy"]
            examples += len(batch)
            tr_loss += loss.item()
            tr_accuracy += accuracy.item()
        if best_loss == None or tr_loss < best_loss:
            tqdm.write("Loss Improved Saving Model")
            best_loss = tr_loss
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, 'best_model'))
        logger.debug(
            "Epoch {} - Dev Loss: {:.6f} - Dev Precision: {:.6f}".format(
                epoch_num, tr_loss / float(examples),
                tr_accuracy / len(dev_data)))
        tqdm.write(
            "Epoch {} - Dev Loss: {:.6f} - Dev Precision: {:.6f}".format(
                epoch_num, tr_loss / float(examples),
                tr_accuracy / len(dev_data)))


def train_model(train_set, dev_set):
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    model = EncoderCosineRanker(device)
    model.to(device)
    train_event_mentions, train_event_gold = structure_dataset(train_set,
                                                               events=True)
    #train_entity_mentions, train_entity_gold = structure_dataset(train_set, events = False)
    dev_event_mentions, dev_event_gold = structure_dataset(dev_set,
                                                           events=True)
    #dev_entity_mentions, dev_entity_gold = structure_dataset(dev_set, events = False)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, len(train_event_mentions))
    train_sampler = RandomSampler(train_event_mentions)
    train_dataloader = DataLoader(train_event_mentions,
                                  sampler=train_sampler,
                                  batch_size=config_dict["batch_size"])
    dev_sampler = RandomSampler(dev_event_mentions)
    dev_dataloader = DataLoader(dev_event_mentions,
                                sampler=dev_sampler,
                                batch_size=config_dict["batch_size"])

    model.train()
    for epoch_idx in trange(int(config_dict["epochs"]), desc="Epoch"):
        tr_loss = 0.0
        model.update_cluster_lookup(train_event_gold)
        for step, batch in enumerate(tqdm(train_dataloader, desc="Batch")):
            batch = tuple(t.to(device) for t in batch)
            sentences, start_pieces, end_pieces, labels = batch
            out_dict = model(sentences, start_pieces, end_pieces, labels)
            loss = out_dict["loss"]
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) * config_dict["batch_size"] % config_dict[
                    "accumulated_batch_size"] == 0:
                tqdm.write("Step {} - epoch {} average loss: {:.6f}".format(
                    step,
                    epoch_idx,
                    tr_loss / float(config_dict["accumulated_batch_size"]),
                ))
                tr_loss = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config_dict["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        evaluate(model, dev_dataloader, dev_set, dev_event_gold, epoch_idx)


def main():
    logging.info('Loading training and dev data...')
    with open(config_dict["train_path"], 'rb') as f:
        training_data = cPickle.load(f)
    with open(config_dict["dev_path"], 'rb') as f:
        dev_data = cPickle.load(f)

    logging.info('Training and dev data have been loaded.')

    train_model(training_data, dev_data)


if __name__ == '__main__':
    main()
