#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

import jieba
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from collections import Counter
import codecs
import tensorflow as tf
import os
import json
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df

def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1, 0, -1, -2], average='macro')

def save_hparams(out_dir, hparams):
    """Save hparams."""
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    hparams_file = os.path.join(out_dir, "hparams")
    print("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())

def load_hparams(out_dir, overidded = None):
    hparams_file = os.path.join(out_dir,"hparams")
    print("loading hparams from %s" % hparams_file)
    hparams_json = json.load(open(hparams_file))
    hparams = tf.contrib.training.HParams()
    for k,v in hparams_json.items():
        hparams.add_hparam(k,v)
    if overidded:
        for k,v in overidded.items():
            if k not in hparams_json:
                hparams.add_hparam(k,v)
            else:
                hparams.set_hparam(k,v)
    return hparams

def get_config_proto(log_device_placement=True, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0, per_process_gpu_memory_fraction=0.95, allow_growth=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = allow_growth
    config_proto.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto

def early_stop(values, no_decrease=3):
    if len(values) < 2:
        return False
    best_index = np.argmin(values)
    if values[-1] > values[best_index] and (best_index + no_decrease) <= len(values):
        return True
    else:
        return False

def read_vocab(vocab_file):
    """read vocab from file, one word per line
    """
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())

    word2id = {}
    for word in vocab:
        word2id[word] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word

class DataSet(object):
    def __init__(self, data_file,csv_file,column, batch_size,  vocab_file,max_sent_in_doc,max_word_in_sent):
        self.data_file = data_file
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.max_word_in_sent = max_word_in_sent
        self.max_sent_in_doc = max_sent_in_doc

        self.w2i, self.i2w = read_vocab(self.vocab_file)
        self.batch_len = 10500/self.batch_size
        self.labels = list(load_data_from_csv(csv_file)[column])
        print(len(self.labels))
        # self.processdata()
        print('*******************************')

    def vocab_size(self):
        return len(self.w2i)

    def processdata(self):
        print("preprocess data")
        with open(self.data_file, 'r') as f:
            for line in f:
                self.doc = []
                sentences = line.split("[。？！~ ]")
                for i,sent in enumerate(sentences):
                    sent_to_index = []
                    if i < self.max_sent_in_doc:
                        word_to_index = []
                        for j,word in enumerate(sent.split()):
                            if j< self.max_word_in_sent:
                                word_to_index = [self.w2i.get(word,UNK_ID)]
                        sent_to_index.append([SOS_ID]+ word_to_index + [EOS_ID])
                self.doc.append(sent_to_index)


    # def get_next_1(self):
    #     pos = 0
    #     while( pos<self.batch_len-1):
    #         yield(self.doc[pos:pos+self.batch_size], self.labels[pos:pos+self.batch_size])
    #         pos +=self.batch_size














    def get_next(self):
        pos = 0
        with open(self.data_file, 'r') as f:
            while pos< self.batch_len-1:
                source_tokens = []
                for line in f.readlines()[pos:pos+self.batch_size]:  # 单条原始语料
                    one_sent_source_tokens = []
                    sentences = re.split(r'[。？！~]', line)
                    if len(sentences) > self.max_sent_in_doc:
                        sentences = sentences[:self.max_sent_in_doc]

                    for sentence in sentences:
                        sentence = sentence.split()  # str --> list
                        if len(sentence)<self.max_word_in_sent:
                            sentence.extend([UNK_ID] * (self.max_word_in_sent-len(sentence)))
                        else:
                            sentence = sentence[:self.max_word_in_sent]
                        tokens = [SOS_ID] + [self.w2i.get(t,UNK_ID) for t in sentence] + [EOS_ID]
                        one_sent_source_tokens.append(tokens)

                    if len(sentences) < self.max_sent_in_doc:
                        for _ in range(self.max_sent_in_doc-len(sentences)):
                            one_sent_source_tokens.append([UNK_ID]*self.max_word_in_sent)
                    source_tokens.append(one_sent_source_tokens)
                yield(source_tokens, self.labels[pos:pos+self.batch_size])
                pos += 1

