#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import gensim
import numpy as np
import time
import tensorflow as tf
from sklearn.externals import joblib
from data_helper import DataSet,load_data_from_csv,save_hparams, load_hparams,get_config_proto,early_stop
from HAN import HAN

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

def add_arguments(parser):
    """Build ArgumentParser."""
    parser.add_argument("--mode", type=str, default='train', help="running mode: train | eval | inference")
    parser.add_argument('--learning_rate', type=float, nargs='?',
                        default=0.1)
    parser.add_argument('--num_train_epoch', type=int, nargs='?',default=20)
    parser.add_argument('--train_csv_file', type=str,default=None)
    parser.add_argument('--train_data_file', type=str,default=None)
    parser.add_argument('--eval_data_file', type=str,default=None)
    parser.add_argument('--eval_csv_file', type=str,default=None)
    parser.add_argument('--vocab_file', type=str,default=None)
    parser.add_argument('--decay_schema', type=str,default=None)

    parser.add_argument('--max_sent_in_doc',  type=int, nargs='?',default=50)
    parser.add_argument('--decay_steps',  type=int, nargs='?',default=100)
    parser.add_argument('--max_word_in_sent',  type=int, nargs='?',default=30)
    parser.add_argument('--batch_size', type=int, nargs='?',default=32)
    parser.add_argument('--embedding_size', type=int, nargs='?',default=256)
    parser.add_argument('--hidden_size', type=int, nargs='?',default=256)
    parser.add_argument('--steps_per_summary', type=int, nargs='?',default=50)
    parser.add_argument('--steps_per_eval', type=int, nargs='?',default=100)
    parser.add_argument('--steps_per_stats', type=int, nargs='?',default=50)
    parser.add_argument('--checkpoint_dir', type=str, nargs='?',default='/data/share/tongjianing/project',)
    parser.add_argument('--num_classes', type=int, nargs='?',default=4)
    parser.add_argument('--num_gpus', type=int, nargs='?',default=1)
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")


def create_hparams(flags):
    return tf.contrib.training.HParams(
        mode = flags.mode,
        vocab_file = flags.vocab_file ,
        learning_rate = flags.learning_rate,
        train_data_file = flags.train_data_file,
        eval_data_file = flags.eval_data_file,
        train_csv_file = flags.train_csv_file,
        eval_csv_file = flags.eval_csv_file,
        num_train_epoch = flags.num_train_epoch,
        hidden_size = flags.hidden_size,
        embedding_size = flags.embedding_size ,
        batch_size = flags.batch_size,
        steps_per_summary = flags.steps_per_summary,
        steps_per_stats = flags.steps_per_stats,
        steps_per_eval  = flags.steps_per_eval ,
        checkpoint_dir = flags.checkpoint_dir,
        max_sent_in_doc = flags.max_sent_in_doc,
        max_word_in_sent = flags.max_word_in_sent,
        num_classes = flags.num_classes,
        decay_schema = flags.decay_schema,
        decay_steps = flags.decay_steps,
        num_gpus = flags.num_gpus,
        max_gradient_norm = flags.max_gradient_norm

)


def train(flags):
    train_data_df = load_data_from_csv(flags.train_csv_file)
    columns = train_data_df.columns.values.tolist()
    hparams = create_hparams(flags)
    hparams.add_hparam("vocab_size",10000)
    save_hparams(flags.checkpoint_dir, hparams)
    for column in columns[2:]:
        dataset = DataSet(flags.train_data_file, flags.train_csv_file, column, flags.batch_size, flags.vocab_file,flags.max_sent_in_doc,flags.max_word_in_sent)
        eval_dataset = DataSet(flags.eval_data_file, flags.eval_csv_file, column, flags.batch_size,  flags.vocab_file,flags.max_sent_in_doc,flags.max_word_in_sent)
        train_graph = tf.Graph()
        eval_graph = tf.Graph()

        with train_graph.as_default():
            train_model = HAN(hparams)
            initializer = tf.global_variables_initializer()

        with eval_graph.as_default():
            eval_hparams = load_hparams(flags.checkpoint_dir,{"mode":'eval','checkpoint_dir':flags.checkpoint_dir+"/best_dev"})
            eval_model = HAN(eval_hparams)

        train_sess = tf.Session(graph=train_graph, config=get_config_proto(log_device_placement=False))
        try:
            train_model.restore_model(train_sess)
        except:
            print("unable to restore model, initialize model with fresh params")
            train_model.init_model(train_sess, initializer = initializer)
        print("#{0} model starts to train with learning rate {1}, {2}".format(column, flags.learning_rate,time.ctime()))

        global_step = train_sess.run(train_model.global_step)
        eval_ppls = []
        best_eval = 1000000000
        for epoch in range(flags.num_train_epoch):
            checkpoint_loss = 0.0,

            for i,(x,y) in enumerate(dataset.get_next()):
                batch_loss, accuracy, summary, global_step = train_model.train_one_batch(train_sess, x, y) 
                checkpoint_loss += batch_loss * flags.batch_size 
                if global_step == 0:
                    continue

                if global_step % flags.steps_per_stats == 0:
                    summary = tf.Summary()
                    summary.value.add(tag='accuracy', simple_value = accuracy)
                    train_model.summary_writer.add_summary(summary, global_step=global_step)

                    print(
                        "# Epoch %d  global step %d batch %d/%d  "
                        "batch loss %.5f accuracy %.2f "%
                        (epoch+1, global_step,i+1, flags.batch_size, batch_loss, accuracy))
                    

                if global_step % flags.steps_per_eval == 0:
                    print("# global step {0}, eval model at {1}".format(global_step, time.ctime()))
                    checkpoint_path  = train_model.save_model(train_sess)
                    with tf.Session(graph=eval_graph, config=get_config_proto(log_device_placement=False)) as eval_sess:
                        eval_model.saver.restore(eval_sess, checkpoint_path)
                        eval_ppl = train_eval(eval_model, eval_sess, eval_dataset)
                        if eval_ppl < best_eval:
                            eval_model.save_model(eval_sess)
                            best_eval = eval_ppl
                    eval_ppls.append(eval_ppl)
                    if early_stop(eval_ppls):
                        print("# No loss decrease, early stop")
                        print("# Best perplexity {0}".format(best_eval))
                        exit(0)

            print("# Finsh epoch {1}, global step {0}".format(global_step, epoch+1))
   
def eval(flags):
    train_data_df = load_data_from_csv(flags.train_csv_file)
    columns = train_data_df.columns.values.tolist()
    hparams = load_hparams(flags.checkpoint_dir,{"mode":'eval','checkpoint_dir':flags.checkpoint_dir+"/best_dev","batch_size":flags.batch_size})

    save_hparams(flags.checkpoint_dir, hparams)
    for column in columns[2:]:
        dataset = DataSet(flags.data_file, flags.train_csv_file, column, flags.batch_size,  flags.vocab_file,flags.max_sent_in_doc,flags.max_word_in_sent)
        with tf.Session(config = get_config_proto(log_device_placement=False)) as sess:
            model = HAN(hparams)
            try:
                model.restore_model(sess)  #restore best solution
            except Exception as e:
                print("unable to restore model with exception",e)
                exit(1)

            checkpoint_loss= 0.0
            for i,(x,y) in enumerate(dataset.get_next()):
                batch_loss, accuracy = model.eval_one_batch(sess, x,y )
                checkpoint_loss += batch_loss * flags.batch_size
                print("# batch {0}/{1}, loss {2},accuracy{3}".format(i,flags.batch_size, checkpoint_loss,accuracy))
            return batch_loss,accuracy

def train_eval(model,sess,dataset):
    checkpoint_loss = 0.0
    for i,(x,y) in enumerate(dataset.get_next()):
        batch_loss, accuracy = model.eval_one_batch(sess, x,y )
        checkpoint_loss += batch_loss * flags.batch_size
        if (i+1) % 100 == 0:
            print("# batch {0}, loss {1},accuracy{2}".format(i, checkpoint_loss,accuracy))
    return checkpoint_loss,accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    if flags.mode == 'train':
        train(flags) 
    elif flags.mode == 'eval':
        eval(flags)
    # elif flags.mode == 'test':
    #     test(flags)