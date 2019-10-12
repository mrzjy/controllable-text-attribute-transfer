#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:45:29 2019

use LSTM-based seq2seq for simplicity

@author: zjy
"""
import os
import random
import json
from utils.Hparam_utils import create_hparams
from utils.Model_TransformeAutoEncoder_utils import TransformerAE
from utils.Common_utils import Tokenizer, get_seq_length, estimate_total_steps
from utils.Model_graph_utils import print_params, padded_cross_entropy_loss
from utils.Optimization import create_optimizer
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataBatchLoader:
    def __init__(self, filePath, hparams):
        self.filePath = filePath
        self.batch_size = hparams.batch_size
        self.max_seq_len = hparams.max_seq_len
        self.total_num_epochs = hparams.total_num_epochs
        self.tokenizer = Tokenizer(vocab_path=hparams.vocab_file)

    def read(self, prefetch_size):
        with open(self.filePath, "r") as f:
            prefetch = []
            for l in f:
                # skip invalid lines
                if l.strip() == "":
                    continue
                # append processed data
                prefetch.append(self.preprocess(l.strip()))
                if len(prefetch) == prefetch_size:
                    random.shuffle(prefetch)
                    yield prefetch
                    prefetch = []
            if len(prefetch):  # not forget final batch
                yield prefetch

    def preprocess(self, string):
        # tokenize
        tokens = self.tokenizer.tokenize_line(string, return_string=False)
        # word to ids
        word_ids = [self.tokenizer.w2i.get(token, 2) for token in tokens]
        word_ids_with_EOS = word_ids + [1]
        # padding to max_seq_len
        while len(word_ids) < self.max_seq_len:
            word_ids.append(0)
            word_ids_with_EOS.append(0)
        return word_ids[:self.max_seq_len], word_ids_with_EOS[:self.max_seq_len+1]

    def generateBatch(self):
        for prefetch in self.read(prefetch_size=self.batch_size * 1000):
            for i in range(0, len(prefetch), self.batch_size):
                inputs = [p[0] for p in prefetch[i: i + self.batch_size]]
                targets = [p[1] for p in prefetch[i: i + self.batch_size]]
                yield inputs, targets


def estimate_num_steps(filePath, hparams):
    num_samples = 0
    tf.logging.info("Estimating total_num_steps (reading %s)" % filePath)
    with open(filePath, 'r') as f:  # add random filter for large files
        for i, l in enumerate(f):
            num_samples += 1
    # add learning parameters (add max for local debug(else num_steps may be 0 and thus NaN occurs))
    total_num_steps = max(1, estimate_total_steps(num_samples, hparams))
    hparams.add_hparam("total_num_steps_per_epoch", total_num_steps//hparams.total_num_epochs)
    hparams.add_hparam("total_num_steps", total_num_steps)
    warmup_num_steps = max(1, int(total_num_steps * hparams.warmup_proportion))
    hparams.add_hparam("warmup_num_steps", warmup_num_steps)


if __name__ == '__main__':
    # ========== hparams ==========
    hparams = create_hparams()
    hparams.get_all_beams = False
    hparams.total_num_epochs = 10
    hparams.learning_rate = 0.0002
    hparams.dropout = 0.05
    params = hparams.__dict__
    save_path = hparams.model_dir
    estimate_num_steps(hparams.train_y, hparams)
    data = DataBatchLoader(hparams.train_y, hparams)

    # ========== build session and graph ==========
    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=gpu_config)
    # input
    input_ids = tf.placeholder(shape=[None, params["max_seq_len"]], name="input_ids", dtype=tf.int32)
    target_ids = tf.placeholder(shape=[None, params["max_seq_len"]+1], name="target_ids", dtype=tf.int32)
    input_len = get_seq_length(input_ids)
    # model
    transformerAE = TransformerAE(params, mode=tf.estimator.ModeKeys.TRAIN)
    # outputs
    outputs = transformerAE(input_ids, target_ids)
    output_ids = tf.argmax(outputs["logits"], axis=2)
    greedy_ids = outputs["greedy_decode"]
    # loss
    loss, _ = padded_cross_entropy_loss(
        outputs["logits"], target_ids, params["label_smoothing"], params["vocab_size"])
    num_samples = tf.cast(tf.reduce_sum(input_len), tf.float32)
    loss = tf.reduce_sum(loss) / num_samples
    # train_op
    train_op, actual_lr = create_optimizer(
        loss, params["learning_rate"], params["total_num_steps"],
        params["warmup_num_steps"], params=params)
    # view model
    print_params(tf.trainable_variables())
    
    # ========== initialization & training ==========
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        saver.restore(sess, os.path.join(save_path, "model"))
    fetches = {"_": train_op, "loss": loss, "lr": actual_lr,
               "input_ids": input_ids, "input_len": input_len, "output_ids": output_ids}
    # iterate over epoches
    for e in range(params["total_num_epochs"]):
        # iterate over batches
        for b, batch in enumerate(data.generateBatch()):
            res = sess.run(fetches, feed_dict={"input_ids:0": batch[0], "target_ids:0": batch[1]})
            # logging
            if b % params["log_frequency"] == 0:
                tf.logging.info("Epoch:{:0>2d}/{:0>2d}, Batch:{:0>5d}/{:0>5d}, loss:{:.7f}, lr:{:.7f}".format(
                    e+1, params["total_num_epochs"], b, params["total_num_steps_per_epoch"], res["loss"], res["lr"]
                ))
                rand_id = random.randint(0, len(batch) - 1)
                inp = data.tokenizer.ids_2_words(res["input_ids"], res["input_len"], EOS_end=False, split=True)
                out = data.tokenizer.ids_2_words(res["output_ids"], res["input_len"], EOS_end=False, split=True)
                greedy_ids_val = sess.run(greedy_ids, feed_dict={"input_ids:0": batch[0]})
                pred = data.tokenizer.ids_2_words(greedy_ids_val, res["input_len"]+1, EOS_end=True, split=True)
                tf.logging.info("\tInput : %s" % inp[rand_id])
                tf.logging.info("\tOutput: %s" % out[rand_id])
                tf.logging.info("\tGreedy: %s" % pred[rand_id])

        # ========== save model at each end of epoch ==========
        saver.save(sess, os.path.join(save_path, "model"))
        tf.logging.info("Model saved in path: %s" % os.path.join(save_path, "model"))