#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:45:29 2019

use LSTM-based seq2seq for simplicity

@author: zjy
"""
import os
import random
from utils.Hparam_utils import create_hparams
from utils.Model_TransformeAutoEncoder_utils import TransformerAE
from utils.Common_utils import Tokenizer, get_seq_length, estimate_total_steps, sixClasses
from utils.Model_graph_utils import print_params, padded_cross_entropy_loss
from utils.Optimization import create_simple_optimizer
import tensorflow as tf
from sklearn.metrics import f1_score


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
num_class = 3  # len(sixClasses)


class DataBatchLoaderClf:
    def __init__(self, filePathList, hparams):
        """ filePathList: [textfile, classfile] """
        assert len(filePathList) == 2
        self.filePathList = filePathList
        self.batch_size = hparams.batch_size
        self.max_seq_len = hparams.max_seq_len
        self.total_num_epochs = hparams.total_num_epochs
        self.tokenizer = Tokenizer(vocab_path=hparams.vocab_file)
        self.sixClasses = sixClasses

    def read(self, prefetch_size):
        with open(self.filePathList[0], "r") as f:
            with open(self.filePathList[1], "r") as g:
                prefetch = []
                for l, cls in zip(f, g):
                    # skip invalid lines
                    if l.strip() == "" or cls.strip() == "":
                        continue
                    # append processed data
                    prefetch.append(self.preprocess(l.strip(), cls.strip()))
                    if len(prefetch) == prefetch_size:
                        random.shuffle(prefetch)
                        yield prefetch
                        prefetch = []
                if len(prefetch):  # not forget final batch
                    yield prefetch

    def preprocess(self, string, cls):
        # tokenize
        tokens = self.tokenizer.tokenize_line(string, return_string=False)
        # word to ids
        word_ids = [self.tokenizer.w2i.get(token, 2) for token in tokens]
        # padding to max_seq_len
        while len(word_ids) < self.max_seq_len:
            word_ids.append(0)
        # class to id
        class_id = self.sixClasses.index(cls)
        # TODO:  sixClass -> 3Class
        if class_id < 3:
            class_id = 0
        elif class_id == 3:
            class_id = 1
        else:
            class_id = 2
        return word_ids[:self.max_seq_len], class_id

    def generateBatch(self):
        for prefetch in self.read(prefetch_size=self.batch_size * 1000):
            for i in range(0, len(prefetch), self.batch_size):
                batch_word_ids = [p[0] for p in prefetch[i: i + self.batch_size]]
                batch_class_ids = [p[1] for p in prefetch[i: i + self.batch_size]]
                yield batch_word_ids, batch_class_ids


class Dense(tf.layers.Layer):
    def __init__(self, params, mode, num_class):
        super(Dense, self).__init__()
        self.hidden_size = params["hidden_size"]
        self.num_class = num_class
        self.mode = mode

    def build(self, _):
        self.fc1 = tf.layers.Dense(self.hidden_size//2)
        self.fc2 = tf.layers.Dense(self.hidden_size//4)
        self.fc3 = tf.layers.Dense(self.num_class)
        self.built = True

    def call(self, input):
        out = self.fc1(input)
        out = tf.nn.leaky_relu(out)
        out = self.fc2(out)
        out = tf.nn.leaky_relu(out)
        out = self.fc3(out)
        out = tf.nn.sigmoid(out)
        return out


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
    hparams.batch_size = 1024
    hparams.total_num_epochs = 10
    hparams.dropout = 0
    params = hparams.__dict__
    params["dense_dropout"] = 0.15
    restore_path = hparams.model_dir
    save_path = hparams.model_dir + "_clf"
    estimate_num_steps(hparams.train_y, hparams)
    data = DataBatchLoaderClf([hparams.train_y, hparams.train_y_multilabel], hparams)

    # ========== build session and graph ==========
    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=gpu_config)
    # input
    input_ids = tf.placeholder(shape=[None, params["max_seq_len"]], name="input_ids", dtype=tf.int32)
    input_len = get_seq_length(input_ids)
    target_ids = tf.placeholder(shape=[None], name="target_ids", dtype=tf.int32)
    # model
    with tf.variable_scope("Transformer"):
        transformerAE = TransformerAE(params, mode=tf.estimator.ModeKeys.TRAIN)
        latent = transformerAE.get_latent(input_ids)
    # dense layer for classification
    with tf.variable_scope("classification"):  # 2-layer FFN
        classifer = Dense(params, mode=tf.estimator.ModeKeys.TRAIN, num_class=num_class)
        logits = classifer(latent)
        output_ids = tf.argmax(logits, axis=1)
    # loss
    loss, _ = padded_cross_entropy_loss(
        logits, target_ids, 0.0, num_class, label_dim=2)
    num_samples = tf.cast(tf.reduce_sum(input_len), tf.float32)
    loss = tf.reduce_sum(loss) / num_samples
    # train_op
    train_op, actual_lr = create_simple_optimizer(
        loss, params["learning_rate"], params["total_num_steps"], params=params,
        tvars=[var for var in tf.trainable_variables() if "classification" in var.name])
    # view model
    print_params([var for var in tf.trainable_variables() if "classification" in var.name.lower()])
    
    # ========== initialization & training ==========
    sess.run(tf.global_variables_initializer())
    # restore seq2seq model
    variables_to_restore = [var for var in tf.trainable_variables() if "transformer" in var.name.lower()]
    restorer = tf.train.Saver(variables_to_restore)
    if not os.path.exists(restore_path):
        raise Exception("No seq2seq model found at {}".format(restore_path))
    restorer.restore(sess, os.path.join(restore_path, "model"))
    # saver for classification model
    variables_to_save = [var for var in tf.trainable_variables() if "classification" in var.name.lower()]
    saver = tf.train.Saver(variables_to_save, max_to_keep=1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fetches = {"_": train_op, "loss": loss, "lr": actual_lr,
               "input_ids": input_ids, "input_len": input_len,
               "target_ids": target_ids, "output_ids": output_ids}
    # iterate over epoches
    for e in range(params["total_num_epochs"]):
        # iterate over batches
        for b, batch in enumerate(data.generateBatch()):
            res = sess.run(fetches, feed_dict={"input_ids:0": batch[0], "target_ids:0": batch[1]})
            f1 = f1_score(res["target_ids"], res["output_ids"], average='weighted')
            # logging
            if b % params["log_frequency"] == 0:
                tf.logging.info("Epoch:{:0>2d}/{:0>2d}, Batch:{:0>5d}/{:0>5d}, loss:{:.7f}, lr:{:.7f}, batch_F1:{:.3f}".format(
                    e+1, params["total_num_epochs"], b, params["total_num_steps_per_epoch"], res["loss"], res["lr"], f1
                ))
        # ========== save model at each end of epoch ==========
        saver.save(sess, os.path.join(save_path, "model"))
        tf.logging.info("Model saved in path: %s" % os.path.join(save_path, "model"))