#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:45:29 2019

@author: zjy
"""

import os

from utils.Hparam_utils import create_hparams
from utils.Model_TransformeAutoEncoder_utils import TransformerAE
from utils.Common_utils import get_seq_length, threeClasses
from utils.Model_graph_utils import print_params
import tensorflow as tf

from train_clf import DataBatchLoaderClf, Dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
num_class = 3


if __name__ == '__main__':
    # ========== hparams ==========
    hparams = create_hparams()
    hparams.get_all_beams = False
    params = hparams.__dict__
    restore_seq2seq_path = hparams.model_dir
    restore_clf_path = hparams.model_dir + "_clf"
    data = DataBatchLoaderClf([hparams.pred_x, hparams.pred_label], hparams)

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
        transformerAE = TransformerAE(params, mode=tf.estimator.ModeKeys.PREDICT)
        latent = transformerAE.get_latent(input_ids)
        latent_to_feed = tf.identity(latent, name="latent")
        # greedy_decode
        greedy_ids = transformerAE.greedy_decode(tf.expand_dims(latent_to_feed, axis=1))
    # dense layer for classification
    with tf.variable_scope("classification"):
        classifer = Dense(params, mode=tf.estimator.ModeKeys.PREDICT, num_class=num_class)
        logits = classifer(latent_to_feed)
        prob = tf.nn.softmax(logits, axis=1)
        output_ids = tf.argmax(logits, axis=1)
    # loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_ids)
    # gradients
    gradients_for_latent = tf.gradients(loss, latent)

    # view model
    print_params(tf.trainable_variables())

    # ========== initialization & prediction ==========
    # restore seq2seq model
    variables_to_restore_seq2seq = [var for var in tf.trainable_variables() if "transformer" in var.name.lower()]
    restorer = tf.train.Saver(variables_to_restore_seq2seq)
    if not os.path.exists(restore_seq2seq_path):
        raise Exception("No seq2seq model found at {}".format(restore_seq2seq_path))
    restorer.restore(sess, os.path.join(restore_seq2seq_path, "model"))
    # restore classification model
    variables_to_restore_clf = [var for var in tf.trainable_variables() if "classification" in var.name.lower()]
    restorer_clf = tf.train.Saver(variables_to_restore_clf)
    if not os.path.exists(restore_clf_path):
        raise Exception("No classification model found at {}".format(restore_clf_path))
    restorer_clf.restore(sess, os.path.join(restore_clf_path, "model"))

    predictions = []
    # iterate over batches
    for b, batch in enumerate(data.generateBatch()):
        fetches = {"input_ids": input_ids, "input_len": input_len, "logits": logits,
                   "greedy_ids": greedy_ids, "output_class": output_ids, "output_prob": prob}
        res = sess.run(fetches, feed_dict={"input_ids:0": batch[0], "target_ids:0": batch[1]})
        inp = data.tokenizer.ids_2_words(res["input_ids"], res["input_len"], EOS_end=False, split=True)
        out = data.tokenizer.ids_2_words(res["greedy_ids"], res["input_len"] + 20, EOS_end=True, split=True)
        tgt_class = [threeClasses[c_id] for c_id in batch[1]]
        out_class = [threeClasses[c_id] for c_id in res["output_class"]]
        tf.logging.info("Original input text        : {}".format(inp))
        tf.logging.info("Original reconstructed text: {}".format(out))
        tf.logging.info("Original --> Target        : {}-->{}".format(out_class, tgt_class))
        tf.logging.info("Original logits: {}".format(res["logits"]))

        max_iters = 10
        for epsilon in [0.02, 0.1, 0.25, 0.5, 1.0, 2.0]:
            tf.logging.info("epsilon:{} =========================".format(epsilon))
            it = 0
            # get latent
            previous_latent_val, input_len_val = sess.run(
                [latent, input_len], feed_dict={"input_ids:0": batch[0], "target_ids:0": batch[1]})
            # fast gradient iterative method
            while it < max_iters:
                gradients_val = sess.run(
                    gradients_for_latent, feed_dict={"target_ids:0": batch[1],
                                                     "Transformer/latent:0": previous_latent_val})
                # update latent
                latent_val = previous_latent_val - epsilon * gradients_val[0]
                # print(latent_val[0][0], epsilon * gradients_val[0][0][0])
                # greedy decode
                greedy_ids_val, loss_val, logits_val, prob_val = sess.run(
                    [greedy_ids, loss, logits, prob],
                    feed_dict={"target_ids:0": batch[1], "Transformer/latent:0": latent_val})
                out = data.tokenizer.ids_2_words(greedy_ids_val, input_len_val, EOS_end=True, split=True)
                # decay epsilon
                epsilon = epsilon * 0.9
                # logging
                tf.logging.info("\titer:{}/{}, loss:{:.6f}, logits:{}, output:{}".format(
                    it+1, max_iters, loss_val[0], logits_val, out))
                it += 1
                previous_latent_val = latent_val