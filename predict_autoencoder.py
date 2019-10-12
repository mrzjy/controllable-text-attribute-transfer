#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:45:29 2019

@author: zjy
"""

import os
from utils.Hparam_utils import create_hparams
from utils.Model_TransformeAutoEncoder_utils import TransformerAE
from utils.Common_utils import get_seq_length
from utils.Model_graph_utils import print_params
import tensorflow as tf

from train_autoencoder import DataBatchLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # ========== hparams ==========
    hparams = create_hparams()
    hparams.get_all_beams = False
    params = hparams.__dict__
    save_path = hparams.model_dir
    data = DataBatchLoader(hparams.pred_x, hparams)

    # ========== build session and graph ==========
    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=gpu_config)
    # input
    input_ids = tf.placeholder(shape=[None, params["max_seq_len"]], name="input_ids", dtype=tf.int32)
    input_len = get_seq_length(input_ids)
    # model
    transformerAE = TransformerAE(params, mode=tf.estimator.ModeKeys.PREDICT)
    # outputs
    outputs = transformerAE(input_ids)
    latent = outputs["latent"]
    output_ids = outputs["predict_ids"]
    output_len = outputs["predict_len"]
    greedy_ids = outputs["greedy_decode"]
    # view model
    print_params(tf.trainable_variables())

    # ========== initialization & prediction ==========
    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        raise Exception("No model found at {}".format(save_path))
    saver.restore(sess, os.path.join(save_path, "model"))
    fetches = {"input_ids": input_ids, "input_len": input_len,
               "greedy_ids": greedy_ids, "output_len": output_len,
               "latent": latent}
    predictions = []
    # iterate over batches
    for b, batch in enumerate(data.generateBatch()):
        res = sess.run(fetches, feed_dict={"input_ids:0": batch[0]})
        inp = data.tokenizer.ids_2_words(res["input_ids"], res["input_len"], EOS_end=False, split=True)
        out = data.tokenizer.ids_2_words(res["greedy_ids"], res["input_len"] + 5, EOS_end=True, split=True)
        batch_predictions = ["{}\t{}".format(i, o) for i, o in zip(inp, out)]
        predictions.extend(batch_predictions)
        print(batch_predictions)
        # logging
        if b % 200 == 0:
            tf.logging.info(b)
    # with open("prediction.txt", "w+") as f:
    #     print("\n".join(predictions), file=f)