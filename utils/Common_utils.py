#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:24:32 2019

@author: zjy
"""
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)
sixClasses = ["anger", "disgust", "sadness", "neutral", "like", "happiness"]
threeClasses = ["negative", "neutral", "positive"]

class Tokenizer:
    def __init__(self, tool='none', vocab_path='data/vocab'):
        self.tool_name = tool
        if tool.lower() == 'jieba':
            import jieba
            self.tool = jieba
        elif tool.lower() == 'hanlp':
            from pyhanlp import HanLP
            self.tool = HanLP
        elif tool.lower() == 'none':
            self.tool = None
        else:
            raise Exception("Unknown tokenization tool")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [l.strip().split()[0] for l in f.readlines()]

        self.w2i = {word: i for i, word in enumerate(self.vocab)}
        self.i2w = {i: word for i, word in enumerate(self.vocab)}

    def special_tokenize(self, string):
        substring_list = string.split(" <SEP> ")
        reunited = " <SEP> ".join([self.tokenize_line(sub) for sub in substring_list])
        return reunited.split()

    def tokenize_line(self, string, return_string=True):
        if self.tool_name.lower() == 'jieba':
            tokenized = [w for w in self.tool.cut(string) if w != ' ']
        elif self.tool_name.lower() == 'hanlp':
            tokenized = [term.word for term in self.tool.segment(string) if term != ' ']
        elif self.tool_name.lower() == 'none':
            tokenized = string.split()
        else:
            raise Exception("Unknown tokenization tool")
        if len(tokenized) == 0:
            return string
        return " ".join(tokenized) if return_string else tokenized

    def ids_2_words(self, output_ids, output_lens, EOS_end=False, split=False):
        sentences = []
        assert output_ids.shape[0] == output_lens.shape[0]
        for out_id, out_len in zip(output_ids, output_lens):
            if EOS_end:
                word_list = []
                for word_id in out_id:
                    if word_id != 1:
                        word_list.append(self.i2w[word_id])
                    else:
                        break
            else:
                word_list = [self.i2w[word_id] for word_id in out_id[:out_len]]
            sentences.append(" ".join(word_list) if split else "".join(word_list))
        return sentences


def estimate_total_steps(num_samples, hparams):
    total_num_steps_per_epoch = num_samples // hparams.batch_size
    if hparams.num_gpus:
        total_num_steps_per_epoch = total_num_steps_per_epoch // hparams.num_gpus
    tf.logging.info("Estimated total_num_steps = %d for %d epochs" % (
        total_num_steps_per_epoch * hparams.total_num_epochs,
        hparams.total_num_epochs))
    return total_num_steps_per_epoch * hparams.total_num_epochs


def get_seq_length(seq_ids):
    padding = tf.to_float(tf.equal(seq_ids, 0))
    pad_len = tf.cast(tf.reduce_sum(padding, axis=1), dtype=tf.int32)
    seq_len = tf.shape(seq_ids)[1] - pad_len
    return seq_len