import tensorflow as tf
import argparse
import os
import pickle
import numpy as np


# hard coded, do not change this
PAD = '<PAD>'
PAD_id = 0
EOS = '<EOS>'
EOS_id = 1
UNK = '<UNK>'
UNK_id = 2


def create_standard_hparams():
    return tf.contrib.training.HParams(
        # envirionment
        num_gpus=1,
        port=8666,

        # Data constraints
        num_buckets=20,
        bucket_width=2,
        max_seq_len=25,
        min_seq_len=1,
        num_parallel_calls=8,

        # Data format
        PAD=PAD,
        PAD_id=PAD_id,
        EOS=EOS,
        EOS_id=EOS_id,
        UNK=UNK,
        UNK_id=UNK_id,

        # dir
        data_dir='data',
        test_data_dir='',
        model_dir='saved_model/S2S/Trans',

        # data
        train_y='train_y',
        train_y_multilabel='train_y_multilabel',
        pred_x='pred_x',  # only pred_x is absolute path, for others we'll join the path with data_dir
        pred_label='pred_label',

        # vocab
        vocab_dir='',
        vocab_file='vocab',

        # Transformer
        # ======================================================================
        hidden_size=200,
        initializer_gain=1.0,  # Used in trainable variable initialization.
        num_hidden_layers=2,  # Number of layers in the encoder and decoder stacks.
        num_heads=4,  # Number of heads to use in multi-headed attention.
        filter_size=800,  # Inner layer dimension in the feedforward network (about 4 times the hidden size).
        allow_ffn_pad=True,
        label_smoothing=0.1,
        extra_decode_length=5,

        # ======================================================================
        # Default prediction params
        max_decode_length=50,
        beam_size=4,
        length_penalty_weight=0.8,
        coverage_penalty_weight=0.0,

        # tricks
        char=False,  # char-level tokenization
        reverse_sequence=False,   # reverse sequence generation (will be reversed back to normal finally)

        # Train
        dropout=0,
        threshold=0.5,
        warmup_proportion=0.005,
        summary_frequency=80000,
        log_frequency=100,  # per steps / batches
        eval_frequency=100000,
        batch_size=256,
        total_num_epochs=20,
        learning_rate=4e-3,
        optimizer="adam",
        max_gradient_norm=5.0,
        l2_reg=0.01,

        # infer
        infer_mode="beam_search",
        get_all_beams=True,  # only used when running predict_autoencoder.py
        conditional="哈哈",
        sampling=True,  # sampling probable tokens instead of greedy decoding
        max_prediction=10000,
    )


def create_hparams():
    hparams = create_standard_hparams()
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('--hparams', type=str, default="",
                        help='Comma separated list of "name=value" pairs.')
    args = parser.parse_args()
    hparams.parse(args.hparams)

    hparams.set_hparam("train_y", os.path.join(hparams.data_dir, hparams.train_y))
    hparams.set_hparam("train_y_multilabel", os.path.join(hparams.data_dir, hparams.train_y_multilabel))
    if hparams.vocab_dir == '':
        hparams.set_hparam("vocab_dir", hparams.data_dir)
        tf.logging.info("Vocab_dir not defined, using same dir as data_dir: %s" % hparams.data_dir)
    hparams.set_hparam("vocab_file", os.path.join(hparams.vocab_dir, hparams.vocab_file))
    # set vocab
    if hparams.char:
        hparams.vocab_file += "_char"
        hparams.model_dir += "_char"
        tf.logging.info("Using char-level vocab setting, vocab_file name must end with \"_char\" ")

    with open(hparams.vocab_file, 'r') as f:
        vocab = [l.strip() for l in f.readlines()]
        hparams.add_hparam("vocab", vocab)
        hparams.add_hparam("vocab_size", len(vocab))
        tf.logging.info("Found %d vocab_size from %s" % (len(vocab), hparams.vocab_file))
    return hparams


def print_mission(hparams):
    tf.logging.info("Training schedule:")
    tf.logging.info("\t1. Train for {} epochs (Train-Eval-inferenceTest) in total.".format(hparams.total_num_epochs))
    tf.logging.info("\t2. Train for {} total num steps.".format(hparams.total_num_steps))
    tf.logging.info("\t3. Evaluate every %d steps." % hparams.eval_frequency)
    tf.logging.info("\t4. Batch_size=%d." % hparams.batch_size)


def load_pretrained_embedding(hparams):
    w2v = pickle.load(open(hparams.pretrained_embedding_file, "rb"))
    key_list = list(w2v.keys())
    embedding, dim = [], len(w2v[key_list[0]])
    assert dim == hparams.embed_size, "pretrained embedding dim not equal to hparam dim setting {} v.s. {}".format(
        dim, hparams.embed_size)
    for word in hparams.vocab:
        embedding.append(w2v.get(word, [0]*dim))
    return np.array(embedding)
