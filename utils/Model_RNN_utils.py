#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:34:23 2019

Seq2Seq WITHOUT Attention

@author: zjy
"""
import tensorflow as tf


class Seq2Seq(object):
    def __init__(self, params, mode):
        """
        Initialize layers to build.
        """
        self.mode = mode
        self.params = params
        self.scope_name = "S2SA"
        # embedding
        self.embedding = Embedding(vocab_size=params["vocab_size"],
                                   embed_size=params["embed_size"],
                                   pretrained=params["pretrained_embedding"])
        # seq2seq framework
        self.encoder = Encoder(params, mode)
        self.decoder = Decoder(params, mode)
        self.output_layer = Output_layer(params, self.embedding.linear)  # tied weights

    def __call__(self, inputs, targets=None):
        """
        :param inputs:  [batch, length1] (input ids)
        :param targets: [batch, length2] (target ids)
        :return: logits: [batch, length, vocab_size]
        """
        # Variance scaling might be useful.
        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")

        with tf.variable_scope(self.scope_name, initializer=initializer):
            source_embed, source_length = self.embedding(inputs), get_seq_length(inputs)
            source_encoding, source_state = self.encoder(source_embed, source_length)
            # Generate output sequence if targets is None, or return logits if given target
            if targets is None:
                target_embed, target_length = None, None
            else:
                target_embed, target_length = self.embedding(targets), get_seq_length(targets)

            outputs, outputs_len = self.decoder(source_encoding, source_length, source_state,
                                self.embedding.embedding,
                                self.output_layer,
                                target_embed=target_embed,
                                target_length=target_length)
            if targets is None:
                return {"predict_ids": outputs, "predict_len": outputs_len}
            else:
                return {"logits": outputs}

    def encode(self, inputs):
        """
        :param inputs:  [batch, length1] (input ids)
        :return: source_encoding, source_state
        """
        source_embed, source_length = self.embedding(inputs), get_seq_length(inputs)
        source_encoding, source_state = self.encoder(source_embed, source_length)
        return source_encoding, source_state

    def decode(self, source_encoding, source_length, source_state):
        return self.decoder(source_encoding, source_length, source_state,
                     self.embedding.embedding,
                     self.output_layer)


class Embedding(tf.layers.Layer):
    def __init__(self, vocab_size, embed_size, pretrained=None):
        super(Embedding, self).__init__(name="Embedding")
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pretrained = pretrained
        if self.pretrained is not None:
            assert self.pretrained.shape[0] == self.vocab_size
            assert self.pretrained.shape[1] == self.embed_size

    def build(self, _):
        self.embedding = tf.get_variable("embedding",
                                         shape=[self.vocab_size, self.embed_size],
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
        if self.pretrained is not None:
            pretrained_embedding = tf.convert_to_tensor(self.pretrained, dtype=tf.float32)
            self.embedding = tf.where(condition=tf.cast(tf.reduce_sum(pretrained_embedding, axis=-1), dtype=tf.bool),
                                      x=pretrained_embedding,  # if true: there is pretrained_embedding
                                      y=self.embedding)  # if false: no pretrained_embedding (zero embedding)

        self.linear = TiedWeightDense(self.vocab_size, kernel=tf.transpose(self.embedding))
        self.built = True

    def call(self, input_ids, **kwargs):
        return tf.nn.embedding_lookup(self.embedding, input_ids)

    def linear(self, x):
        return self.linear(x)


class Encoder(tf.layers.Layer):
    def __init__(self, params, mode):
        super(Encoder, self).__init__(name="Encoder")
        # bidirectional
        self.input_size = params["embed_size"]
        self.hidden_size = params["hidden_size"] // 2
        self.num_hidden_layers = params["num_hidden_layers"]
        self.mode = mode
        self.dropout = params["dropout"]

    def build(self, _):
        self.forward_cell = build_MultiRNNCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            mode=self.mode,
            dropout=self.dropout)
        self.backward_cell = build_MultiRNNCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            mode=self.mode,
            dropout=self.dropout)
        self.built = True

    def call(self, input_embed, input_length=None):
        """
        :param input_embed: [batch, length, emb_size]
        :param input_length: [batch]
        :return:
            encoder_outputs: [batch, length, hidden_size]
            encoder_state: tuple(list of fw_state [Batch, hidden], list of bw_state [Batch, hidden])
        """
        bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            self.forward_cell,
            self.backward_cell,
            inputs=input_embed,
            sequence_length=input_length,
            dtype=tf.float32, time_major=False)

        # concat encoder_outputs to shape [batch, length, hidden_size]
        encoder_outputs = tf.concat(bi_outputs, axis=-1)
        # concat forward and backward encoder_state if necessary
        if self.num_hidden_layers == 1:
            encoder_state = bi_encoder_state
        else:
            encoder_state = concat_state_GRU(bi_encoder_state)
        return encoder_outputs, encoder_state


class Decoder(tf.layers.Layer):
    def __init__(self, params, mode):
        super(Decoder, self).__init__(name="Decoder")
        self.input_size = params["embed_size"]
        self.hidden_size = params["hidden_size"]
        self.dropout = params["dropout"]
        self.max_decode_length = params["max_decode_length"]
        self.num_hidden_layers = params["num_hidden_layers"]
        self.infer_mode = params["infer_mode"]
        self.beam_width = params["beam_size"]
        self.length_penalty_weight = params["length_penalty_weight"]
        self.coverage_penalty_weight = params["coverage_penalty_weight"]
        self.get_all_beams = params["get_all_beams"]
        self.mode = mode

    def build(self, _):
        self.decoder_cell = build_MultiRNNCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            mode=self.mode, variational_dropout=False)
        self.built = True

    def call(self, source_encoding, source_length=None, source_state=None,
             embedding_matrix=None, output_layer=None, target_embed=None, target_length=None):
        # train or eval
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            if target_embed is not None and target_length is not None:
                # Shift targets to the right, and remove the last element (hence the same length as before)
                with tf.name_scope("shift_targets"):
                    decoder_inputs_emb = tf.pad(target_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
                    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_emb, target_length)
            else:
                raise Exception("Train or Eval mode must provide target_embed and target_length")

            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=helper,
                initial_state=source_state,
                output_layer=output_layer)
            # 10%~20% speed up if we put output_layer outside the recurrent loop (OK since each step is independent).
            # according to https://github.com/tensorflow/nmt/blob/master/nmt/model.py#L518
            # Dynamic decoding
            outputs, final_state, outputs_length = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory=True)
            return outputs.rnn_output, outputs_length
        # infer
        else:
            batch_size = tf.shape(source_encoding)[0]
            if self.infer_mode == "beam_search":
                decoder_initial_state = tf.contrib.seq2seq.tile_batch(source_state, multiplier=self.beam_width)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self.decoder_cell,
                    embedding=embedding_matrix,
                    start_tokens=tf.fill([batch_size], 0),
                    end_token=1,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=output_layer,
                    length_penalty_weight=self.length_penalty_weight,
                    coverage_penalty_weight=self.coverage_penalty_weight)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embedding_matrix,
                    start_tokens=tf.fill([batch_size], 0),
                    end_token=1)  # EOS_id is hardcoded as 1
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=helper,
                    initial_state=source_state,
                    output_layer=output_layer)

            outputs, final_state, outputs_length = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.max_decode_length)
            if self.infer_mode == "beam_search":
                if self.get_all_beams:  # all to beam-major
                    # outputs.predicted_ids is [batch, length, beam]
                    predicted_ids = tf.transpose(outputs.predicted_ids, [2, 0, 1])
                    # outputs_length is [batch, beam]
                    outputs_length = tf.transpose(outputs_length, [1, 0])
                else:
                    predicted_ids = outputs.predicted_ids[:, :, 0]
                    outputs_length = outputs_length[:, 0]
                return predicted_ids, outputs_length
            else:
                return outputs.sample_id, outputs_length

class Output_layer(tf.layers.Layer):
    def __init__(self, params, func):
        super(Output_layer, self).__init__(name="Output")
        self.func = func
        self.hidden_size = params["vocab_size"]
        self.projection_size = params["embed_size"]
        self.vocab_size = params["vocab_size"]
        self.embed_size = params["embed_size"]

    def build(self, _):
        self.projection = tf.layers.Dense(self.projection_size, name="Projection/Hidden_to_Embed")
        self.build = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.hidden_size)

    def call(self, x, **kwargs):
        # TODO: make broadcastable
        return self.func(self.projection(x))


class TiedWeightDense(tf.keras.layers.Dense):
    def __init__(self, units, kernel):
        super(TiedWeightDense, self).__init__(
            units,
            activation=None,
            use_bias=False,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None)
        self.kernel_tensor = kernel

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = tf.layers.InputSpec(min_ndim=2, axes={-1: input_shape[-1].value})
        self.kernel = self.kernel_tensor
        self.bias = None
        self.built = True


def build_MultiRNNCell(input_size, hidden_size, num_hidden_layers, mode, dropout=0.2,
                       state_is_tuple=True, cell_type="gru", variational_dropout=True):
    # basic cell
    if cell_type.lower() == "gru":
        make_cell = tf.contrib.rnn.GRUBlockCell
    else:
        raise Exception("cell type %s unimplemented" % cell_type)
    # add dropout based on mode (training or not)
    if mode == tf.contrib.learn.ModeKeys.TRAIN and variational_dropout:
        wrap_first = lambda cell: tf.nn.rnn_cell.DropoutWrapper(
            cell, input_keep_prob=1 - dropout, output_keep_prob=1 - dropout, state_keep_prob=1 - dropout,
            variational_recurrent=True, dtype=tf.float32, input_size=input_size)
        wrap_others = lambda cell: tf.nn.rnn_cell.DropoutWrapper(
            cell, input_keep_prob=1 - dropout, output_keep_prob=1 - dropout, state_keep_prob=1 - dropout,
            variational_recurrent=True, dtype=tf.float32, input_size=hidden_size)
    else:
        wrap_first = wrap_others = lambda cell: cell
    return tf.contrib.rnn.MultiRNNCell(
        [wrap_first(make_cell(hidden_size)) if i == 0 else wrap_others(make_cell(hidden_size)) \
            for i in range(num_hidden_layers)], state_is_tuple=state_is_tuple)


def get_seq_length(seq_ids):
    padding = tf.to_float(tf.equal(seq_ids, 0))
    pad_len = tf.cast(tf.reduce_sum(padding, axis=1), dtype=tf.int32)
    seq_len = tf.shape(seq_ids)[1] - pad_len
    return seq_len


def concat_state_GRU(states):
    states_fw, states_bw = states[0], states[1]
    concated_states = []
    for fw, bw in zip(states_fw, states_bw):
        concated_states.append(tf.concat([fw, bw], -1))
    return tuple(concated_states)
