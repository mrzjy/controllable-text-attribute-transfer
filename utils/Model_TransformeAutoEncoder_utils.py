# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Defines the Transformer model, and its encoder and decoder stacks.
Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.python.util import nest

from utils import Model_graph_utils
from utils.Model_RNN_utils import Encoder as Encoder_GRU
from utils.Model_RNN_utils import concat_state_GRU
from utils.Model_Transformer_utils import Transformer, sequence_beam_search, _get_shape_keep_last_dim
from utils.Model_Transformer_utils import EmbeddingSharedWeights, EncoderStack, DecoderStack
# Default value
EOS_ID = 1
INF = 1. * 1e7
_NEG_INF = -1e9


# In[] 
# XXX: transformer main graph
class TransformerAE(Transformer):
    def __init__(self, params, mode):
        """
        Initialize layers to build Transformer model.
        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            mode: e.g, tf.estimator.ModeKeys.TRAIN
        """
        self.train = mode == tf.estimator.ModeKeys.TRAIN
        self.params = params

        self.embedding_softmax_layer = EmbeddingSharedWeights(
            params["vocab_size"], params["hidden_size"], method="gather")

        self.encoder_stack = EncoderStack(params, self.train)
        self.representation_layer = RepresentationLayer(params, mode)
        self.decoder_stack = DecoderStack(params, self.train)

    def __call__(self, inputs, targets=None):
        """Calculate target logits or inferred target sequences.
        Args:
            inputs: int tensor with shape [batch_size, input_length].
        Returns:
            If targets is defined, then return logits for each word in the target
            sequence. float tensor with shape [batch_size, target_length, vocab_size]
            If target is none, then generate output sequence one token at a time.
            returns tuple of (predict_ids, length, scores)
        """
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        with tf.variable_scope("Transformer", initializer=initializer):
            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            # latent variable : dim = [batch, hidden]
            latent = self.get_latent(inputs)
            latent = tf.expand_dims(latent, axis=1)  # [batch, 1, hidden]

            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            greedy_decode_ids = self.greedy_decode(latent)
            if not self.train:
                predicts, scores, length = self.predict(inputs, latent)
                return {"predict_ids": predicts, "predict_len": length, "latent": latent, "greedy_decode": greedy_decode_ids}
            else:
                logits = self.decode(targets, latent)
                return {"logits": logits, "latent": latent, "greedy_decode": greedy_decode_ids}

    def get_latent(self, inputs):
        # Calculate attention bias for encoder self-attention and decoder
        # multi-headed attention layers.
        attention_bias = Model_graph_utils.get_padding_bias(inputs)
        # Run the inputs through the encoder layer to map the symbol
        # representations to continuous representations.
        encoder_outputs = self.encode(inputs, attention_bias)
        # latent variable : dim = [batch, hidden]
        return self.representation_layer(encoder_outputs)

    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs.
        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            embedded_inputs = self.embedding_softmax_layer(inputs)
            inputs_padding = Model_graph_utils.get_padding(inputs)

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = Model_graph_utils.get_position_encoding(
                    length, self.params["hidden_size"])
                encoder_inputs = embedded_inputs + pos_encoding

            if self.train:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, 1 - self.params["dropout"])

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, targets, latent):
        """Generate logits for each value in the target sequence.
        Args:
          latent: self representation with shape [batch_size, 1, hidden_size]
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
          cache: cache
        Returns:
          float32 tensor with shape [batch_size, target_length, vocab_size]
        """
        with tf.name_scope("decode_ae"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_softmax_layer(targets)
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += Model_graph_utils.get_position_encoding(
                    length, self.params["hidden_size"])
            if self.train:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, 1 - self.params["dropout"])

            batch_size = tf.shape(latent)[0]
            attention_bias = tf.zeros(shape=[batch_size, 1, 1, 1])

            # Run values
            decoder_self_attention_bias = Model_graph_utils.get_decoder_self_attention_bias(length)
            outputs = self.decoder_stack(
                decoder_inputs, latent, decoder_self_attention_bias, attention_bias, None)
            logits = self.embedding_softmax_layer.linear(outputs)
            return logits

    def greedy_decode(self, latent):
        batch_size = tf.shape(latent)[0]
        max_decode_length = self.params["max_seq_len"]
        timing_signal = Model_graph_utils.get_position_encoding(
            max_decode_length + 1, self.params["hidden_size"])
        decoder_self_attention_bias = Model_graph_utils.get_decoder_self_attention_bias(
            max_decode_length)
        encoder_decoder_attention_bias = tf.zeros(shape=[batch_size, 1, 1, 1])

        def condition(ids, i, cache):
            return i < max_decode_length

        def symbols_to_logits_fn(ids, i, cache):
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, latent, self_attention_bias,
                encoder_decoder_attention_bias, cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            new_id = tf.argmax(logits, axis=1)
            new_id = tf.expand_dims(new_id, axis=1)
            new_ids = tf.concat([ids, new_id], axis=1)
            return new_ids, i+1, cache

        initial_ids = tf.zeros([batch_size, 1], dtype=tf.int64)
        initial_step = tf.constant(0, dtype=tf.int32)
        initial_cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
            } for layer in range(self.params["num_hidden_layers"])}

        ids, _, _ = tf.while_loop(
            cond=condition,
            body=symbols_to_logits_fn,
            loop_vars=[initial_ids, initial_step, initial_cache],
            shape_invariants=[tf.TensorShape([None, None]),
                              initial_step.get_shape(),
                              nest.map_structure(_get_shape_keep_last_dim, initial_cache)])
        return ids[:, 1:]   # initial_ids is useless

    def predict(self, inputs, latent):
        """Return predicted sequence."""
        batch_size = tf.shape(latent)[0]
        input_length = tf.shape(inputs)[1]
        max_decode_length = input_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)
        initial_index = 0
        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
            } for layer in range(self.params["num_hidden_layers"])}

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = latent
        cache["encoder_decoder_attention_bias"] = tf.zeros(shape=[batch_size, 1, 1, 1])

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params["vocab_size"],
            beam_size=self.params["beam_size"],
            length_penalty_weight=self.params["length_penalty_weight"],
            max_decode_length=max_decode_length,
            eos_id=EOS_ID,
            initial_index=initial_index,
            conditional_generation=False,
            sampling=self.params["sampling"])

        # results
        if self.params["get_all_beams"]:
            predicts = decoded_ids[:, :, 1:]
            predicts = tf.transpose(predicts, [1, 0, 2])
            length = tf.map_fn(lambda x: Model_graph_utils.get_seq_length(x),
                               predicts,  # beam-major, so perform map_fn beam-wisely
                               dtype=tf.int32)
        else:  # only return top beam
            predicts = decoded_ids[:, 0, 1:]
            scores = scores[:, 0]
            length = Model_graph_utils.get_seq_length(predicts)

        # reverse or not
        if self.params["reverse_sequence"]:  # remember to leave EOS
            if self.params["get_all_beams"]:
                def get_len(x):  # avoid len < 0
                    return tf.where(x[1] - 1 > 0, x[1] - 1, tf.zeros_like(x[1]))

                predicts = tf.map_fn(lambda x: tf.reverse_sequence(x[0], get_len(x), seq_axis=1),
                                     [predicts, length],  # beam-major, so perform map_fn beam-wisely
                                     dtype=tf.int32)
            else:
                predicts = tf.reverse_sequence(predicts, length - 1, seq_axis=1)

        if self.params["get_all_beams"]:
            # final response is batch-major
            predicts = tf.transpose(predicts, [1, 0, 2])
            length = tf.transpose(length, [1, 0])
        return predicts, scores, length


class RepresentationLayer(Encoder_GRU):
    def __init__(self, params, mode):
        super(Encoder_GRU, self).__init__(name="representation")
        # bidirectional
        self.input_size = params["hidden_size"]
        self.hidden_size = params["hidden_size"] // 2
        self.num_hidden_layers = 1
        self.mode = mode
        self.dropout = params["dropout"]

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
        return concat_state_GRU(bi_encoder_state)[-1]
