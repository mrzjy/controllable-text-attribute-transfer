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

from utils import Model_graph_utils
from tensorflow.python.util import nest


# Default value
EOS_ID = 1
INF = 1. * 1e7
_NEG_INF = -1e9


# In[] 
# XXX: transformer main graph
class Transformer(object):
    """Transformer model for sequence to sequence data."""
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
        self.decoder_stack = DecoderStack(params, self.train)

    def __call__(self, inputs, targets=None, conditional_targets=None):
        """Calculate target logits or inferred target sequences.
        Args:
            inputs: int tensor with shape [batch_size, input_length].
            targets: None or int tensor with shape [batch_size, target_length].

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
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = Model_graph_utils.get_padding_bias(inputs)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, attention_bias)

            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            if targets is None:
                # if conditional_targets is None, then initial decoder state is zero;
                # otherwise, initial decoder state is non-zero
                initial_ids, initial_cache, current_index = self.create_conditional_initial_state(
                    conditional_targets, encoder_outputs, attention_bias)
                predicts, scores, length = self.predict(
                    encoder_outputs, attention_bias,
                    initial_ids=initial_ids, initial_cache=initial_cache, initial_index=current_index)
                return {"predict_ids": predicts, "predict_len": length, "scores": scores}
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias)
                return {"logits": logits}

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

    def decode(self, targets, encoder_outputs, attention_bias, cache=None):
        """Generate logits for each value in the target sequence.
        Args:
          targets: target values for the output sequence.
            int tensor with shape [batch_size, target_length]
          encoder_outputs: continuous representation of input sequence.
            float tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
          cache: cache
        Returns:
          float32 tensor with shape [batch_size, target_length, vocab_size]
        """
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_softmax_layer(targets)
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(
                    decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += Model_graph_utils.get_position_encoding(
                    length, self.params["hidden_size"])
            if self.train:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, 1 - self.params["dropout"])

            # Run values
            decoder_self_attention_bias = Model_graph_utils.get_decoder_self_attention_bias(length)
            outputs = self.decoder_stack(
                decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                attention_bias, cache)
            logits = self.embedding_softmax_layer.linear(outputs)
            return logits

    def create_conditional_initial_state(self, conditional_targets, encoder_outputs, attention_bias):
        if conditional_targets is None:
            return None, None, None
        # Create cache storing decoder attention values for each layer.
        batch_size, current_index = tf.shape(encoder_outputs)[0], tf.shape(conditional_targets)[1]
        initial_cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
            } for layer in range(self.params["num_hidden_layers"])}
        # implicitly update cache
        _ = self.decode(conditional_targets, encoder_outputs, attention_bias, cache=initial_cache)
        initial_cache["encoder_outputs"] = encoder_outputs
        initial_cache["encoder_decoder_attention_bias"] = attention_bias
        cond_targets_length = Model_graph_utils.get_seq_length(conditional_targets) - 1
        # initial_ids : [batch_size, ]
        dummy = tf.range(0, batch_size, dtype=tf.int32)
        indices = tf.concat([tf.expand_dims(dummy, 1), tf.expand_dims(cond_targets_length, 1)], axis=-1)
        initial_ids = tf.gather_nd(params=conditional_targets, indices=indices)
        return initial_ids, initial_cache, current_index

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = Model_graph_utils.get_position_encoding(
            max_decode_length + 1, self.params["hidden_size"])
        decoder_self_attention_bias = Model_graph_utils.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
            Args:
              ids: Current decoded sequences.
                int tensor with shape [batch_size * beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_decoder_attention_bias,
                initial_ids=None, initial_cache=None, initial_index=None):
        """Return predicted sequence."""
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params["extra_decode_length"]

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        if initial_cache is None and initial_ids is None:
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
            cache["encoder_outputs"] = encoder_outputs
            cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
            conditional_generation = False
        else:
            assert initial_index is not None
            initial_ids = initial_ids
            cache = initial_cache
            conditional_generation = True

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
            conditional_generation=conditional_generation,
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


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params, train):
        self.layer = layer
        self.postprocess_dropout = params["dropout"]
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(params["hidden_size"])

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


class EncoderStack(tf.layers.Layer):
    """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

    def __init__(self, params, train):
        super(EncoderStack, self).__init__()
        self.layers = []
        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["dropout"], train)
            feed_forward_network = FeedFowardNetwork(
                params["hidden_size"], params["filter_size"],
                params["dropout"], train, params["allow_ffn_pad"])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.
        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer.
            [batch_size, 1, 1, input_length]
          inputs_padding: P

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
                with tf.variable_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)


class DecoderStack(tf.layers.Layer):
    """Transformer decoder stack.
    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """

    def __init__(self, params, train):
        super(DecoderStack, self).__init__()
        self.layers = []
        for _ in range(params["num_hidden_layers"]):
            self_attention_layer = SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["dropout"], train)
            enc_dec_attention_layer = Attention(
                params["hidden_size"], params["num_heads"],
                params["dropout"], train)
            feed_forward_network = FeedFowardNetwork(
                params["hidden_size"], params["filter_size"],
                params["dropout"], train, params["allow_ffn_pad"])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(enc_dec_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)])

        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, cache=None):
        """Return the output of the decoder layer stacks.
        Args:
          decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
          encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
          decoder_self_attention_bias: bias for decoder self-attention layer.
            [1, 1, target_len, target_length]
          attention_bias: bias for encoder-decoder attention layer.
            [batch_size, 1, 1, input_length]
          cache: (Used for fast decoding) A nested dictionary storing previous
            decoder self-attention values. The items are:
              {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]},
               ...}

        Returns:
          Output of decoder layer stack.
          float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
                with tf.variable_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs, encoder_outputs, attention_bias)
                with tf.variable_scope("ffn"):
                    decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)


# In[] 
# XXX: embedding layer
class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size, method="gather", scope="vocab"):
        """Specify characteristic parameters of embedding layer.
        Args:
          vocab_size: Number of tokens in the embedding. (Typically ~32,000)
          hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
          method: Strategy for performing embedding lookup. "gather" uses tf.gather
            which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
            one-hot encodes the indicies and formulates the embedding as a sparse
            matrix multiplication. The matmul formulation is wasteful as it does
            extra work, however matrix multiplication is very fast on TPUs which
            makes "matmul" considerably faster than "gather" on TPUs.
        """
        super(EmbeddingSharedWeights, self).__init__(name=scope)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        if method not in ("gather", "matmul"):
            raise ValueError("method {} must be 'gather' or 'matmul'".format(method))
        self.method = method

    def build(self, _):
        with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
            # Create and initialize weights. The random normal initializer was chosen
            # randomly, and works well.
            self.shared_weights = tf.get_variable(
                "weights", [self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    0., self.hidden_size ** -0.5))
        self.built = True

    def call(self, x):
        """Get token embeddings of x.
        Args:
          x: An int64 tensor with shape [batch_size, length]
        Returns:
          embeddings: float32 tensor with shape [batch_size, length, embedding_size]
          padding: float32 tensor with shape [batch_size, length] indicating the
            locations of the padding tokens in x.
        """
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            mask = tf.to_float(tf.not_equal(x, 0))

            embeddings = tf.gather(self.shared_weights, x)
            embeddings *= tf.expand_dims(mask, -1)
            # embedding_matmul already zeros out masked positions, so
            # `embeddings *= tf.expand_dims(mask, -1)` is unnecessary.

            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5
            return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.
        Args:
          x: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])


# In[] 
# XXX: attention layer
class Attention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, dropout, train):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.train = train

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
          x: A tensor with shape [batch_size, length, hidden_size]

        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y.
        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.

        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.  # --> [batch_size, num_heads, length, hidden_size/num_heads]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)   # --> [batch_size, num_heads, length_q, length_k]
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.train:
            weights = tf.nn.dropout(weights, 1.0 - self.dropout)
        attention_output = tf.matmul(weights, v)   # # --> [batch_size, num_heads, length_q, hidden_size/num_heads]

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)  # --> [batch_size, length_q, hidden_size]
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)


# In[] 
# XXX: transformer ffn layer
class FeedFowardNetwork(tf.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, dropout, train, allow_pad):
        super(FeedFowardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.dropout = dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(
            filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name="output_layer")

    def call(self, x, padding=None):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, length, hidden_size]
          padding: (optional) If set, the padding values are temporarily removed
            from x (provided self.allow_pad is set). The padding values are placed
            back in the output tensor in the same locations.
            shape [batch_size, length]

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """
        padding = None if not self.allow_pad else padding

        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope("remove_padding"):
                # Flatten padding to [batch_size*length]
                pad_mask = tf.reshape(padding, [-1])
                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions.
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)
        if self.train:
            output = tf.nn.dropout(output, 1.0 - self.dropout)
        output = self.output_dense_layer(output)

        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size]
                )
                output = tf.reshape(output, [batch_size, length, self.hidden_size])
        return output


# In[] 
# XXX: transformer beam search
class _StateKeys(object):
    """Keys to dictionary storing the state of the beam search loop."""
    # Variable storing the loop index.
    CUR_INDEX = "CUR_INDEX"
    # Top sequences that are alive for each batch item. Alive sequences are ones
    # that have not generated an EOS token. Sequences that reach EOS are marked as
    # finished and moved to the FINISHED_SEQ tensor.
    # Has shape [batch_size, beam_size, CUR_INDEX + 1]
    ALIVE_SEQ = "ALIVE_SEQ"
    # Log probabilities of each alive sequence. Shape [batch_size, beam_size]
    ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
    # Dictionary of cached values for each alive sequence. The cache stores
    # the encoder output, attention bias, and the decoder attention output from
    # the previous iteration.
    ALIVE_CACHE = "ALIVE_CACHE"

    # Top finished sequences for each batch item.
    # Has shape [batch_size, beam_size, CUR_INDEX + 1]. Sequences that are
    # shorter than CUR_INDEX + 1 are padded with 0s.
    FINISHED_SEQ = "FINISHED_SEQ"
    # Scores for each finished sequence. Score = log probability / length norm
    # Shape [batch_size, beam_size]
    FINISHED_SCORES = "FINISHED_SCORES"
    # Flags indicating which sequences in the finished sequences are finished.
    # At the beginning, all of the sequences in FINISHED_SEQ are filler values.
    # True -> finished sequence, False -> filler. Shape [batch_size, beam_size]
    FINISHED_FLAGS = "FINISHED_FLAGS"


class SequenceBeamSearch(object):
    """Implementation of beam search loop."""

    def __init__(self, symbols_to_logits_fn, vocab_size, batch_size,
                 beam_size, length_penalty_weight, max_decode_length, eos_id,
                 conditional_generation, sampling=False):
        self.symbols_to_logits_fn = symbols_to_logits_fn
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.length_penalty_weight = length_penalty_weight
        self.max_decode_length = max_decode_length
        self.eos_id = eos_id
        self.conditional_generation = conditional_generation
        self.sampling = sampling

    def search(self, initial_ids, initial_cache, initial_index):
        """Beam search for sequences with highest scores."""
        state, state_shapes = self._create_initial_state(initial_ids, initial_cache, initial_index)

        finished_state = tf.while_loop(
            self._continue_search, self._search_step, loop_vars=[state],
            shape_invariants=[state_shapes], parallel_iterations=1, back_prop=False)
        finished_state = finished_state[0]

        alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
        finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
        finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
        finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]

        # Account for corner case where there are no finished sequences for a
        # particular batch item. In that case, return alive sequences for that batch
        # item.
        finished_seq = tf.where(
            tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
        finished_scores = tf.where(
            tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
        return finished_seq, finished_scores

    def _create_initial_state(self, initial_ids, initial_cache, initial_index):
        """Return initial state dictionary and its shape invariants.
        Args:
          initial_ids: initial ids to pass into the symbols_to_logits_fn.
            int tensor with shape [batch_size, 1]
          initial_cache: dictionary storing values to be passed into the
            symbols_to_logits_fn.

        Returns:
            state and shape invariant dictionaries with keys from _StateKeys
        """
        cur_index = initial_index
        # Create alive sequence with shape [batch_size, beam_size, 1]
        alive_seq = _expand_to_beam_size(initial_ids, self.beam_size)
        alive_seq = tf.expand_dims(alive_seq, axis=2)
        # Create tensor for storing initial log probabilities.
        # Assume initial_ids are prob 1.0
        initial_log_probs = tf.constant(
            [[0.] + [-float("inf")] * (self.beam_size - 1)])
        alive_log_probs = tf.tile(initial_log_probs, [self.batch_size, 1])

        # Expand all values stored in the dictionary to the beam size, so that each
        # beam has a separate cache.
        alive_cache = nest.map_structure(
            lambda t: _expand_to_beam_size(t, self.beam_size), initial_cache)

        # Initialize tensor storing finished sequences with filler values.
        finished_seq = tf.zeros(tf.shape(alive_seq), tf.int32)

        # Set scores of the initial finished seqs to negative infinity.
        finished_scores = tf.ones([self.batch_size, self.beam_size]) * -INF

        # Initialize finished flags with all False values.
        finished_flags = tf.zeros([self.batch_size, self.beam_size], tf.bool)

        # Create state dictionary
        state = {
            _StateKeys.CUR_INDEX: cur_index,
            _StateKeys.ALIVE_SEQ: alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
            _StateKeys.ALIVE_CACHE: alive_cache,
            _StateKeys.FINISHED_SEQ: finished_seq,
            _StateKeys.FINISHED_SCORES: finished_scores,
            _StateKeys.FINISHED_FLAGS: finished_flags
        }

        # Create state invariants for each value in the state dictionary. Each
        # dimension must be a constant or None. A None dimension means either:
        #   1) the dimension's value is a tensor that remains the same but may
        #      depend on the input sequence to the model (e.g. batch size).
        #   2) the dimension may have different values on different iterations.
        state_shape_invariants = {
            _StateKeys.CUR_INDEX: tf.TensorShape([]),
            _StateKeys.ALIVE_SEQ: tf.TensorShape([None, self.beam_size, None]),
            _StateKeys.ALIVE_LOG_PROBS: tf.TensorShape([None, self.beam_size]),
            _StateKeys.ALIVE_CACHE: nest.map_structure(
                _get_shape_keep_last_dim, alive_cache),
            _StateKeys.FINISHED_SEQ: tf.TensorShape([None, self.beam_size, None]),
            _StateKeys.FINISHED_SCORES: tf.TensorShape([None, self.beam_size]),
            _StateKeys.FINISHED_FLAGS: tf.TensorShape([None, self.beam_size])
        }
        return state, state_shape_invariants

    def _continue_search(self, state):
        """Return whether to continue the search loop.
        The loops should terminate when
          1) when decode length has been reached, or
          2) when the worst score in the finished sequences is better than the best
             score in the alive sequences (i.e. the finished sequences are provably
             unchanging)

        Args:
          state: A dictionary with the current loop state.

        Returns:
          Bool tensor with value True if loop should continue, False if loop should
          terminate.
        """
        i = state[_StateKeys.CUR_INDEX]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        not_at_max_decode_length = tf.less(i, self.max_decode_length)

        # Calculate largest length penalty (the larger penalty, the better score).
        max_length_norm = _length_normalization(self.length_penalty_weight, self.max_decode_length)
        # Get the best possible scores from alive sequences.
        best_alive_scores = alive_log_probs[:, 0] / max_length_norm

        # Compute worst score in finished sequences for each batch element
        finished_scores *= tf.to_float(finished_flags)  # set filler scores to zero
        lowest_finished_scores = tf.reduce_min(finished_scores, axis=1)

        # If there are no finished sequences in a batch element, then set the lowest
        # finished score to -INF for that element.
        finished_batches = tf.reduce_any(finished_flags, 1)
        lowest_finished_scores += (1. - tf.to_float(finished_batches)) * -INF

        worst_finished_score_better_than_best_alive_score = tf.reduce_all(
            tf.greater(lowest_finished_scores, best_alive_scores)
        )

        return tf.logical_and(
            not_at_max_decode_length,
            tf.logical_not(worst_finished_score_better_than_best_alive_score)
        )

    def _search_step(self, state):
        """Beam search loop body.
        Grow alive sequences by a single ID. Sequences that have reached the EOS
        token are marked as finished. The alive and finished sequences with the
        highest log probabilities and scores are returned.

        A sequence's finished score is calculating by dividing the log probability
        by the length normalization factor. Without length normalization, the
        search is more likely to return shorter sequences.

        Args:
          state: A dictionary with the current loop state.

        Returns:
          new state dictionary.
        """
        # Grow alive sequences by one token.
        new_seq, new_log_probs, new_cache = self._grow_alive_seq(state)
        # Collect top beam_size alive sequences
        alive_state = self._get_new_alive_state(new_seq, new_log_probs, new_cache)

        # Combine newly finished sequences with existing finished sequences, and
        # collect the top k scoring sequences.
        finished_state = self._get_new_finished_state(state, new_seq, new_log_probs)

        # Increment loop index and create new state dictionary
        new_state = {_StateKeys.CUR_INDEX: state[_StateKeys.CUR_INDEX] + 1}
        new_state.update(alive_state)
        new_state.update(finished_state)
        return [new_state]

    def _grow_alive_seq(self, state):
        """Grow alive sequences by one token, and collect top 2*beam_size sequences.
        2*beam_size sequences are collected because some sequences may have reached
        the EOS token. 2*beam_size ensures that at least beam_size sequences are
        still alive.

        Args:
          state: A dictionary with the current loop state.
        Returns:
          Tuple of
          (Top 2*beam_size sequences [batch_size, 2 * beam_size, cur_index + 1],
           Scores of returned sequences [batch_size, 2 * beam_size],
           New alive cache, for each of the 2 * beam_size sequences)
        """
        i = state[_StateKeys.CUR_INDEX]
        alive_seq = state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        alive_cache = state[_StateKeys.ALIVE_CACHE]

        beams_to_keep = 2 * self.beam_size

        # Get logits for the next candidate IDs for the alive sequences. Get the new
        # cache values at the same time.
        flat_ids = _flatten_beam_dim(alive_seq)  # [batch_size * beam_size]
        flat_cache = nest.map_structure(_flatten_beam_dim, alive_cache)

        flat_logits, flat_cache = self.symbols_to_logits_fn(flat_ids, i, flat_cache)

        # Unflatten logits to shape [batch_size, beam_size, vocab_size]
        logits = _unflatten_beam_dim(flat_logits, self.batch_size, self.beam_size)
        new_cache = nest.map_structure(
            lambda t: _unflatten_beam_dim(t, self.batch_size, self.beam_size),
            flat_cache)

        # Convert logits to normalized log probs
        # mask_hot = [0] * self.vocab_size
        # mask_hot[2] = -INF
        # mask = tf.constant([[mask_hot]], tf.float32)
        # logits += mask

        candidate_log_probs = _log_prob_from_logits(logits)

        # Calculate new log probabilities if each of the alive sequences were
        # extended # by the the candidate IDs.
        # Shape [batch_size, beam_size, vocab_size]
        log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

        # Each batch item has beam_size * vocab_size candidate sequences. For each
        # batch item, get the k candidates with the highest log probabilities.
        flat_log_probs = tf.reshape(log_probs,
                                    [-1, self.beam_size * self.vocab_size])
        topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=beams_to_keep)

        # Extract the alive sequences that generate the highest log probabilities
        # after being extended.
        topk_beam_indices = topk_indices // self.vocab_size
        topk_seq, new_cache = _gather_beams(
            [alive_seq, new_cache], topk_beam_indices, self.batch_size,
            beams_to_keep)

        # Append the most probable IDs to the topk sequences
        topk_ids = topk_indices % self.vocab_size
        topk_ids = tf.expand_dims(topk_ids, axis=2)
        topk_seq = tf.concat([topk_seq, topk_ids], axis=2)
        return topk_seq, topk_log_probs, new_cache

    def _grow_alive_seq_probabilistic(self, state):
        """Grow alive sequences by one token, and collect top 2*beam_size sequences.
        2*beam_size sequences are collected because some sequences may have reached
        the EOS token. 2*beam_size ensures that at least beam_size sequences are
        still alive.

        Args:
          state: A dictionary with the current loop state.
        Returns:
          Tuple of
          (Top 2*beam_size sequences [batch_size, 2 * beam_size, cur_index + 1],
           Scores of returned sequences [batch_size, 2 * beam_size],
           New alive cache, for each of the 2 * beam_size sequences)
        """
        i = state[_StateKeys.CUR_INDEX]
        alive_seq = state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        alive_cache = state[_StateKeys.ALIVE_CACHE]

        beams_to_keep = 2 * self.beam_size

        # Get logits for the next candidate IDs for the alive sequences. Get the new
        # cache values at the same time.
        flat_ids = _flatten_beam_dim(alive_seq)  # [batch_size * beam_size]
        flat_cache = nest.map_structure(_flatten_beam_dim, alive_cache)

        flat_logits, flat_cache = self.symbols_to_logits_fn(flat_ids, i, flat_cache)

        # Unflatten logits to shape [batch_size, beam_size, vocab_size]
        logits = _unflatten_beam_dim(flat_logits, self.batch_size, self.beam_size)
        new_cache = nest.map_structure(
            lambda t: _unflatten_beam_dim(t, self.batch_size, self.beam_size),
            flat_cache)

        # Convert logits to normalized log probs
        mask_hot = [0] * self.vocab_size
        mask_hot[2] = -INF
        mask = tf.constant([[mask_hot]], tf.float32)
        logits += mask
        candidate_log_probs = _log_prob_from_logits(logits)

        # Calculate new log probabilities if each of the alive sequences were
        # extended # by the the candidate IDs.
        # Shape [batch_size, beam_size, vocab_size]
        log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

        # Each batch item has beam_size * vocab_size candidate sequences. For each
        # batch item, get the k candidates with the highest log probabilities.
        flat_log_probs = tf.reshape(log_probs, [-1, self.beam_size * self.vocab_size])

        # top_k with probablity
        if self.sampling:
            topk_indices = tf.cast(tf.multinomial(flat_log_probs, beams_to_keep), tf.int32)
            topk_log_probs = tf.map_fn(lambda x: tf.gather(x[0], x[1]),
                                       [flat_log_probs, topk_indices],
                                       dtype=tf.float32)
        else:
            topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=beams_to_keep)

        # Extract the alive sequences that generate the highest log probabilities
        # after being extended.
        topk_beam_indices = topk_indices // self.vocab_size
        topk_seq, new_cache = _gather_beams(
            [alive_seq, new_cache], topk_beam_indices, self.batch_size,
            beams_to_keep)

        # Append the most probable IDs to the topk sequences
        topk_ids = topk_indices % self.vocab_size
        topk_ids = tf.expand_dims(topk_ids, axis=2)
        topk_seq = tf.concat([topk_seq, topk_ids], axis=2)
        return topk_seq, topk_log_probs, new_cache

    def _get_new_alive_state(self, new_seq, new_log_probs, new_cache):
        """Gather the top k sequences that are still alive.
        Args:
          new_seq: New sequences generated by growing the current alive sequences
            int32 tensor with shape [batch_size, 2 * beam_size, cur_index + 1]
          new_log_probs: Log probabilities of new sequences
            float32 tensor with shape [batch_size, beam_size]
          new_cache: Dict of cached values for each sequence.

        Returns:
          Dictionary with alive keys from _StateKeys:
            {Top beam_size sequences that are still alive (don't end with eos_id)
             Log probabilities of top alive sequences
             Dict cache storing decoder states for top alive sequences}
        """
        # To prevent finished sequences from being considered, set log probs to -INF
        new_finished_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
        new_log_probs += tf.to_float(new_finished_flags) * -INF

        top_alive_seq, top_alive_log_probs, top_alive_cache = _gather_topk_beams(
            [new_seq, new_log_probs, new_cache], new_log_probs, self.batch_size,
            self.beam_size)

        return {
            _StateKeys.ALIVE_SEQ: top_alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
            _StateKeys.ALIVE_CACHE: top_alive_cache
        }

    def _get_new_finished_state(self, state, new_seq, new_log_probs):
        """Combine new and old finished sequences, and gather the top k sequences.
        Args:
          state: A dictionary with the current loop state.
          new_seq: New sequences generated by growing the current alive sequences
            int32 tensor with shape [batch_size, beam_size, i + 1]
          new_log_probs: Log probabilities of new sequences
            float32 tensor with shape [batch_size, beam_size]

        Returns:
          Dictionary with finished keys from _StateKeys:
            {Top beam_size finished sequences based on score,
             Scores of finished sequences,
             Finished flags of finished sequences}
        """
        i = state[_StateKeys.CUR_INDEX]
        finished_seq = state[_StateKeys.FINISHED_SEQ]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        # First append a column of 0-ids to finished_seq to increment the length.
        # New shape of finished_seq: [batch_size, beam_size, i + 1]
        finished_seq = tf.concat(
            [finished_seq,
             tf.zeros([self.batch_size, self.beam_size, 1], tf.int32)], axis=2)

        # Calculate new seq scores from log probabilities.
        length_norm = _length_normalization(self.length_penalty_weight, i + 1)
        new_scores = new_log_probs / length_norm

        # Set the scores of the still-alive seq in new_seq to large negative values.
        new_finished_flags = tf.equal(new_seq[:, :, -1], self.eos_id)
        new_scores += (1. - tf.to_float(new_finished_flags)) * -INF

        # Combine sequences, scores, and flags.
        finished_seq = tf.concat([finished_seq, new_seq], axis=1)
        finished_scores = tf.concat([finished_scores, new_scores], axis=1)
        finished_flags = tf.concat([finished_flags, new_finished_flags], axis=1)

        # Return the finished sequences with the best scores.
        top_finished_seq, top_finished_scores, top_finished_flags = (
            _gather_topk_beams([finished_seq, finished_scores, finished_flags],
                               finished_scores, self.batch_size, self.beam_size))

        return {
            _StateKeys.FINISHED_SEQ: top_finished_seq,
            _StateKeys.FINISHED_SCORES: top_finished_scores,
            _StateKeys.FINISHED_FLAGS: top_finished_flags
        }


def sequence_beam_search(
        symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size,
        length_penalty_weight, max_decode_length, eos_id, initial_index, conditional_generation, sampling=False):
    """Search for sequence of subtoken ids with the largest probability.
    Args:
        symbols_to_logits_fn: A function that takes in ids, index, and cache as
          arguments. The passed in arguments will have shape:
            ids -> [batch_size * beam_size, index]
            index -> [] (scalar)
            cache -> nested dictionary of tensors [batch_size * beam_size, ...]
          The function must return logits and new cache.
            logits -> [batch * beam_size, vocab_size]
            new cache -> same shape/structure as inputted cache
        initial_ids: Starting ids for each batch item.
          int32 tensor with shape [batch_size]
        initial_cache: dict containing starting decoder variables information
        vocab_size: int size of tokens
        beam_size: int number of beams
        length_penalty_weight: float defining the strength of length normalization
        max_decode_length: maximum length to decoded sequence
        eos_id: int id of eos token, used to determine when a sequence has finished
        conditional_generation: boolean indicating whether to perform conditional generation

    Returns:
        Top decoded sequences [batch_size, beam_size, max_decode_length]
        sequence scores [batch_size, beam_size]
    """
    batch_size = tf.shape(initial_ids)[0]
    sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size,
                             beam_size, length_penalty_weight, max_decode_length, eos_id,
                             conditional_generation, sampling)
    return sbs.search(initial_ids, initial_cache, initial_index)


def _log_prob_from_logits(logits):
    return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)


def _length_normalization(length_penalty_weight, length):
    """Return length normalization factor."""
    return tf.pow(((5. + tf.to_float(length)) / 6.), length_penalty_weight)


def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.
    Args:
        tensor: tensor to tile [batch_size, ...]
        beam_size: How much to tile the tensor by.

    Returns:
        Tiled tensor [batch_size, beam_size, ...]
    """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def _shape_list(tensor):
    """Return a list of the tensor's shape, and ensure no None values in list."""
    # Get statically known shape (may contain None's for unknown dimensions)
    shape = tensor.get_shape().as_list()

    # Ensure that the shape values are not None
    dynamic_shape = tf.shape(tensor)
    for i in range(len(shape)):  # pylint: disable=consider-using-enumerate
        if shape[i] is None:
            shape[i] = dynamic_shape[i]
    return shape


def _get_shape_keep_last_dim(tensor):
    shape_list = _shape_list(tensor)

    # Only the last
    for i in range(len(shape_list) - 1):
        shape_list[i] = None

    if isinstance(shape_list[-1], tf.Tensor):
        shape_list[-1] = None
    return tf.TensorShape(shape_list)


def _flatten_beam_dim(tensor):
    """Reshapes first two dimensions in to single dimension.
    Args:
        tensor: Tensor to reshape of shape [A, B, ...]

    Returns:
        Reshaped tensor of shape [A*B, ...]
    """
    shape = _shape_list(tensor)
    shape[0] *= shape[1]
    shape.pop(1)  # Remove beam dim
    return tf.reshape(tensor, shape)


def _unflatten_beam_dim(tensor, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].
    Args:
        tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
        batch_size: Tensor, original batch size.
        beam_size: int, original beam size.

    Returns:
        Reshaped tensor of shape [batch_size, beam_size, ...]
    """
    shape = _shape_list(tensor)
    new_shape = [batch_size, beam_size] + shape[1:]
    return tf.reshape(tensor, new_shape)


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
    """Gather beams from nested structure of tensors.
    Each tensor in nested represents a batch of beams, where beam refers to a
    single search state (beam search involves searching through multiple states
    in parallel).

    This function is used to gather the top beams, specified by
    beam_indices, from the nested tensors.

    Args:
        nested: Nested structure (tensor, list, tuple or dict) containing tensors
          with shape [batch_size, beam_size, ...].
        beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each
         value in beam_indices must be between [0, beam_size), and are not
         necessarily unique.
        batch_size: int size of batch
        new_beam_size: int number of beams to be pulled from the nested tensors.

    Returns:
        Nested structure containing tensors with shape
          [batch_size, new_beam_size, ...]
    """
    # Computes the i'th coodinate that contains the batch index for gather_nd.
    # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..].
    batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])

    # Create coordinates to be passed to tf.gather_nd. Stacking creates a tensor
    # with shape [batch_size, beam_size, 2], where the last dimension contains
    # the (i, j) gathering coordinates.
    coordinates = tf.stack([batch_pos, beam_indices], axis=2)

    return nest.map_structure(
        lambda state: tf.gather_nd(state, coordinates), nested)


def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
    """Gather top beams from nested structure."""
    _, topk_indexes = tf.nn.top_k(score_or_log_prob, k=beam_size)
    return _gather_beams(nested, topk_indexes, batch_size, beam_size)