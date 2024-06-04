# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import keras

def get_text_encoder(max_length=77, vocab_size=49408,embedding_size=128,batch_size=None):                
    tokens = keras.layers.Input(
        shape=(max_length,), dtype="int32", name="tokens",batch_size=batch_size
    )
    positions = keras.layers.Input(
        shape=(max_length,), dtype="int32", name="positions",batch_size=batch_size
    )        
    inter_outputs = []
    x = CLIPEmbedding(vocab_size, embedding_size, max_length)([tokens, positions])        
    x = ResBlock(768)(x)
    inter_outputs.append(x)
    x = ResBlock(256)(x)

    for i in range(3):        
        x = CLIPEncoderLayer(256, 8, activation=quick_gelu,name=f"clip_encoder_{i}")(x)
        x = ResBlock(768)(x)
        inter_outputs.append(x)
        x = ResBlock(256)(x)        

    x = ResBlock(768)(x)
    embedded = keras.layers.LayerNormalization(epsilon=1e-5)(x)    
    model = keras.Model([tokens, positions], [embedded] + inter_outputs)
    return model

def quick_gelu(x):
    return x * tf.nn.sigmoid(x * 1.702)


class ResBlock(keras.Model):
    def __init__(self,out_channels,**kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.dense_1 = keras.layers.Dense(out_channels)
        self.norm = keras.layers.LayerNormalization()
        self.act = keras.layers.Activation("swish")

    def build(self,input_shape):
        if self.out_channels == input_shape[-1]:
            self.skip_dense = tf.identity()
        else: 
            self.skip_dense = keras.layers.Dense(self.out_channels)
    
    def call(self,x):
        residual = x 
        x = self.norm(x)
        x = self.act(x)
        x = self.dense_1(x)
        x = x + self.skip_dense(residual)
        return x 


class CLIPEmbedding(keras.layers.Layer):
    def __init__(
        self, input_dim=49408, output_dim=768, max_length=77, **kwargs
    ):
        super().__init__(**kwargs)
        self.token_embedding = keras.layers.Embedding(input_dim, output_dim)
        self.position_embedding = keras.layers.Embedding(max_length, output_dim)

    def call(self, inputs):
        tokens, positions = inputs
        tokens = self.token_embedding(tokens)
        positions = self.position_embedding(positions)
        return tokens + positions
        # return tokens


class CLIPEncoderLayer(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.clip_attn = CLIPAttention(embed_dim, num_heads, causal=True)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.fc1 = keras.layers.Dense(embed_dim * 4)
        self.fc2 = keras.layers.Dense(embed_dim)
        self.activation = activation

    def call(self, inputs):
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.clip_attn(x)
        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual


class CLIPAttention(keras.layers.Layer):
    def __init__(self, embed_dim=768, num_heads=12, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = keras.layers.Dense(self.embed_dim)
        self.k_proj = keras.layers.Dense(self.embed_dim)
        self.v_proj = keras.layers.Dense(self.embed_dim)
        self.out_proj = keras.layers.Dense(self.embed_dim)

    def reshape_states(self, x, sequence_length, batch_size):
        x = tf.reshape(
            x, (batch_size, sequence_length, self.num_heads, self.head_dim)
        )
        return tf.transpose(
            x, (0, 2, 1, 3)
        )  # bs, heads, sequence_length, head_dim

    def call(self, inputs, attention_mask=None):
        if attention_mask is None and self.causal:
            length = tf.shape(inputs)[1]
            attention_mask = tf.experimental.numpy.triu(
                tf.ones((1, 1, length, length), dtype=self.compute_dtype)
                * -float("inf"),
                k=1,
            )

        _, tgt_len, embed_dim = inputs.shape
        query_states = self.q_proj(inputs) * self.scale
        key_states = self.reshape_states(self.k_proj(inputs), tgt_len, -1)
        value_states = self.reshape_states(self.v_proj(inputs), tgt_len, -1)

        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self.reshape_states(query_states, tgt_len, -1)
        query_states = tf.reshape(query_states, proj_shape)
        key_states = tf.reshape(key_states, proj_shape)

        src_len = tgt_len
        value_states = tf.reshape(value_states, proj_shape)
        attn_weights = query_states @ tf.transpose(key_states, (0, 2, 1))

        attn_weights = tf.reshape(
            attn_weights, (-1, self.num_heads, tgt_len, src_len)
        )
        attn_weights = attn_weights + attention_mask
        attn_weights = tf.reshape(attn_weights, (-1, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ value_states

        attn_output = tf.reshape(
            attn_output, (-1, self.num_heads, tgt_len, self.head_dim)
        )
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (-1, tgt_len, embed_dim))
        return self.out_proj(attn_output)