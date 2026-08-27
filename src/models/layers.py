\
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="SACSSM")
class ContinuousTimeSSMCell(layers.Layer):
    """
    Input-dependent continuous-time state update.

    The last input channel is treated as a normalized time-gap value.
    """

    def __init__(self, hidden_dim: int, use_time_decay: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.use_time_decay = bool(use_time_decay)

        self.input_proj = layers.Dense(self.hidden_dim, activation="tanh")
        self.rate_proj = layers.Dense(self.hidden_dim)
        self.select_gate = layers.Dense(self.hidden_dim, activation="sigmoid")
        self.output_gate = layers.Dense(self.hidden_dim, activation="sigmoid")

    @property
    def state_size(self):
        return self.hidden_dim

    @property
    def output_size(self):
        return self.hidden_dim

    def call(self, inputs, states):
        prev = states[0]
        x = inputs[:, :-1]
        dt = tf.clip_by_value(inputs[:, -1:], 0.0, 1.0)

        proposal = self.input_proj(x)
        select = self.select_gate(x)
        candidate = select * proposal

        if self.use_time_decay:
            rate = tf.nn.softplus(self.rate_proj(x)) + 1e-4
            retention = tf.exp(-rate * dt)
        else:
            retention = tf.sigmoid(self.rate_proj(x))

        state = retention * prev + (1.0 - retention) * candidate
        output = self.output_gate(x) * state
        return output, [state]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "hidden_dim": self.hidden_dim,
            "use_time_decay": self.use_time_decay,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="SACSSM")
class ContinuousTimeBiSSM(layers.Layer):
    def __init__(self, hidden_dim: int, use_time_decay: bool = True, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.use_time_decay = bool(use_time_decay)
        self.dropout_rate = float(dropout)

        self.forward_rnn = layers.RNN(
            ContinuousTimeSSMCell(hidden_dim, use_time_decay),
            return_sequences=True,
        )
        self.backward_rnn = layers.RNN(
            ContinuousTimeSSMCell(hidden_dim, use_time_decay),
            return_sequences=True,
            go_backwards=True,
        )
        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        forward = self.forward_rnn(inputs, training=training)
        backward = self.backward_rnn(inputs, training=training)
        backward = tf.reverse(backward, axis=[1])
        x = tf.concat([forward, backward], axis=-1)
        x = self.dropout(x, training=training)
        return self.norm(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "hidden_dim": self.hidden_dim,
            "use_time_decay": self.use_time_decay,
            "dropout": self.dropout_rate,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="SACSSM")
class AttentionPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(1)

    def call(self, inputs):
        logits = self.score(inputs)
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)


@tf.keras.utils.register_keras_serializable(package="SACSSM")
class WeightedScaleFusion(layers.Layer):
    def __init__(self, n_scales: int, hidden_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.n_scales = int(n_scales)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = layers.Dense(hidden_dim, activation="relu")
        self.fc2 = layers.Dense(n_scales)

    def call(self, inputs):
        branch_vectors, context = inputs
        concat = tf.concat(branch_vectors + [context], axis=-1)
        weights = tf.nn.softmax(self.fc2(self.fc1(concat)), axis=-1)

        stacked = tf.stack(branch_vectors, axis=1)
        fused = tf.reduce_sum(stacked * tf.expand_dims(weights, -1), axis=1)
        return fused, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_scales": self.n_scales, "hidden_dim": self.hidden_dim})
        return cfg
