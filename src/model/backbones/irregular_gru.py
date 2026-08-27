import tensorflow as tf
from tensorflow.keras import layers
from .base import TemporalBackbone

@tf.keras.utils.register_keras_serializable(package="SAC")
class GRUwECell(layers.Layer):
    """GRU cell with observation-dependent time decay."""
    def __init__(self, units, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.state_size = units
        # FIX: Hardcode to 0.0 to avoid Keras 3 scoping bug.
        # Regularization is already handled by the Dropout layer right before this!
        self.gru_cell = layers.GRUCell(units, dropout=0.0)

    def build(self, input_shape):
        feature_dim = input_shape[1] - 1
        # g(x) network to learn how strongly the current observation should decay the past
        self.decay_dense = layers.Dense(self.units, activation="softplus", name="decay_gate")
        self.decay_dense.build((input_shape[0], feature_dim))
        self.gru_cell.build((input_shape[0], feature_dim))
        self.built = True

    def call(self, inputs, states, training=None):
        h_prev = states[0]
        x = inputs[:, :-1]
        delta_t = inputs[:, -1:] # (batch, 1)

        # Calculate decay factor: exp(-g(x) * dt)
        g_x = self.decay_dense(x)
        decay_factor = tf.exp(-g_x * delta_t)

        # Apply decay before standard GRU update
        h_decayed = h_prev * decay_factor
        return self.gru_cell(x, [h_decayed], training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout": self.dropout})
        return config

@tf.keras.utils.register_keras_serializable(package="SAC")
class IrregularTimeGRU(TemporalBackbone):
    """Single-directional GRUwE Backbone to ensure exact hidden dim match."""
    def __init__(self, units, dropout=0.0, return_sequences=True, **kwargs):
        super().__init__(units, dropout, return_sequences, **kwargs)

        # SINGLE direction only! Output dim will exactly match 'units'
        self.rnn = layers.RNN(
            GRUwECell(units, dropout=dropout),
            return_sequences=return_sequences
        )

    def call(self, inputs, training=None):
        return self.rnn(inputs, training=training)