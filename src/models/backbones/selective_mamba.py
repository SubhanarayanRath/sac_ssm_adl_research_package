import tensorflow as tf
from tensorflow.keras import layers
from .base import TemporalBackbone

@tf.keras.utils.register_keras_serializable(package="SAC")
class SelectiveSSMCell(layers.Layer):
    """
    Selective State Space (Mamba) Cell with bounded state update.
    The step size and input projection are functions of the input.
    """
    def __init__(self, units, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.state_size = units

    def build(self, input_shape):
        self.proj_x = layers.Dense(self.units, name="proj_x")
        self.dense_B = layers.Dense(self.units, name="ssm_B")
        self.dense_delta = layers.Dense(self.units, name="ssm_delta")

        # Learned continuous-time state transition matrix (diagonal)
        self.A = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
            name="ssm_A"
        )
        self.built = True

    def call(self, inputs, states, training=None):
        h_prev = states[0]
        x = inputs[:, :-1]
        delta_t = inputs[:, -1:]

        x_proj = self.proj_x(x)
        B = self.dense_B(x)

        # Softplus ensures positive step size. Added 1e-6 for stability.
        delta = tf.nn.softplus(self.dense_delta(x)) * (delta_t + 1e-6)

        # Zero-order hold discretization (bounded update)
        A_bar = tf.exp(self.A * delta)
        candidate = B * x_proj

        # State update
        h_t = (A_bar * h_prev) + ((1.0 - A_bar) * candidate)
        return h_t, [h_t]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout": self.dropout})
        return config

@tf.keras.utils.register_keras_serializable(package="SAC")
class SelectiveMamba(TemporalBackbone):
    """Single-directional Vanilla Mamba Backbone."""
    def __init__(self, units, dropout=0.0, return_sequences=True, **kwargs):
        super().__init__(units, dropout, return_sequences, **kwargs)
        self.rnn = layers.RNN(
            SelectiveSSMCell(units, dropout=0.0),
            return_sequences=return_sequences
        )

    def call(self, inputs, training=None):
        return self.rnn(inputs, training=training)
