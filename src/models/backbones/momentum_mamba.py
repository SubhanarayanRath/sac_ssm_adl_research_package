import tensorflow as tf
from tensorflow.keras import layers
from .base import TemporalBackbone

@tf.keras.utils.register_keras_serializable(package="SAC")
class MomentumSSMCell(layers.Layer):
    """
    Momentum-enhanced Selective State Space Cell.
    Treats velocity (v_t) as a latent state rather than just a smoothed input.
    """
    def __init__(self, units, dropout=0.0, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.momentum = momentum
        self.state_size = [units, units]

    def build(self, input_shape):
        self.proj_x = layers.Dense(self.units, name="proj_x")
        self.dense_B = layers.Dense(self.units, name="ssm_B")
        self.dense_delta = layers.Dense(self.units, name="ssm_delta")
        self.dense_alpha = layers.Dense(self.units, name="ssm_alpha")

        self.A = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
            name="ssm_A"
        )
        self.built = True

    def call(self, inputs, states, training=None):
        h_prev, v_prev = states[0], states[1]
        x = inputs[:, :-1]
        delta_t = inputs[:, -1:]

        x_proj = self.proj_x(x)
        B = self.dense_B(x)

        # Discretization parameters
        delta = tf.nn.softplus(self.dense_delta(x)) * (delta_t + 1e-6)
        A_bar = tf.exp(self.A * delta)
        candidate = B * x_proj

        # 1. Latent Velocity Update: v_t = mu * v_{t-1} + (1 - A_t)(B_t x_t)
        v_t = (self.momentum * v_prev) + ((1.0 - A_bar) * candidate)

        # 2. Input-dependent mixing gate (alpha_t)
        alpha_t = tf.math.sigmoid(self.dense_alpha(x))

        # 3. State Update: h_t = A_t h_{t-1} + alpha_t v_t
        h_t = (A_bar * h_prev) + (alpha_t * v_t)

        return h_t, [h_t, v_t]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout": self.dropout, "momentum": self.momentum})
        return config

@tf.keras.utils.register_keras_serializable(package="SAC")
class MomentumMamba(TemporalBackbone):
    """Single-directional Momentum Mamba Backbone."""
    def __init__(self, units, dropout=0.0, return_sequences=True, **kwargs):
        super().__init__(units, dropout, return_sequences, **kwargs)
        self.rnn = layers.RNN(
            MomentumSSMCell(units, dropout=0.0),
            return_sequences=return_sequences
        )

    def call(self, inputs, training=None):
        return self.rnn(inputs, training=training)
