import tensorflow as tf
from tensorflow.keras import layers
from .base import TemporalBackbone

class TransitionDetector(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transition_conv = layers.Conv1D(
            filters=32, kernel_size=3, padding="causal", activation="relu", name="transition_conv"
        )
        self.transition_dense = layers.Dense(
            1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(-2.0), name="boundary_prob"
        )

    def call(self, inputs):
        x = self.transition_conv(inputs)
        return self.transition_dense(x)

class AdaptiveMemoryFusion(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.alpha_proj = layers.Dense(units, activation='sigmoid', name='fusion_alpha')

    def call(self, fast_states, slow_states):
        concat_states = tf.concat([fast_states, slow_states], axis=-1)
        alpha = self.alpha_proj(concat_states)
        return (alpha * fast_states) + ((1.0 - alpha) * slow_states)

@tf.keras.utils.register_keras_serializable(package="SAC")
class MomentumSSMCell(layers.Layer):
    def __init__(self, units, dropout=0.0, momentum=0.9, transition_aware=False, reset_strength=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.momentum = momentum
        self.transition_aware = transition_aware
        self.reset_strength = reset_strength
        self.state_size = [units, units]

    def build(self, input_shape):
        self.proj_x = layers.Dense(self.units, name="proj_x")
        self.dense_B = layers.Dense(self.units, name="ssm_B")
        self.dense_delta = layers.Dense(self.units, name="ssm_delta")
        self.dense_alpha = layers.Dense(self.units, name="ssm_alpha")
        self.A = self.add_weight(shape=(self.units,), initializer=tf.keras.initializers.Constant(-1.0), trainable=True, name="ssm_A")
        self.built = True

    def call(self, inputs, states, training=None):
        h_prev, v_prev = states[0], states[1]

        if self.transition_aware:
            x = inputs[:, :-2]
            delta_t = inputs[:, -2:-1]
            b_t = inputs[:, -1:]
        else:
            x = inputs[:, :-1]
            delta_t = inputs[:, -1:]

        x_proj = self.proj_x(x)
        B = self.dense_B(x)
        delta = tf.nn.softplus(self.dense_delta(x)) * (delta_t + 1e-6)
        A_bar = tf.exp(self.A * delta)

        if self.transition_aware:
            A_bar *= (1.0 - self.reset_strength * b_t)

        candidate = B * x_proj
        v_t = (self.momentum * v_prev) + ((1.0 - A_bar) * candidate)
        alpha_t = tf.math.sigmoid(self.dense_alpha(x))
        h_t = (A_bar * h_prev) + (alpha_t * v_t)

        return h_t, [h_t, v_t]

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units, "dropout": self.dropout, "momentum": self.momentum,
            "transition_aware": self.transition_aware, "reset_strength": self.reset_strength
        })
        return config

@tf.keras.utils.register_keras_serializable(package="SAC")
class TARMambaV1Backbone(TemporalBackbone):
    def __init__(self, units, dropout=0.0, return_sequences=True, **kwargs):
        super().__init__(units, dropout, return_sequences, **kwargs)
        self.units = units
        self.detector = TransitionDetector()
        self.fast_rnn = layers.RNN(MomentumSSMCell(units, dropout=0.0, transition_aware=True, reset_strength=1.0), return_sequences=return_sequences, name='fast_branch')
        self.slow_rnn = layers.RNN(MomentumSSMCell(units, dropout=0.0, transition_aware=True, reset_strength=0.2), return_sequences=return_sequences, name='slow_branch')
        self.dropout_layer = layers.Dropout(dropout)
        self.fusion = AdaptiveMemoryFusion(units)
        self.norm = layers.LayerNormalization()

    def call(self, inputs, training=None):
        b_t = self.detector(inputs)
        rnn_inputs = tf.concat([inputs, b_t], axis=-1)

        fast_states = self.fast_rnn(rnn_inputs, training=training)
        slow_states = self.slow_rnn(rnn_inputs, training=training)

        fast_states = self.dropout_layer(fast_states, training=training)
        slow_states = self.dropout_layer(slow_states, training=training)

        fused_states = self.fusion(fast_states, slow_states)
        fused_states = self.dropout_layer(fused_states, training=training)
        fused_states = self.norm(fused_states)

        x_features = inputs[:, :, :-1]
        if x_features.shape[-1] == fused_states.shape[-1]:
            return x_features + fused_states
        return fused_states