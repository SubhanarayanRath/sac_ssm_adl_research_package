import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, RNN, SeparableConv1D

class TransitionAwareSSMCell(Layer):
    """
    Commit 1: Transition-Aware Selective State-Space Cell.
    This replaces the GRU-like logic from your prototype with true SSM math,
    adding only the retention modulation.
    """
    def __init__(self, units, reset_strength=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.reset_strength = reset_strength

        # TODO: Copy your existing state-space initializations here
        # (e.g., self.A, self.B_proj, self.C_proj, self.dt_proj)

    def call(self, inputs, states, training=False):
        # 1. Unpack features and the boundary gate (b_t)
        x = inputs[:, :-1]
        boundary_gate = inputs[:, -1:]

        current_state = states[0]

        # 2. Compute base Mamba dynamics
        # TODO: Copy your existing Mamba projections here to get base_retention
        # Example:
        # dt = self.dt_proj(x)
        # A = ...
        # base_retention = tf.exp(A * dt)

        # 3. The TAR Novelty: Explicitly modulate the SSM retention
        retention = base_retention * (1.0 - self.reset_strength * boundary_gate)

        # 4. State update and output computation
        # TODO: Copy your existing state update logic here
        # Example:
        # B = self.B_proj(x)
        # next_state = retention * current_state + (B * x * dt)
        # C = self.C_proj(x)
        # output = next_state * C

        return output, [next_state]

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "reset_strength": self.reset_strength
        })
        return config


class TARMambaBackbone(Layer):
    """
    Commit 2: TAR-Mamba Temporal Backbone.
    Strictly isolated to temporal processing. No pooling, no classifier.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        # 1. Causal Transition Detector
        self.detector_conv = SeparableConv1D(
            filters=d_model,
            kernel_size=3,
            padding="causal",
            activation="relu",
            name="tar_detector_conv"
        )
        self.detector_dense = Dense(
            1,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-2.0),
            name="tar_detector_gate"
        )

        # 2. Dual-Timescale SSM Branches
        self.fast_cell = TransitionAwareSSMCell(units=d_model, reset_strength=1.0)
        self.fast_rnn = RNN(self.fast_cell, return_sequences=True, name="tar_fast_branch")

        self.slow_cell = TransitionAwareSSMCell(units=d_model, reset_strength=0.2)
        self.slow_rnn = RNN(self.slow_cell, return_sequences=True, name="tar_slow_branch")

        # 3. Adaptive Feature Fusion
        self.fusion_gate = Dense(d_model, activation="sigmoid", name="tar_fusion_gate")

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, seq_len, d_model)

        # Calculate transition probabilities
        x_conv = self.detector_conv(inputs, training=training)
        b_t = self.detector_dense(x_conv) # (batch_size, seq_len, 1)

        # Concatenate b_t so the SSM cells can unpack it at each timestep
        rnn_inputs = tf.concat([inputs, b_t], axis=-1)

        # Process through fast and slow streams
        fast_out = self.fast_rnn(rnn_inputs, training=training)
        slow_out = self.slow_rnn(rnn_inputs, training=training)

        # Adaptive dimension-wise fusion
        concat_features = tf.concat([fast_out, slow_out], axis=-1)
        z = self.fusion_gate(concat_features)

        # Final sequence representation
        output = z * fast_out + (1.0 - z) * slow_out

        # Returns exactly (batch_size, seq_len, d_model) for AttentionPooling
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config