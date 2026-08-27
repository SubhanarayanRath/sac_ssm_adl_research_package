import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="SAC")
class TemporalBackbone(layers.Layer):
    """Base interface for all temporal backbones in the SAC framework."""
    def __init__(self, units, dropout=0.0, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.return_sequences = return_sequences

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout": self.dropout,
            "return_sequences": self.return_sequences
        })
        return config
