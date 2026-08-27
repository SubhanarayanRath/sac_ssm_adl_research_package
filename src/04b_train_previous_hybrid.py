\
from __future__ import annotations

import argparse

import tensorflow as tf
from tensorflow.keras import layers, Model

from config_utils import load_config, set_global_seed, ensure_project_dirs
from data_utils import load_split


def transformer_encoder(x, num_heads=4, key_dim=24, ff_dim=128, dropout=0.2):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(x.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)


def build_model(meta):
    event_in = layers.Input((meta["max_window"],), dtype="int32", name="event_input")
    time_in = layers.Input((meta["max_window"], meta["time_dim"]), name="time_input")

    e = layers.Embedding(meta["num_events"], 64)(event_in)
    x = layers.Concatenate()([e, time_in])
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = transformer_encoder(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    scores = layers.Dense(1, activation="tanh")(x)
    weights = tf.nn.softmax(scores, axis=1)
    x = tf.reduce_sum(x * weights, axis=1)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(meta["num_classes"], activation="softmax")(x)

    model = Model([event_in, time_in], out, name="Previous_CNN_Transformer_BiLSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["seed"]))
    ensure_project_dirs()

    tr_x, tr_y, meta = load_split(cfg["data"]["windows_npz"], "train")
    va_x, va_y, _ = load_split(cfg["data"]["windows_npz"], "val")

    model = build_model(meta)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(
            "models/previous_hybrid_chronological.keras", save_best_only=True
        ),
    ]

    model.fit(
        [tr_x["event_input"], tr_x["time_input"]],
        tr_y["activity"],
        validation_data=(
            [va_x["event_input"], va_x["time_input"]],
            va_y["activity"],
        ),
        epochs=60,
        batch_size=int(cfg["training"]["batch_size"]),
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
