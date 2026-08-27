from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from config_utils import load_config, set_global_seed, ensure_project_dirs
from data_utils import load_split


def build_lstm(meta):
    event_in = layers.Input((meta["max_window"],), dtype="int32", name="event_input")
    time_in = layers.Input((meta["max_window"], meta["time_dim"]), name="time_input")

    e = layers.Embedding(meta["num_events"], 48)(event_in)
    x = layers.Concatenate()([e, time_in])

    # Reduced from 96 to 48: Speeds up training on CPU and restricts model capacity to fight overfitting
    x = layers.Bidirectional(layers.LSTM(48))(x)

    # Increased from 0.35 to 0.50: Actively forces the model to generalize instead of memorizing sensor noise
    x = layers.Dropout(0.50)(x)

    # Adjusted dense layer to match the new feature size
    x = layers.Dense(48, activation="relu")(x)
    out = layers.Dense(meta["num_classes"], activation="softmax")(x)

    model = Model([event_in, time_in], out, name="BiLSTM_Baseline")
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

    print("Loading dataset splits...")
    tr_x, tr_y, meta = load_split(cfg["data"]["windows_npz"], "train")
    va_x, va_y, _ = load_split(cfg["data"]["windows_npz"], "val")

    print("Building and compiling the optimized BiLSTM architecture...")
    model = build_lstm(meta)

    # Adjusted patience to 3: Ensures the training stops early if overfitting occurs, saving hours on CPU
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/bilstm_baseline.keras", save_best_only=True),
    ]

    print("Starting training process...")
    model.fit(
        [tr_x["event_input"], tr_x["time_input"]],
        tr_y["activity"],
        validation_data=([va_x["event_input"], va_x["time_input"]], va_y["activity"]),
        epochs=50,
        batch_size=int(cfg["training"]["batch_size"]),
        callbacks=cb,
    )
    print("✅ Training finished and optimal weights successfully restored!")


if __name__ == "__main__":
    main()