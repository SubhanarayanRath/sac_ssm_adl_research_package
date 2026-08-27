\
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config_utils import load_config
from losses import ClassBalancedFocalLoss
from models.layers import ContinuousTimeSSMCell, ContinuousTimeBiSSM, AttentionPooling, WeightedScaleFusion


def invert_vocab(vocab):
    return {k: int(v) for k, v in vocab.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="models/sac_ssm_full.keras")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mapping = json.loads(Path(cfg["data"]["mappings_json"]).read_text(encoding="utf-8"))
    labels_inv = {v: k for k, v in mapping["labels"].items()}

    custom = {
        "ClassBalancedFocalLoss": ClassBalancedFocalLoss,
        "ContinuousTimeSSMCell": ContinuousTimeSSMCell,
        "ContinuousTimeBiSSM": ContinuousTimeBiSSM,
        "AttentionPooling": AttentionPooling,
        "WeightedScaleFusion": WeightedScaleFusion,
    }
    model = tf.keras.models.load_model(args.model, custom_objects=custom, compile=False)

    # Reuse the prepared test windows as a controlled real-time replay.
    data = np.load(cfg["data"]["windows_npz"])
    n = len(data["test_X_event"])
    for i in range(min(n, 500)):
        x = {
            "event_input": data["test_X_event"][i:i+1],
            "type_input": data["test_X_type"][i:i+1],
            "room_input": data["test_X_room"][i:i+1],
            "state_input": data["test_X_state"][i:i+1],
            "time_input": data["test_X_time"][i:i+1],
            "context_input": data["test_X_context"][i:i+1],
        }
        activity_probs, _, _, scale_weights = model.predict(x, verbose=0)
        pred = int(activity_probs.argmax(axis=1)[0])
        conf = float(activity_probs.max())
        print(
            f"{i:04d} activity={labels_inv[pred]:20s} "
            f"confidence={conf:.3f} scales={scale_weights[0].round(3)}"
        )


if __name__ == "__main__":
    main()
