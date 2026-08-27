from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from config_utils import load_config, ensure_project_dirs
from data_utils import load_split
from losses import ClassBalancedFocalLoss
from models.layers import ContinuousTimeSSMCell, ContinuousTimeBiSSM, AttentionPooling, WeightedScaleFusion


def expected_calibration_error(y_true, probs, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def label_names(mapping_path):
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    inv = {v: k for k, v in mapping["labels"].items()}
    return [inv[i] for i in range(len(inv))]


def load_model(path):
    custom = {
        "ClassBalancedFocalLoss": ClassBalancedFocalLoss,
        "ContinuousTimeSSMCell": ContinuousTimeSSMCell,
        "ContinuousTimeBiSSM": ContinuousTimeBiSSM,
        "AttentionPooling": AttentionPooling,
        "WeightedScaleFusion": WeightedScaleFusion,
    }
    # safe_mode=False added below to allow Lambda layers to load
    return tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="models/sac_ssm_full.keras")
    parser.add_argument("--prefix", default="sac_ssm_full")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()
    te_x, te_y, _ = load_split(cfg["data"]["windows_npz"], "test")
    names = label_names(cfg["data"]["mappings_json"])

    model = load_model(args.model)
    outputs = model.predict(te_x, batch_size=256, verbose=1)
    probs = outputs[0]
    scale_weights = outputs[3]
    y_true = te_y["activity"]
    y_pred = probs.argmax(axis=1)

    metrics = {
        "model": args.prefix,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "ece": expected_calibration_error(y_true, probs),
        "parameters": model.count_params(),
    }

    # Single-sample latency.
    sample = {k: v[:1] for k, v in te_x.items()}
    for _ in range(10):
        model(sample, training=False)
    times = []
    for _ in range(int(cfg["evaluation"]["latency_runs"])):
        t0 = time.perf_counter()
        model(sample, training=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    metrics["latency_ms_mean"] = float(np.mean(times))
    metrics["latency_ms_p95"] = float(np.percentile(times, 95))

    Path(f"results/{args.prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    report = classification_report(
        y_true, y_pred, labels=list(range(len(names))),
        target_names=names, zero_division=0, digits=4
    )
    Path(f"results/{args.prefix}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix: {args.prefix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"results/{args.prefix}_confusion_matrix.png", dpi=300)
    plt.close()

    pd.DataFrame(scale_weights, columns=[f"scale_{i}" for i in range(scale_weights.shape[1])]).to_csv(
        f"results/{args.prefix}_scale_weights.csv", index=False
    )

    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    main()