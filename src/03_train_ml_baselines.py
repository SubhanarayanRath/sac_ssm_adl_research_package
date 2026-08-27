from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from config_utils import load_config, set_global_seed, ensure_project_dirs
from data_utils import load_split


def histogram_features(x, num_events):
    event = x["event_input"]
    context = x["context_input"]
    last_time = x["time_input"][:, -1, :]
    hist = np.zeros((len(event), num_events), dtype=np.float32)
    for i, row in enumerate(event):
        hist[i] = np.bincount(row, minlength=num_events)
    hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
    return np.concatenate([hist, context, last_time], axis=1)


def evaluate(name, model, Xtr, ytr, Xte, yte):
    print(f"[{name}] Starting training on {Xtr.shape[0]} rows...")
    model.fit(Xtr, ytr)

    print(f"[{name}] Training finished. Running predictions on {Xte.shape[0]} test rows...")
    pred = model.predict(Xte)

    return {
        "model": name,
        "accuracy": accuracy_score(yte, pred),
        "balanced_accuracy": balanced_accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(yte, pred, average="weighted", zero_division=0),
    }, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_global_seed(seed)
    ensure_project_dirs()

    print("Loading data splits...")
    tr_x, tr_y, meta = load_split(cfg["data"]["windows_npz"], "train")
    va_x, va_y, _ = load_split(cfg["data"]["windows_npz"], "val")
    te_x, te_y, _ = load_split(cfg["data"]["windows_npz"], "test")

    print("Extracting histogram features and concatenating train/val sets...")
    Xtr = np.concatenate([
        histogram_features(tr_x, meta["num_events"]),
        histogram_features(va_x, meta["num_events"]),
    ], axis=0)
    ytr = np.concatenate([tr_y["activity"], va_y["activity"]])

    Xte = histogram_features(te_x, meta["num_events"])
    yte = te_y["activity"]

    # ---------------------------------------------------------
    # OPTIMIZED RESEARCH MODELS (Max Accuracy, Safe Memory)
    # ---------------------------------------------------------
    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,             # Maximum research power
            max_depth=40,                 # Safe RAM limit, prevents overfitting
            min_samples_split=10,         # Forces model to find real patterns, not noise
            class_weight="balanced",
            random_state=seed,
            n_jobs=4                      # Uses 4 cores (safe for 12GB WSL limit)
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=40,
            min_samples_split=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=4
        ),
    }

    rows = []
    print("\n--- Starting Baseline Training ---")
    for name, model in candidates.items():
        print(f"\n---> Preparing to run: {name}")
        metrics, fitted = evaluate(name, model, Xtr, ytr, Xte, yte)
        rows.append(metrics)

        print(f"[{name}] Saving model to disk...")
        joblib.dump(fitted, f"models/{name}.pkl")

        print(f"[{name}] Final Metrics:")
        print(json.dumps(metrics, indent=4))

    print("\nSaving final CSV results...")
    pd.DataFrame(rows).to_csv("results/ml_baselines.csv", index=False)
    print("✅ All baselines completed successfully!")


if __name__ == "__main__":
    main()