import os
import sys
import json
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.tar_data_utils import load_tar_dataset
from src.models.backbones.momentum_mamba import (
    MomentumSSMCell,
    MomentumMamba,
)
from src.models.layers import (
    AttentionPooling,
    WeightedScaleFusion,
)


def main():
    print("===================================================")
    print(" PHASE 9.5: Momentum Mamba Evaluation")
    print("===================================================")

    os.makedirs("results", exist_ok=True)

    ###################################################
    # Load Dataset
    ###################################################

    _, _, test_ds, vocab_config = load_tar_dataset(
        filepath="data/processed/aruba_tar_windows.npz",
        batch_size=128,
    )

    print("\nExtracting test targets...")

    y_true = []

    for _, y in test_ds:
        y_true.extend(y["activity"].numpy())

    y_true = np.asarray(y_true)

    ###################################################
    # Load Model
    ###################################################

    model_path = "models/momentum_mamba_baseline.keras"

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        return

    print(f"\nLoading model from {model_path}...")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "MomentumSSMCell": MomentumSSMCell,
            "MomentumMamba": MomentumMamba,
            "AttentionPooling": AttentionPooling,
            "WeightedScaleFusion": WeightedScaleFusion,
        },
        safe_mode=False,
    )

    ###################################################
    # Prediction
    ###################################################

    print("\nRunning predictions...")

    test_inputs = test_ds.map(lambda x, y: x)

    y_pred_prob = model.predict(test_inputs, verbose=1)

    y_pred = np.argmax(y_pred_prob, axis=1)

    ###################################################
    # Metrics
    ###################################################

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n========================================")
    print(" PERFORMANCE METRICS (TEST SET)")
    print("========================================")

    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:\n")
    print(report)

    ###################################################
    # Save Results
    ###################################################

    print("\nSaving evaluation artifacts to 'results/'...")

    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(
        "results/momentum_mamba_predictions.csv",
        index=False,
    )

    with open(
        "results/momentum_mamba_classification_report.txt",
        "w",
    ) as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    np.save(
        "results/momentum_mamba_confusion_matrix.npy",
        cm,
    )

    ###################################################
    # Efficiency
    ###################################################

    print("\n========================================")
    print(" EFFICIENCY METRICS (SIMULATED EDGE)")
    print("========================================")

    params = model.count_params()

    print(f"Total Parameters: {params:,}")

    sample = next(iter(test_inputs.take(1)))

    single_sample = {
        k: v[:1]
        for k, v in sample.items()
    }

    print("Warming up GPU...")

    for _ in range(10):
        model(single_sample, training=False)

    print("Measuring inference latency (100 runs)...")

    start = time.perf_counter()

    for _ in range(100):
        model(single_sample, training=False)

    end = time.perf_counter()

    latency = ((end - start) / 100) * 1000

    print(f"Average Inference Latency: {latency:.2f} ms")

    ###################################################
    # Save JSON Summary
    ###################################################

    summary = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "parameters": int(params),
        "latency_ms": float(latency),
    }

    with open(
        "results/momentum_mamba_metrics.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=4)

    print("\nSUCCESS: Momentum Mamba evaluation completed.")


if __name__ == "__main__":
    main()