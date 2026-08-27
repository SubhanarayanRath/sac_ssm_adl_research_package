import os
import sys
import json
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tar_data_utils import load_tar_dataset
from src.models.backbones.selective_mamba import SelectiveSSMCell, SelectiveMamba
from src.models.layers import AttentionPooling, WeightedScaleFusion

def main():
    print("===================================================")
    print(" PHASE 8.5: Evaluating Vanilla Mamba")
    print("===================================================")

    os.makedirs("results", exist_ok=True)

    _, _, test_ds, _ = load_tar_dataset(
        filepath="data/processed/aruba_tar_windows.npz",
        batch_size=128
    )

    print("\nExtracting test targets...")
    y_true = []
    for x, y in test_ds:
        y_true.extend(y["activity"].numpy())
    y_true = np.array(y_true)

    model_path = "models/mamba_baseline.keras"
    print(f"\nLoading entire model from {model_path}...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "SelectiveSSMCell": SelectiveSSMCell,
            "SelectiveMamba": SelectiveMamba,
            "AttentionPooling": AttentionPooling,
            "WeightedScaleFusion": WeightedScaleFusion,
        },
        safe_mode=False
    )

    print("\nRunning predictions on test set...")
    test_inputs = test_ds.map(lambda x, y: x)
    y_pred_probs = model.predict(test_inputs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "="*40)
    print(" PERFORMANCE METRICS (TEST SET)")
    print("="*40)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}\n")

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print("Classification Report:")
    print(report)

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv("results/mamba_predictions.csv", index=False)
    with open("results/mamba_classification_report.txt", "w") as f: f.write(report)
    np.save("results/mamba_confusion_matrix.npy", confusion_matrix(y_true, y_pred))

    print("\n" + "="*40)
    print(" EFFICIENCY METRICS (SIMULATED EDGE)")
    print("="*40)
    param_count = model.count_params()
    print(f"Total Parameters: {param_count:,}")

    sample_input = next(iter(test_inputs.take(1)))
    single_sample = {k: v[0:1] for k, v in sample_input.items()}

    print("Warming up GPU for latency test...")
    for _ in range(10): _ = model(single_sample, training=False)

    print("Measuring inference latency (100 passes)...")
    start = time.perf_counter()
    for _ in range(100): _ = model(single_sample, training=False)
    end = time.perf_counter()

    avg_latency_ms = ((end - start) / 100) * 1000
    print(f"Average Inference Latency (Batch=1): {avg_latency_ms:.2f} ms")

    summary = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "parameters": int(param_count),
        "latency_ms": float(avg_latency_ms),
    }
    with open("results/mamba_metrics.json", "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()
