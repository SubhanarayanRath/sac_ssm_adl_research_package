import os
import sys
import json
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tar_data_utils import load_tar_dataset
from src.models.layers import AttentionPooling, WeightedScaleFusion
from src.models.backbones.tar_mamba_v1 import (
    TransitionDetector,
    AdaptiveMemoryFusion,
    MomentumSSMCell,
    TARMambaV1Backbone
)


def main():
    print("=" * 60)
    print("PHASE 11: TAR-Mamba Evaluation & Benchmarking")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    dataset_path = "data/processed/aruba_tar_windows_w60_s5.npz"

    print("\nLoading test dataset...")
    _, _, test_ds, vocab = load_tar_dataset(
        filepath=dataset_path,
        batch_size=128
    )

    print("\nDataset information:")
    print(f"Test samples : {test_ds.cardinality().numpy() * 128}")
    print(f"Classes      : {vocab['num_classes']}")
    print(f"Window       : {vocab['max_window']}")

    # ========================================================
    # 2. EXTRACT TRUE LABELS
    # ========================================================

    print("\nExtracting test targets...")

    y_true = []

    for _, y in test_ds:
        y_true.extend(y["activity"].numpy())

    y_true = np.asarray(y_true, dtype=np.int32)

    print(f"Total test labels: {len(y_true)}")

    # ========================================================
    # 3. LOAD BEST MODEL
    # ========================================================

    model_path = "models/tar_mamba_best.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model not found: {model_path}"
        )

    print(f"\nLoading best model:")
    print(model_path)

    custom_objects = {
        "TransitionDetector": TransitionDetector,
        "AdaptiveMemoryFusion": AdaptiveMemoryFusion,
        "MomentumSSMCell": MomentumSSMCell,
        "TARMambaV1Backbone": TARMambaV1Backbone,
        "AttentionPooling": AttentionPooling,
        "WeightedScaleFusion": WeightedScaleFusion
    }

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        safe_mode=False,
        compile=False
    )

    print("\nModel loaded successfully.")
    print(f"Parameters: {model.count_params():,}")

    # ========================================================
    # 4. CREATE INPUT-ONLY TEST DATASET
    # ========================================================

    test_inputs = test_ds.map(
        lambda x, y: x,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # ========================================================
    # 5. RUN PREDICTIONS
    # ========================================================

    print("\nRunning test-set predictions...")

    predictions = model.predict(
        test_inputs,
        verbose=1
    )

    if isinstance(predictions, dict):
        if "activity" in predictions:
            y_pred_prob = predictions["activity"]
        elif "activity_output" in predictions:
            y_pred_prob = predictions["activity_output"]
        else:
            raise ValueError(
                "Activity output not found in model dictionary."
            )
    elif isinstance(predictions, (list, tuple)):
        y_pred_prob = predictions[0]
    else:
        y_pred_prob = predictions

    y_pred_prob = np.asarray(
        y_pred_prob,
        dtype=np.float32
    )

    print(f"\nPrediction shape: {y_pred_prob.shape}")

    if y_pred_prob.shape[0] != len(y_true):
        raise ValueError(
            f"Prediction/label mismatch: "
            f"{y_pred_prob.shape[0]} predictions vs "
            f"{len(y_true)} labels"
        )

    # ========================================================
    # 6. PREDICTED CLASSES
    # ========================================================

    y_pred = np.argmax(
        y_pred_prob,
        axis=1
    )

    confidence = np.max(
        y_pred_prob,
        axis=1
    )

    # ========================================================
    # 7. CORE METRICS
    # ========================================================

    labels = np.arange(
        vocab["num_classes"]
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    balanced_accuracy = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    # ========================================================
    # 8. PRINT METRICS
    # ========================================================

    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print(f"Cohen's Kappa     : {kappa:.4f}")

    # ========================================================
    # 9. CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # ========================================================
    # 10. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("\nConfusion Matrix:")
    print(cm)

    np.save(
        "results/tar_mamba_confusion_matrix.npy",
        cm
    )

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(
        "TAR-Mamba Confusion Matrix"
    )
    plt.xlabel(
        "Predicted Activity"
    )
    plt.ylabel(
        "True Activity"
    )
    plt.tight_layout()

    plt.savefig(
        "results/tar_mamba_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # 11. SAVE PREDICTIONS
    # ========================================================

    prediction_data = {
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence
    }

    for i in range(
        y_pred_prob.shape[1]
    ):
        prediction_data[
            f"p{i}"
        ] = y_pred_prob[:, i]

    pd.DataFrame(
        prediction_data
    ).to_csv(
        "results/tar_mamba_predictions.csv",
        index=False
    )

    # ========================================================
    # 12. SAVE CLASSIFICATION REPORT
    # ========================================================

    with open(
        "results/tar_mamba_classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    # ========================================================
    # 13. PARAMETER COUNT
    # ========================================================

    params = model.count_params()

    print("\n" + "=" * 60)
    print("MODEL SIZE")
    print("=" * 60)
    print(f"Total Parameters : {params:,}")

    # ========================================================
    # 14. INFERENCE LATENCY
    # ========================================================

    print("\nPreparing latency benchmark...")

    sample = next(
        iter(
            test_inputs.take(1)
        )
    )

    single_sample = {
        key: value[:1]
        for key, value in sample.items()
    }

    print("Warming up model...")

    for _ in range(10):
        model(
            single_sample,
            training=False
        )

    print("Measuring inference latency...")

    start = time.perf_counter()

    for _ in range(100):
        model(
            single_sample,
            training=False
        )

    end = time.perf_counter()

    latency_ms = (
        (end - start) / 100
    ) * 1000

    print(
        f"Average Inference Latency : "
        f"{latency_ms:.2f} ms"
    )

    # ========================================================
    # 15. SAVE JSON SUMMARY
    # ========================================================

    summary = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "parameters": int(params),
        "latency_ms": float(latency_ms),
        "test_samples": int(len(y_true)),
        "num_classes": int(vocab["num_classes"]),
        "window_size": int(vocab["max_window"])
    }

    with open(
        "results/tar_mamba_metrics.json",
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    # ========================================================
    # 16. FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Test Accuracy          : {accuracy:.4f}"
    )
    print(
        f"Test Balanced Accuracy : {balanced_accuracy:.4f}"
    )
    print(
        f"Test Macro F1          : {macro_f1:.4f}"
    )
    print(
        f"Test Weighted F1       : {weighted_f1:.4f}"
    )
    print(
        f"MCC                    : {mcc:.4f}"
    )
    print(
        f"Cohen's Kappa          : {kappa:.4f}"
    )
    print(
        f"Parameters             : {params:,}"
    )
    print(
        f"Latency                : {latency_ms:.2f} ms"
    )

    print("\nSaved results:")
    print("  results/tar_mamba_metrics.json")
    print("  results/tar_mamba_predictions.csv")
    print("  results/tar_mamba_classification_report.txt")
    print("  results/tar_mamba_confusion_matrix.npy")
    print("  results/tar_mamba_confusion_matrix.png")


if __name__ == "__main__":
    main()