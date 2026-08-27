import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)


PREDICTIONS_PATH = "results/tar_mamba_predictions.csv"
BIAS_PATH = "results/tar_mamba_validation_bias.npy"


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def evaluate(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(12),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print(f"Cohen's Kappa     : {kappa:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(12),
            digits=4,
            zero_division=0,
        )
    )

    return macro_f1


def main():

    print("=" * 60)
    print("TAR-Mamba VALIDATION-DERIVED TEST CALIBRATION")
    print("=" * 60)

    df = pd.read_csv(PREDICTIONS_PATH)

    y_true = df["y_true"].to_numpy()

    probability_columns = [
        f"p{i}" for i in range(12)
    ]

    probabilities = df[probability_columns].to_numpy(
        dtype=np.float64
    )

    bias = np.load(BIAS_PATH).astype(np.float64)

    print("\nValidation-derived biases:")

    for i, b in enumerate(bias):
        print(f"class {i:2d}: {b:+.2f}")

    # ---------------------------------------------------------
    # Original prediction
    # ---------------------------------------------------------

    original_pred = np.argmax(
        probabilities,
        axis=1,
    )

    original_f1 = evaluate(
        "ORIGINAL TEST RESULT",
        y_true,
        original_pred,
    )

    # ---------------------------------------------------------
    # Apply validation-derived class biases
    # ---------------------------------------------------------

    eps = 1e-7

    clipped = np.clip(
        probabilities,
        eps,
        1.0,
    )

    logits = np.log(clipped)

    calibrated_logits = (
        logits + bias[None, :]
    )

    calibrated_probabilities = softmax(
        calibrated_logits
    )

    calibrated_pred = np.argmax(
        calibrated_probabilities,
        axis=1,
    )

    calibrated_f1 = evaluate(
        "CALIBRATED TEST RESULT",
        y_true,
        calibrated_pred,
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(
        f"Original Test Macro F1   : {original_f1:.4f}"
    )

    print(
        f"Calibrated Test Macro F1 : {calibrated_f1:.4f}"
    )

    print(
        f"Absolute improvement     : "
        f"{calibrated_f1 - original_f1:+.4f}"
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        calibrated_pred,
        labels=np.arange(12),
    )

    print("\nCalibrated confusion matrix:")
    print("Rows=true, columns=predicted")
    print(cm)

    # ---------------------------------------------------------
    # Save calibrated predictions
    # ---------------------------------------------------------

    output = df.copy()

    output["original_pred"] = original_pred
    output["calibrated_pred"] = calibrated_pred

    output.to_csv(
        "results/tar_mamba_calibrated_predictions.csv",
        index=False,
    )

    np.save(
        "results/tar_mamba_calibrated_confusion_matrix.npy",
        cm,
    )

    print(
        "\nSaved calibrated predictions to:"
    )
    print(
        "results/tar_mamba_calibrated_predictions.csv"
    )


if __name__ == "__main__":
    main()
