import numpy as np
import tensorflow as tf
import yaml

from tar_data_utils import load_tar_dataset
from models.backbones.tar_mamba_v1 import TARMambaV1Backbone
from models.layers import AttentionPooling, WeightedScaleFusion

from sklearn.metrics import f1_score


MODEL_PATH = "models/tar_mamba_best.keras"
DATA_PATH = "data/processed/aruba_tar_windows.npz"


def extract_activity_output(predictions):
    if isinstance(predictions, dict):
        return predictions["activity"]

    if isinstance(predictions, (list, tuple)):
        return predictions[0]

    return predictions


def collect_predictions(model, ds):
    y_true = []
    y_prob = []

    for x, y in ds:
        pred = model(x, training=False)
        pred = extract_activity_output(pred)

        y_true.append(y["activity"].numpy())
        y_prob.append(pred.numpy())

    return (
        np.concatenate(y_true, axis=0),
        np.concatenate(y_prob, axis=0),
    )


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def probability_to_logits(prob):
    eps = 1e-7
    prob = np.clip(prob, eps, 1.0 - eps)

    logits = np.log(prob)

    # Softmax is invariant to adding/subtracting the same constant.
    # This representation is therefore sufficient for class-bias search.
    return logits


def apply_bias(prob, bias):
    logits = probability_to_logits(prob)
    logits = logits + bias[None, :]
    return softmax(logits)


def evaluate(y_true, prob):
    pred = np.argmax(prob, axis=1)

    score = f1_score(
        y_true,
        pred,
        labels=np.arange(prob.shape[1]),
        average="macro",
        zero_division=0,
    )

    return score


def main():

    print("=" * 70)
    print("TAR-Mamba VALIDATION CALIBRATION")
    print("=" * 70)

    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    _, val_ds, _, vocab = load_tar_dataset(
        filepath=DATA_PATH,
        batch_size=cfg.get("training", {}).get("batch_size", 128),
    )

    print("\nLoading model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "TARMambaV1Backbone": TARMambaV1Backbone,
            "AttentionPooling": AttentionPooling,
            "WeightedScaleFusion": WeightedScaleFusion,
        },
        safe_mode=False,
        compile=False,
    )

    print("\nCollecting validation probabilities...")

    y_val, p_val = collect_predictions(model, val_ds)

    print("Validation samples:", len(y_val))
    print("Probability shape:", p_val.shape)

    original_score = evaluate(y_val, p_val)

    print("\nOriginal validation Macro-F1:")
    print(f"{original_score:.6f}")

    # ------------------------------------------------------------
    # Grid search class-specific additive logit biases.
    #
    # We deliberately use a small range first.
    # This prevents extreme overfitting to validation.
    # ------------------------------------------------------------

    best_bias = np.zeros(p_val.shape[1], dtype=np.float32)
    best_score = original_score

    print("\nStarting coordinate-search calibration...")

    # Small bias range.
    candidates = np.arange(
    -1.00,
    1.001,
    0.05,
    dtype=np.float32,
)

    # Coordinate descent.
    #
    # One class is changed at a time while the others remain fixed.
    # Repeat several passes until no improvement occurs.

    for iteration in range(5):

        improved = False

        for cls in range(p_val.shape[1]):

            local_best_bias = best_bias[cls]
            local_best_score = best_score

            for value in candidates:

                trial_bias = best_bias.copy()
                trial_bias[cls] = value

                p_cal = apply_bias(p_val, trial_bias)
                score = evaluate(y_val, p_cal)

                if score > local_best_score:
                    local_best_score = score
                    local_best_bias = value

            if local_best_score > best_score:

                best_bias[cls] = local_best_bias
                best_score = local_best_score
                improved = True

                print(
                    f"Iteration {iteration + 1}, "
                    f"class {cls}: "
                    f"bias={local_best_bias:+.2f}, "
                    f"val_macro_f1={best_score:.6f}"
                )

        if not improved:
            print(
                f"\nNo further improvement after iteration "
                f"{iteration + 1}."
            )
            break

    print("\n" + "=" * 70)
    print("CALIBRATION RESULT")
    print("=" * 70)

    print(f"Original validation Macro-F1 : {original_score:.6f}")
    print(f"Calibrated validation Macro-F1: {best_score:.6f}")

    print("\nBest class biases:")

    for i, b in enumerate(best_bias):
        print(f"class {i:2d}: {b:+.2f}")

    np.save(
        "results/tar_mamba_validation_bias.npy",
        best_bias,
    )

    print(
        "\nSaved:",
        "results/tar_mamba_validation_bias.npy",
    )


if __name__ == "__main__":
    main()
