import numpy as np
import tensorflow as tf

# IMPORTANT:
# Import custom classes BEFORE loading the model.
from src.models.backbones.tar_mamba_v1 import (
    TARMambaV1Backbone,
    MomentumSSMCell,
    TransitionDetector,
    AdaptiveMemoryFusion,
)

from src.models.modular_sac import build_modular_sac
from src.tar_data_utils import load_tar_dataset


MODEL_PATH = "models/tar_mamba_best.keras"
DATA_PATH = "data/processed/aruba_tar_windows.npz"


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

train_ds, val_ds, test_ds, vocab = load_tar_dataset(
    DATA_PATH,
    batch_size=128
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded successfully.")
print(model.summary())


# ---------------------------------------------------------
# VALIDATION PREDICTIONS
# ---------------------------------------------------------

y_true = []
y_pred = []

for x, y in val_ds:

    pred = model(x, training=False)

    if isinstance(pred, dict):
        pred = pred["activity"]

    elif isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred = np.asarray(pred)

    y_true.extend(y["activity"].numpy())
    y_pred.extend(np.argmax(pred, axis=1))


y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)


# ---------------------------------------------------------
# DISTRIBUTIONS
# ---------------------------------------------------------

num_classes = vocab["num_classes"]

true_counts = np.bincount(
    y_true,
    minlength=num_classes
)

pred_counts = np.bincount(
    y_pred,
    minlength=num_classes
)

print("\nValidation TRUE counts:")
print(true_counts.tolist())

print("\nValidation PREDICTED counts:")
print(pred_counts.tolist())


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------------------------
# PER-CLASS RECALL
# ---------------------------------------------------------

print("\nPer-class recall:")

for i in range(num_classes):

    support = true_counts[i]

    if support == 0:
        recall = float("nan")
    else:
        recall = cm[i, i] / support

    print(
        f"class {i:2d}: "
        f"support={support:6d} "
        f"recall={recall:.4f}"
    )
