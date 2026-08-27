import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.tar_data_utils import load_tar_dataset

_, val_ds, _, vocab = load_tar_dataset(
    "data/processed/aruba_tar_windows.npz",
    batch_size=128
)

model = tf.keras.models.load_model(
    "models/tar_mamba_best.keras"
)

y_true = []
y_pred = []

for x, y in val_ds:

    predictions = model(
        x,
        training=False
    )

    if isinstance(predictions, dict):
        predictions = predictions["activity"]

    elif isinstance(predictions, (list, tuple)):
        predictions = predictions[0]

    y_true.extend(
        y["activity"].numpy()
    )

    y_pred.extend(
        np.argmax(
            predictions.numpy(),
            axis=1
        )
    )

print("\n" + "=" * 60)
print("VALIDATION CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(vocab["num_classes"])),
        digits=4,
        zero_division=0
    )
)

print("\n" + "=" * 60)
print("VALIDATION CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(vocab["num_classes"]))
    )
)
