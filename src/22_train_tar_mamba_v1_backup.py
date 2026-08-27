import sys
import os
import yaml
import numpy as np
import tensorflow as tf

from tensorflow.keras import callbacks
from sklearn.metrics import f1_score

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.tar_data_utils import load_tar_dataset
from src.models.modular_sac import build_modular_sac

# ============================================================
# CLASS-BALANCED FOCAL LOSS
# ============================================================

class ClassBalancedFocalLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        class_counts,
        beta=0.9999,
        gamma=1.5,
        min_weight=0.5,
        max_weight=5.0,
        name="class_balanced_focal_loss"
    ):
        super().__init__(name=name)

        counts = np.asarray(
            class_counts,
            dtype=np.float32
        )

        effective_num = 1.0 - np.power(
            beta,
            counts
        )

        weights = (
            1.0 - beta
        ) / np.maximum(
            effective_num,
            1e-8
        )

        # Normalize around 1.0
        weights = weights / np.mean(weights)

        # Prevent extreme rare-class amplification
        weights = np.clip(
            weights,
            min_weight,
            max_weight
        )

        self.class_weights = tf.constant(
            weights,
            dtype=tf.float32
        )

        self.gamma = gamma

        print("\nClass-balanced focal weights:")
        for i, w in enumerate(weights):
            print(
                f"   class {i:2d}: "
                f"{w:.4f} "
                f"(count={int(counts[i])})"
            )

    def call(self, y_true, y_pred):

        y_true = tf.cast(
            tf.reshape(y_true, [-1]),
            tf.int32
        )

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        p_t = tf.gather_nd(
            y_pred,
            tf.stack(
                [
                    tf.range(
                        tf.shape(y_true)[0]
                    ),
                    y_true
                ],
                axis=1
            )
        )

        alpha = tf.gather(
            self.class_weights,
            y_true
        )

        focal_factor = tf.pow(
            1.0 - p_t,
            self.gamma
        )

        loss = (
            -alpha
            * focal_factor
            * tf.math.log(p_t)
        )

        return tf.reduce_mean(loss)


# ============================================================
# VALIDATION MACRO-F1 CALLBACK
# ============================================================

class ValMacroF1Callback(callbacks.Callback):

    def __init__(self, val_ds):
        super().__init__()

        self.val_ds = val_ds
        self.best = -1.0

    def on_epoch_end(self, epoch, logs=None):

        y_true = []
        y_pred = []

        for x, y in self.val_ds:

            predictions = self.model(
                x,
                training=False
            )

            if isinstance(predictions, dict):
                predictions = predictions["activity"]

            elif isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            predictions = np.asarray(
                predictions
            )

            y_true.extend(y.numpy())

            y_pred.extend(
                np.argmax(
                    predictions,
                    axis=1
                )
            )

        score = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        if logs is not None:
            logs["val_macro_f1"] = score

        print(
            f" - val_macro_f1: {score:.4f}"
        )

        if score > self.best:

            self.best = score

            self.model.save(
                "models/tar_mamba_best.keras"
            )

            print(
                f"Epoch {epoch + 1}: "
                f"saved best Macro-F1 model "
                f"({score:.4f})"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "PHASE 10: TAR-Mamba V1 Backbone Training"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    with open(
        "config/default.yaml",
        "r"
    ) as f:

        cfg = yaml.safe_load(f)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_ds, val_ds, test_ds, vocab = (
        load_tar_dataset(
            filepath=
                "data/processed/aruba_tar_windows.npz",

            batch_size=
                cfg.get(
                    "training",
                    {}
                ).get(
                    "batch_size",
                    128
                )
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_modular_sac(

        num_events=
            vocab["num_events"],

        num_types=
            vocab["num_types"],

        num_rooms=
            vocab["num_rooms"],

        num_states=
            vocab["num_states"],

        num_classes=
            vocab["num_classes"],

        max_window=
            vocab["max_window"],

        time_dim=
            vocab["time_dim"],

        context_dim=
            vocab["context_dim"],

        cfg=cfg,

        backbone_type=
            "tar_mamba",

        auxiliary_tasks=False
    )

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    class_counts = np.asarray(
        vocab["train_class_counts"],
        dtype=np.float32
    )

    loss_fn = ClassBalancedFocalLoss(
        class_counts=class_counts,
        beta=0.9999,
        gamma=1.5,
        min_weight=0.5,
        max_weight=5.0
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=2e-4
        ),
        loss=loss_fn,
        metrics=[
            "accuracy"
        ]
    )

    # --------------------------------------------------------
    # Convert target dictionaries to activity only
    # --------------------------------------------------------

    train_activity_ds = train_ds.map(
        lambda x, y: (
            x,
            y["activity"]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_activity_ds = val_ds.map(
        lambda x, y: (
            x,
            y["activity"]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # Macro-F1 callback
    # --------------------------------------------------------

    macro_f1_callback = (
        ValMacroF1Callback(
            val_activity_ds
        )
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callback_list = [
        macro_f1_callback,
        callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history = model.fit(
        train_activity_ds,
        validation_data=val_activity_ds,
        epochs=50,
        callbacks=callback_list
    )


if __name__ == "__main__":
    main()