import numpy as np
import tensorflow as tf


# ============================================================
# TF.DATA DATASET CREATION
# ============================================================

def create_tf_dataset(
    x,
    y,
    batch_size,
    shuffle=False
):
    ds = tf.data.Dataset.from_tensor_slices(
        (x, y)
    )

    if shuffle:
        buffer_size = min(
            len(next(iter(x.values()))),
            10000
        )

        ds = ds.shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=True
        )

    ds = ds.batch(
        batch_size,
        drop_remainder=False
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


# ============================================================
# MAIN DATASET LOADER
# ============================================================

def load_tar_dataset(
    filepath,
    batch_size=128
):
    print(
        f"Loading clean TAR dataset from {filepath}..."
    )

    data = np.load(
        filepath,
        allow_pickle=False
    )

    # --------------------------------------------------------
    # Build split
    # --------------------------------------------------------

    def build_split(split):

        x = {
            "event_input":
                data[f"{split}_X_event"],

            "type_input":
                data[f"{split}_X_type"],

            "room_input":
                data[f"{split}_X_room"],

            "state_input":
                data[f"{split}_X_state"],

            "time_input":
                data[f"{split}_X_time"],

            "context_input":
                data[f"{split}_X_context"],
        }

        y = {
            "activity":
                data[f"{split}_y_activity"],

            "boundary_seq":
                data[f"{split}_y_boundary"],

            "phase_seq":
                data[f"{split}_y_phase"],
        }

        return x, y

    # --------------------------------------------------------
    # Load original data
    # --------------------------------------------------------

    train_x, train_y = build_split("train")
    val_x, val_y = build_split("val")
    test_x, test_y = build_split("test")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    num_classes = int(data["num_classes"])

    train_class_counts = np.bincount(
        train_y["activity"],
        minlength=num_classes
    )

    meta = {
        "num_events":
            int(data["num_events"]),

        "num_types":
            int(data["num_types"]),

        "num_rooms":
            int(data["num_rooms"]),

        "num_states":
            int(data["num_states"]),

        "num_classes":
            num_classes,

        "max_window":
            int(data["max_window"]),

        "time_dim":
            int(data["time_dim"]),

        "context_dim":
            int(data["context_dim"]),

        "train_class_counts":
            train_class_counts.tolist(),
    }

    # --------------------------------------------------------
    # Print class distribution
    # --------------------------------------------------------

    print("\nOriginal train class counts:")
    print(train_class_counts.tolist())

    # --------------------------------------------------------
    # IMPORTANT:
    # DO NOT oversample or downsample.
    # Use the complete original training set.
    # --------------------------------------------------------

    train_ds = create_tf_dataset(
        train_x,
        train_y,
        batch_size=batch_size,
        shuffle=True
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_ds = create_tf_dataset(
        val_x,
        val_y,
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_ds = create_tf_dataset(
        test_x,
        test_y,
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    print("\nDataset shapes:")

    print(
        "Train:",
        train_x["event_input"].shape,
        train_y["activity"].shape
    )

    print(
        "Validation:",
        val_x["event_input"].shape,
        val_y["activity"].shape
    )

    print(
        "Test:",
        test_x["event_input"].shape,
        test_y["activity"].shape
    )

    print("\nTrain class counts:")
    print(
        meta["train_class_counts"]
    )

    print("\nVocabulary:")
    print(meta)

    return (
        train_ds,
        val_ds,
        test_ds,
        meta
    )