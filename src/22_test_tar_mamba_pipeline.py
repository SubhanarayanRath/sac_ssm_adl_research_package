import os
import sys
import yaml
import tensorflow as tf

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.tar_data_utils import load_tar_dataset
from src.models.modular_sac import build_modular_sac


def main():

    print("=" * 70)
    print("TAR-MAMBA PIPELINE SMOKE TEST")
    print("=" * 70)

    dataset_path = (
        "data/processed/"
        "aruba_tar_windows_w60_s5.npz"
    )

    with open(
        "config/default.yaml",
        "r"
    ) as f:
        cfg = yaml.safe_load(f)

    batch_size = (
        cfg.get(
            "training",
            {}
        ).get(
            "batch_size",
            128
        )
    )

    print("\nLoading dataset...")

    train_ds, val_ds, test_ds, vocab = (
        load_tar_dataset(
            filepath=dataset_path,
            batch_size=batch_size
        )
    )

    print("\nBuilding model...")

    model = build_modular_sac(

        num_events=vocab["num_events"],
        num_types=vocab["num_types"],
        num_rooms=vocab["num_rooms"],
        num_states=vocab["num_states"],
        num_classes=vocab["num_classes"],
        max_window=vocab["max_window"],
        time_dim=vocab["time_dim"],
        context_dim=vocab["context_dim"],
        cfg=cfg,
        backbone_type="tar_mamba",
        auxiliary_tasks=False
    )

    model.summary()

    print("\nFetching one training batch...")

    x, y = next(iter(train_ds))

    print("\nINPUT SHAPES")

    for key, value in x.items():
        print(
            f"{key:20s}: "
            f"{value.shape} "
            f"{value.dtype}"
        )

    print("\nTARGET SHAPES")

    for key, value in y.items():
        print(
            f"{key:20s}: "
            f"{value.shape} "
            f"{value.dtype}"
        )

    print("\nRunning forward pass...")

    prediction = model(
        x,
        training=False
    )

    print("\nMODEL OUTPUT")

    if isinstance(prediction, dict):

        for key, value in prediction.items():

            print(
                f"{key:20s}: "
                f"{value.shape}"
            )

    elif isinstance(prediction, (list, tuple)):

        for i, value in enumerate(prediction):

            print(
                f"output_{i:2d}: "
                f"{value.shape}"
            )

    else:

        print(
            prediction.shape
        )

    print("\nSUCCESS")
    print(
        "Dataset -> Model forward pass works correctly."
    )


if __name__ == "__main__":
    main()
