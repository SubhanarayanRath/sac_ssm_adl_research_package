from __future__ import annotations

import argparse

from config_utils import load_config, set_global_seed, ensure_project_dirs
from data_utils import load_split
from train_utils import build_and_compile, targets_for_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["seed"]))
    ensure_project_dirs()

    tr_x, tr_y, meta = load_split(cfg["data"]["windows_npz"], "train")
    va_x, va_y, _ = load_split(cfg["data"]["windows_npz"], "val")

    model = build_and_compile(cfg, meta, tr_y, mode="pretrain")

    # 1. Force Keras to drop the compiled loss metric structure tracking scale_weights
    # We explicitly map the real outputs to their functional objectives, leaving scale_weights out.
    model.compile(
        optimizer=model.optimizer,
        loss={
            "activity": "sparse_categorical_crossentropy",
            "next_event": "mse",
            "gap_bin": "sparse_categorical_crossentropy",
            "adaptive_scale_fusion": "mse"
        },
        metrics={
            "activity": "accuracy",
            "gap_bin": "accuracy"
        }
    )

    tr_targets = targets_for_model(tr_y, len(tr_y["activity"]))
    va_targets = targets_for_model(va_y, len(va_y["activity"]))

    # 2. Filter target inputs to strictly match the newly re-compiled structures
    valid_outputs = ["activity", "next_event", "gap_bin", "adaptive_scale_fusion"]
    tr_targets_clean = {k: v for k, v in tr_targets.items() if k in valid_outputs}
    va_targets_clean = {k: v for k, v in va_targets.items() if k in valid_outputs}

    model.fit(
        tr_x,
        tr_targets_clean,
        validation_data=(va_x, va_targets_clean),
        epochs=int(cfg["training"]["pretrain_epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        callbacks=[
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                monitor="val_next_event_loss", patience=5, restore_best_weights=True
            )
        ],
    )
    model.save_weights("models/sac_ssm_pretrained.weights.h5")
    print("Saved models/sac_ssm_pretrained.weights.h5")


if __name__ == "__main__":
    main()