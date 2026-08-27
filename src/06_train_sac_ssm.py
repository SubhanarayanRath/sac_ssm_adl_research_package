from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import load_config, set_global_seed, ensure_project_dirs
from data_utils import load_split
from train_utils import build_and_compile, targets_for_model, callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--name", default="sac_ssm_full")
    parser.add_argument("--single-scale", action="store_true")
    parser.add_argument("--no-semantics", action="store_true")
    parser.add_argument("--no-time-decay", action="store_true")
    parser.add_argument("--no-aux", action="store_true")
    parser.add_argument("--no-pretrain", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["seed"]))
    ensure_project_dirs()

    tr_x, tr_y, meta = load_split(cfg["data"]["windows_npz"], "train")
    va_x, va_y, _ = load_split(cfg["data"]["windows_npz"], "val")

    model = build_and_compile(
        cfg, meta, tr_y, mode="finetune",
        single_scale=args.single_scale,
        no_semantics=args.no_semantics,
        no_time_decay=args.no_time_decay,
        no_aux=args.no_aux,
    )

    pretrain_path = Path("models/sac_ssm_pretrained.weights.h5")
    if not args.no_pretrain and pretrain_path.exists():
        model.load_weights(pretrain_path, skip_mismatch=True)
        print("Loaded self-supervised pretrained weights.")

    # 1. Explicitly re-compile the model to strip the 'scale_weights' bug
    # and force the GPU to focus 100% on the 'activity' classification head.
    model.compile(
        optimizer=model.optimizer,
        loss={
            "activity": "sparse_categorical_crossentropy",
            "next_event": "mse",
            "gap_bin": "sparse_categorical_crossentropy",
            "adaptive_scale_fusion": "mse"
        },
        loss_weights={
            "activity": 1.0,
            "next_event": 0.0,  # Ignored during finetuning
            "gap_bin": 0.0,     # Ignored during finetuning
            "adaptive_scale_fusion": 0.0 # Ignored during finetuning
        },
        metrics={
            "activity": "accuracy"
        }
    )

    model_path = f"models/{args.name}.keras"

    # 2. Generate the target dictionaries
    tr_targets_raw = targets_for_model(tr_y, len(tr_y["activity"]))
    va_targets_raw = targets_for_model(va_y, len(va_y["activity"]))

    # 3. Filter out the phantom 'scale_weights' key from the data inputs
    valid_outputs = ["activity", "next_event", "gap_bin", "adaptive_scale_fusion"]
    tr_targets = {k: v for k, v in tr_targets_raw.items() if k in valid_outputs}
    va_targets = {k: v for k, v in va_targets_raw.items() if k in valid_outputs}

    # 4. Train the model using the clean filtered targets
    model.fit(
        tr_x,
        tr_targets,
        validation_data=(va_x, va_targets),
        epochs=int(cfg["training"]["finetune_epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        callbacks=callbacks(model_path, int(cfg["training"]["patience"])),
    )
    print(f"Saved best model: {model_path}")


if __name__ == "__main__":
    main()