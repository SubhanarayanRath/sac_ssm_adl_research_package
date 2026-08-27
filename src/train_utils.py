\
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_utils import load_split
from losses import ClassBalancedFocalLoss, effective_num_class_weights
from models.sac_ssm import build_sac_ssm


def build_and_compile(cfg, meta, train_y, mode="finetune",
                      single_scale=False, no_semantics=False,
                      no_time_decay=False, no_aux=False):
    model = build_sac_ssm(
        num_events=meta["num_events"],
        num_types=meta["num_types"],
        num_rooms=meta["num_rooms"],
        num_states=meta["num_states"],
        num_classes=meta["num_classes"],
        num_gap_bins=meta["num_gap_bins"],
        max_window=meta["max_window"],
        time_dim=meta["time_dim"],
        context_dim=meta["context_dim"],
        cfg=cfg,
        single_scale=single_scale,
        no_semantics=no_semantics,
        no_time_decay=no_time_decay,
    )

    tr = cfg["training"]
    optimizer = tf.keras.optimizers.Adam(float(tr["learning_rate"]))

    if mode == "pretrain":
        losses = {
            "activity": "sparse_categorical_crossentropy",
            "next_event": "sparse_categorical_crossentropy",
            "gap_bin": "sparse_categorical_crossentropy",
            "scale_weights": None,
        }
        loss_weights = {
            "activity": 0.0,
            "next_event": 1.0,
            "gap_bin": 0.5,
            "scale_weights": 0.0,
        }
    else:
        weights = effective_num_class_weights(
            train_y["activity"],
            meta["num_classes"],
            beta=float(tr["effective_num_beta"]),
        )
        activity_loss = ClassBalancedFocalLoss(
            weights,
            gamma=float(tr["focal_gamma"]),
        )
        losses = {
            "activity": activity_loss,
            "next_event": "sparse_categorical_crossentropy",
            "gap_bin": "sparse_categorical_crossentropy",
            "scale_weights": None,
        }
        if no_aux:
            next_w = 0.0
            gap_w = 0.0
        else:
            next_w = float(tr["next_event_loss_weight"])
            gap_w = float(tr["gap_loss_weight"])
        loss_weights = {
            "activity": float(tr["activity_loss_weight"]),
            "next_event": next_w,
            "gap_bin": gap_w,
            "scale_weights": 0.0,
        }

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "activity": ["accuracy"],
            "next_event": ["accuracy"],
            "gap_bin": ["accuracy"],
        },
    )
    return model


def targets_for_model(y, n):
    return {
        "activity": y["activity"],
        "next_event": y["next_event"],
        "gap_bin": y["gap_bin"],
        "scale_weights": np.zeros((n, 1), dtype=np.float32),
    }


def callbacks(model_path, patience):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_activity_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_activity_loss",
            patience=max(2, patience // 2),
            factor=0.5,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_activity_loss",
            save_best_only=True,
        ),
    ]
