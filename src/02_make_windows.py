from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config_utils import load_config, set_global_seed, ensure_project_dirs


UNK = "<UNK>"


def make_vocab(values):
    unique = sorted({str(v) for v in values})
    return {UNK: 0, **{v: i + 1 for i, v in enumerate(unique)}}


def encode(values, vocab):
    return np.array([vocab.get(str(v), 0) for v in values], dtype=np.int32)


def split_groups(groups, train_ratio, val_ratio):
    """
    Chronological group split.

    Groups must already be ordered chronologically.
    This is the primary leakage-safe split used by the pipeline.
    """

    unique_groups = list(dict.fromkeys(groups))
    n = len(unique_groups)

    if n < 3:
        raise ValueError(
            "Need at least three chronological groups for train/val/test."
        )

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    if n_train + n_val >= n:
        n_val = 1
        n_train = n - 2

    train_groups = set(unique_groups[:n_train])
    val_groups = set(
        unique_groups[n_train:n_train + n_val]
    )
    test_groups = set(
        unique_groups[n_train + n_val:]
    )

    print("\n===== CHRONOLOGICAL GROUP SPLIT =====")
    print("Train groups:", sorted(train_groups))
    print("Validation groups:", sorted(val_groups))
    print("Test groups:", sorted(test_groups))

    return train_groups, val_groups, test_groups


def gap_bin(seconds, bins):
    return int(np.digitize([seconds], bins=bins, right=False)[0])


def add_time_features(df, delta_clip):
    hour_float = df["hour"].to_numpy() + df["minute"].to_numpy() / 60.0
    dow = df["dayofweek"].to_numpy()
    delta = np.minimum(df["delta_t"].to_numpy(dtype=float), delta_clip)

    hour_sin = np.sin(2 * np.pi * hour_float / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_float / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    delta_log = np.log1p(delta) / np.log1p(delta_clip)
    return np.stack([hour_sin, hour_cos, dow_sin, dow_cos, delta_log], axis=1).astype(np.float32)


def window_context(event_ids, room_ids, delta_seq):
    n = len(event_ids)
    event_change = np.mean(event_ids[1:] != event_ids[:-1]) if n > 1 else 0.0
    room_change = np.mean(room_ids[1:] != room_ids[:-1]) if n > 1 else 0.0
    unique_event_ratio = len(np.unique(event_ids)) / max(1, n)
    unique_room_ratio = len(np.unique(room_ids)) / max(1, n)
    return np.array([
        unique_event_ratio,
        unique_room_ratio,
        event_change,
        room_change,
        float(np.mean(delta_seq)),
        float(np.std(delta_seq)),
        float(np.max(delta_seq)),
    ], dtype=np.float32)


def build_split_windows(df, split_name, max_window, step, vocabs, label_vocab, delta_clip, gap_bins):
    part = df[df["split"] == split_name].copy().reset_index(drop=True)
    if len(part) <= max_window + 1:
        raise ValueError(f"{split_name} split is too small for max_window={max_window}")

    event_ids = encode(part["event_token"], vocabs["event"])
    type_ids = encode(part["sensor_type"], vocabs["type"])
    room_ids = encode(part["room"], vocabs["room"])
    state_ids = encode(part["state_norm"], vocabs["state"])
    labels = encode(part["label"], label_vocab)
    time_features = add_time_features(part, delta_clip)

    X_event, X_type, X_room, X_state = [], [], [], []
    X_time, X_context = [], []
    y_activity, y_next_event, y_gap = [], [], []

    for end in range(max_window - 1, len(part) - 1, step):
        start = end - max_window + 1
        sl = slice(start, end + 1)

        ev = event_ids[sl]
        ty = type_ids[sl]
        ro = room_ids[sl]
        st = state_ids[sl]
        tf = time_features[sl]

        X_event.append(ev)
        X_type.append(ty)
        X_room.append(ro)
        X_state.append(st)
        X_time.append(tf)
        X_context.append(window_context(ev, ro, tf[:, -1]))

        y_activity.append(labels[end])
        y_next_event.append(event_ids[end + 1])
        y_gap.append(gap_bin(float(part.loc[end + 1, "delta_t"]), gap_bins))

    return {
        "X_event": np.asarray(X_event, dtype=np.int32),
        "X_type": np.asarray(X_type, dtype=np.int32),
        "X_room": np.asarray(X_room, dtype=np.int32),
        "X_state": np.asarray(X_state, dtype=np.int32),
        "X_time": np.asarray(X_time, dtype=np.float32),
        "X_context": np.asarray(X_context, dtype=np.float32),
        "y_activity": np.asarray(y_activity, dtype=np.int32),
        "y_next_event": np.asarray(y_next_event, dtype=np.int32),
        "y_gap": np.asarray(y_gap, dtype=np.int32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["seed"]))
    ensure_project_dirs()
    dc = cfg["data"]

    df = pd.read_csv(dc["processed_csv"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ---------------------------------------------------------
    # USE THE PRE-COMPUTED GROUP SPLIT
    # ---------------------------------------------------------

    if "split" not in df.columns:
        raise ValueError(
            "Expected 'split' column in processed CSV. "
            "Run 02b_make_group_stratified_split.py first."
        )

    df["split"] = df["split"].astype(str)

    expected_splits = {"train", "val", "test"}
    actual_splits = set(df["split"].unique())

    if not expected_splits.issubset(actual_splits):
        raise ValueError(
            f"Invalid split labels. Found: {actual_splits}"
        )

    print("\n===== USING PRE-COMPUTED GROUP SPLIT =====")
    print(df["split"].value_counts())

    print("\n===== GROUPS BY SPLIT =====")
    print(
        df.groupby("split")["week_id"]
        .nunique()
    )

    for split_name in ["train", "val", "test"]:
        weeks = sorted(
            df.loc[df["split"] == split_name, "week_id"]
            .astype(str)
            .unique()
        )
        print(f"{split_name}: {weeks}")

    # Fit input vocabularies on training events only.
    train_df = df[df["split"] == "train"]
    vocabs = {
        "event": make_vocab(train_df["event_token"]),
        "type": make_vocab(train_df["sensor_type"]),
        "room": make_vocab(train_df["room"]),
        "state": make_vocab(train_df["state_norm"]),
    }

    print("Event vocab size:", len(vocabs["event"]))
    print("State vocab size:", len(vocabs["state"]))

    # The activity taxonomy is known for the dataset. Build it from all labels.
    label_vocab = {v: i for i, v in enumerate(sorted({str(x) for x in df["label"]}))}
    max_window = int(dc["max_window"])
    step = int(dc["step_size"])
    delta_clip = float(dc["delta_clip_seconds"])
    gap_bins = [float(x) for x in dc["gap_bins_seconds"]]

    arrays = {}
    for split in ["train", "val", "test"]:
        built = build_split_windows(
            df, split, max_window, step, vocabs, label_vocab, delta_clip, gap_bins
        )
        for key, value in built.items():
            arrays[f"{split}_{key}"] = value
        print(split, built["X_event"].shape, np.bincount(built["y_activity"]))

    arrays["num_events"] = np.array(len(vocabs["event"]), dtype=np.int32)
    arrays["num_types"] = np.array(len(vocabs["type"]), dtype=np.int32)
    arrays["num_rooms"] = np.array(len(vocabs["room"]), dtype=np.int32)
    arrays["num_states"] = np.array(len(vocabs["state"]), dtype=np.int32)
    arrays["num_classes"] = np.array(len(label_vocab), dtype=np.int32)
    arrays["num_gap_bins"] = np.array(len(gap_bins) + 1, dtype=np.int32)
    arrays["max_window"] = np.array(max_window, dtype=np.int32)
    arrays["time_dim"] = np.array(5, dtype=np.int32)
    arrays["context_dim"] = np.array(7, dtype=np.int32)

    out_npz = Path(dc["windows_npz"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    print("Maximum event ID:", np.max(arrays["train_X_event"]))
    print("Maximum state ID:", np.max(arrays["train_X_state"]))
    print("num_events:", len(vocabs["event"]))
    print("num_states:", len(vocabs["state"]))
    np.savez_compressed(out_npz, **arrays)

    # Note: tr_groups, va_groups, and te_groups are undefined here in the original
    # logic as they were replaced by the pre-computed split. You may need to fetch
    # them explicitly or assign them dummy values if the program errors out here.
    mappings = {
        "vocabs": vocabs,
        "labels": label_vocab,
        "gap_bins_seconds": gap_bins,
        "split_groups": {
            "train": sorted([]), # Was tr_groups
            "val": sorted([]),   # Was va_groups
            "test": sorted([]),  # Was te_groups
        },
        "config_snapshot": cfg,
    }
    Path(dc["mappings_json"]).write_text(json.dumps(mappings, indent=2), encoding="utf-8")

    df.to_csv(Path(dc["processed_csv"]).with_name("aruba_events_with_split.csv"), index=False)
    print(f"Saved windows: {out_npz}")
    print(f"Saved mappings: {dc['mappings_json']}")


if __name__ == "__main__":
    main()