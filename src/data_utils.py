\
from __future__ import annotations

import numpy as np


INPUT_KEYS = ["X_event", "X_type", "X_room", "X_state", "X_time", "X_context"]


def load_split(npz_path: str, split: str):
    data = np.load(npz_path)
    x = {
        "event_input": data[f"{split}_X_event"],
        "type_input": data[f"{split}_X_type"],
        "room_input": data[f"{split}_X_room"],
        "state_input": data[f"{split}_X_state"],
        "time_input": data[f"{split}_X_time"],
        "context_input": data[f"{split}_X_context"],
    }
    y = {
        "activity": data[f"{split}_y_activity"],
        "next_event": data[f"{split}_y_next_event"],
        "gap_bin": data[f"{split}_y_gap"],
    }
    meta = {
        "num_events": int(data["num_events"]),
        "num_types": int(data["num_types"]),
        "num_rooms": int(data["num_rooms"]),
        "num_states": int(data["num_states"]),
        "num_classes": int(data["num_classes"]),
        "num_gap_bins": int(data["num_gap_bins"]),
        "max_window": int(data["max_window"]),
        "time_dim": int(data["time_dim"]),
        "context_dim": int(data["context_dim"]),
    }
    return x, y, meta
