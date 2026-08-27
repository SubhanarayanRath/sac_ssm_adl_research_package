import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/aruba_events_transition.csv"
OUTPUT_FILE = "data/processed/aruba_tar_windows.npz"
WINDOW_SIZE = 60
STRIDE = 5

def make_time_features(part):
    hour_float = part["hour"].to_numpy(dtype=np.float32) + part["minute"].to_numpy(dtype=np.float32) / 60.0
    dow = part["dayofweek"].to_numpy(dtype=np.float32)
    delta = np.minimum(part["delta_t"].to_numpy(dtype=np.float32), 21600.0)
    hour_sin, hour_cos = np.sin(2.0 * np.pi * hour_float / 24.0), np.cos(2.0 * np.pi * hour_float / 24.0)
    dow_sin, dow_cos = np.sin(2.0 * np.pi * dow / 7.0), np.cos(2.0 * np.pi * dow / 7.0)
    delta_log = np.log1p(delta) / np.log1p(21600.0)
    return np.stack([hour_sin, hour_cos, dow_sin, dow_cos, delta_log], axis=1).astype(np.float32)

def window_context(event_ids, room_ids, delta_seq):
    n = len(event_ids)
    event_change = np.mean(event_ids[1:] != event_ids[:-1]) if n > 1 else 0.0
    room_change = np.mean(room_ids[1:] != room_ids[:-1]) if n > 1 else 0.0
    unique_event_ratio, unique_room_ratio = len(np.unique(event_ids)) / max(1, n), len(np.unique(room_ids)) / max(1, n)
    return np.array([unique_event_ratio, unique_room_ratio, event_change, room_change, float(np.mean(delta_seq)), float(np.std(delta_seq)), float(np.max(delta_seq))], dtype=np.float32)

def encode_column(values):
    unique = sorted({str(v) for v in values})
    mapping = {value: idx for idx, value in enumerate(unique)}
    return np.array([mapping[str(v)] for v in values], dtype=np.int32), len(mapping)

def build_global_mapping(values):
    return {v: i for i, v in enumerate(sorted(set(map(str, values))))}

def build_split(part, split_name, event_map, type_map, room_map, state_map, label_map):
    part = part.copy().reset_index(drop=True)
    if len(part) < WINDOW_SIZE: raise ValueError(f"{split_name} has only {len(part)} events.")
    events = part["event_token"].map(event_map).to_numpy(np.int32)
    sensor_types = part["sensor_type"].map(type_map).to_numpy(np.int32)
    rooms = part["room"].map(room_map).to_numpy(np.int32)
    states = part["state_norm"].map(state_map).to_numpy(np.int32)
    labels = part["label"].map(label_map).to_numpy(np.int32)
    time_features = make_time_features(part)
    boundaries = part["boundary"].to_numpy(dtype=np.int8)
    phases = part["activity_phase"].to_numpy(dtype=np.int8)
    X_event, X_type, X_room, X_state, X_time, X_context, y_activity, y_boundary, y_phase = [], [], [], [], [], [], [], [], []

    for start in range(0, len(part) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        ev, ty, ro, st, tf = events[start:end], sensor_types[start:end], rooms[start:end], states[start:end], time_features[start:end]
        act_seq, boundary_seq, phase_seq = labels[start:end], boundaries[start:end], phases[start:end]
        X_event.append(ev); X_type.append(ty); X_room.append(ro); X_state.append(st); X_time.append(tf)
        X_context.append(window_context(ev, ro, tf[:, -1]))
        y_activity.append(act_seq[-1]); y_boundary.append(boundary_seq); y_phase.append(phase_seq)

    result = {f"{split_name}_X_event": np.asarray(X_event, dtype=np.int32), f"{split_name}_X_type": np.asarray(X_type, dtype=np.int32), f"{split_name}_X_room": np.asarray(X_room, dtype=np.int32), f"{split_name}_X_state": np.asarray(X_state, dtype=np.int32), f"{split_name}_X_time": np.asarray(X_time, dtype=np.float32), f"{split_name}_X_context": np.asarray(X_context, dtype=np.float32), f"{split_name}_y_activity": np.asarray(y_activity, dtype=np.int32), f"{split_name}_y_boundary": np.asarray(y_boundary, dtype=np.int8), f"{split_name}_y_phase": np.asarray(y_phase, dtype=np.int8)}
    return result, len(event_map), len(type_map), len(room_map), len(state_map), len(label_map)

def main():
    if not os.path.exists(INPUT_FILE): raise FileNotFoundError(INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)
    event_map, type_map, room_map = build_global_mapping(df["event_token"]), build_global_mapping(df["sensor_type"]), build_global_mapping(df["room"])
    state_map, label_map = build_global_mapping(df["state_norm"]), build_global_mapping(df["label"])
    required = ["split", "event_token", "sensor_type", "room", "state_norm", "delta_t", "hour", "minute", "dayofweek", "label", "boundary", "activity_phase"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

    all_arrays, metadata = {}, {}
    for split_name in ["train", "val", "test"]:
        part = df[df["split"] == split_name]
        print(f"\nBuilding {split_name} windows...")
        built, n_events, n_types, n_rooms, n_states, n_classes = build_split(part, split_name, event_map, type_map, room_map, state_map, label_map)
        all_arrays.update(built)
        metadata[split_name] = {"windows": len(built[f"{split_name}_X_event"]), "class_counts": np.bincount(built[f"{split_name}_y_activity"], minlength=n_classes).tolist()}
        print(f"{split_name}: {metadata[split_name]['windows']:,} windows\nclass counts: {metadata[split_name]['class_counts']}")

    all_arrays.update({"num_events": np.array(n_events, dtype=np.int32), "num_types": np.array(n_types, dtype=np.int32), "num_rooms": np.array(n_rooms, dtype=np.int32), "num_states": np.array(n_states, dtype=np.int32), "num_classes": np.array(n_classes, dtype=np.int32), "max_window": np.array(WINDOW_SIZE, dtype=np.int32), "time_dim": np.array(5, dtype=np.int32), "context_dim": np.array(7, dtype=np.int32)})
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, **all_arrays)
    print(f"\nSUCCESS: Saved clean TAR dataset:\n{OUTPUT_FILE}\n\nDataset summary:")
    for split_name, info in metadata.items(): print(f"{split_name}: {info['windows']:,} windows")

if __name__ == "__main__":
    main()