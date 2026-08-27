import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/aruba_events_transition.csv"

WINDOW_SIZE = 60
STRIDE = 5

OUTPUT_FILE = (
    f"data/processed/"
    f"aruba_tar_windows_w{WINDOW_SIZE}_s{STRIDE}.npz"
)

TIME_DELTA_CLIP = 21600.0


# ============================================================
# TIME FEATURES
# ============================================================

def make_time_features(part):
    hour_float = (
        part["hour"].to_numpy(dtype=np.float32)
        + part["minute"].to_numpy(dtype=np.float32) / 60.0
    )

    dow = part["dayofweek"].to_numpy(dtype=np.float32)

    delta = np.minimum(
        part["delta_t"].to_numpy(dtype=np.float32),
        TIME_DELTA_CLIP
    )

    hour_sin = np.sin(
        2.0 * np.pi * hour_float / 24.0
    )

    hour_cos = np.cos(
        2.0 * np.pi * hour_float / 24.0
    )

    dow_sin = np.sin(
        2.0 * np.pi * dow / 7.0
    )

    dow_cos = np.cos(
        2.0 * np.pi * dow / 7.0
    )

    delta_log = (
        np.log1p(delta)
        / np.log1p(TIME_DELTA_CLIP)
    )

    return np.stack(
        [
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            delta_log,
        ],
        axis=1
    ).astype(np.float32)


# ============================================================
# WINDOW CONTEXT
# ============================================================

def window_context(event_ids, room_ids, delta_seq):

    n = len(event_ids)

    if n > 1:
        event_change = np.mean(
            event_ids[1:] != event_ids[:-1]
        )

        room_change = np.mean(
            room_ids[1:] != room_ids[:-1]
        )
    else:
        event_change = 0.0
        room_change = 0.0

    unique_event_ratio = (
        len(np.unique(event_ids))
        / max(1, n)
    )

    unique_room_ratio = (
        len(np.unique(room_ids))
        / max(1, n)
    )

    return np.array(
        [
            unique_event_ratio,
            unique_room_ratio,
            event_change,
            room_change,
            float(np.mean(delta_seq)),
            float(np.std(delta_seq)),
            float(np.max(delta_seq)),
        ],
        dtype=np.float32
    )


# ============================================================
# GLOBAL VOCABULARY
# ============================================================

def build_global_mapping(values):

    # Convert everything to string first.
    values = values.astype(str)

    unique_values = sorted(
        set(values)
    )

    mapping = {
        value: idx
        for idx, value in enumerate(unique_values)
    }

    return mapping


# ============================================================
# SAFE ENCODING
# ============================================================

def encode_series(series, mapping, column_name):

    values = series.astype(str)

    unknown = sorted(
        set(values) - set(mapping.keys())
    )

    if unknown:
        raise ValueError(
            f"\nUnknown values found in column '{column_name}':\n"
            f"{unknown}\n"
            f"These values are not present in the mapping."
        )

    encoded = values.map(mapping)

    if encoded.isna().any():
        bad_values = values[encoded.isna()].unique()

        raise ValueError(
            f"\nNaN encountered while encoding "
            f"column '{column_name}'.\n"
            f"Problematic values: {bad_values}"
        )

    return encoded.to_numpy(dtype=np.int32)


# ============================================================
# BUILD WINDOWS FOR ONE SPLIT
# ============================================================

def build_split(
    part,
    split_name,
    event_map,
    type_map,
    room_map,
    state_map,
    label_map,
):

    part = (
        part
        .copy()
        .reset_index(drop=True)
    )

    if len(part) < WINDOW_SIZE:
        raise ValueError(
            f"{split_name} has only "
            f"{len(part)} events. "
            f"Need at least {WINDOW_SIZE}."
        )

    # --------------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------------

    events = encode_series(
        part["event_token"],
        event_map,
        "event_token"
    )

    sensor_types = encode_series(
        part["sensor_type"],
        type_map,
        "sensor_type"
    )

    rooms = encode_series(
        part["room"],
        room_map,
        "room"
    )

    states = encode_series(
        part["state_norm"],
        state_map,
        "state_norm"
    )

    labels = encode_series(
        part["label"],
        label_map,
        "label"
    )

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    time_features = make_time_features(part)

    boundaries = part["boundary"].to_numpy(
        dtype=np.int8
    )

    phases = part["activity_phase"].to_numpy(
        dtype=np.int8
    )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    X_event = []
    X_type = []
    X_room = []
    X_state = []
    X_time = []
    X_context = []

    y_activity = []
    y_boundary = []
    y_phase = []

    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for start in range(
        0,
        len(part) - WINDOW_SIZE + 1,
        STRIDE
    ):

        end = start + WINDOW_SIZE

        ev = events[start:end]
        ty = sensor_types[start:end]
        ro = rooms[start:end]
        st = states[start:end]
        tf = time_features[start:end]

        act_seq = labels[start:end]
        boundary_seq = boundaries[start:end]
        phase_seq = phases[start:end]

        X_event.append(ev)
        X_type.append(ty)
        X_room.append(ro)
        X_state.append(st)
        X_time.append(tf)

        X_context.append(
            window_context(
                ev,
                ro,
                tf[:, -1]
            )
        )

        # Target = activity at final event
        y_activity.append(
            act_seq[-1]
        )

        # Auxiliary transition targets
        y_boundary.append(
            boundary_seq
        )

        y_phase.append(
            phase_seq
        )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    result = {

        f"{split_name}_X_event":
            np.asarray(
                X_event,
                dtype=np.int32
            ),

        f"{split_name}_X_type":
            np.asarray(
                X_type,
                dtype=np.int32
            ),

        f"{split_name}_X_room":
            np.asarray(
                X_room,
                dtype=np.int32
            ),

        f"{split_name}_X_state":
            np.asarray(
                X_state,
                dtype=np.int32
            ),

        f"{split_name}_X_time":
            np.asarray(
                X_time,
                dtype=np.float32
            ),

        f"{split_name}_X_context":
            np.asarray(
                X_context,
                dtype=np.float32
            ),

        f"{split_name}_y_activity":
            np.asarray(
                y_activity,
                dtype=np.int32
            ),

        f"{split_name}_y_boundary":
            np.asarray(
                y_boundary,
                dtype=np.int8
            ),

        f"{split_name}_y_phase":
            np.asarray(
                y_phase,
                dtype=np.int8
            ),
    }

    return result


# ============================================================
# VALIDATION
# ============================================================

def check_split_windows(
    result,
    split_name,
    num_events,
    num_types,
    num_rooms,
    num_states,
    num_classes,
):

    print(
        f"\nChecking {split_name} windows..."
    )

    X_event = result[
        f"{split_name}_X_event"
    ]

    X_type = result[
        f"{split_name}_X_type"
    ]

    X_room = result[
        f"{split_name}_X_room"
    ]

    X_state = result[
        f"{split_name}_X_state"
    ]

    X_time = result[
        f"{split_name}_X_time"
    ]

    X_context = result[
        f"{split_name}_X_context"
    ]

    y_activity = result[
        f"{split_name}_y_activity"
    ]

    y_boundary = result[
        f"{split_name}_y_boundary"
    ]

    y_phase = result[
        f"{split_name}_y_phase"
    ]

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    n = len(y_activity)

    assert len(X_event) == n
    assert len(X_type) == n
    assert len(X_room) == n
    assert len(X_state) == n
    assert len(X_time) == n
    assert len(X_context) == n
    assert len(y_boundary) == n
    assert len(y_phase) == n

    assert X_event.shape[1] == WINDOW_SIZE
    assert X_type.shape[1] == WINDOW_SIZE
    assert X_room.shape[1] == WINDOW_SIZE
    assert X_state.shape[1] == WINDOW_SIZE
    assert X_time.shape[1] == WINDOW_SIZE
    assert X_time.shape[2] == 5

    assert X_context.shape[1] == 7

    assert y_boundary.shape[1] == WINDOW_SIZE
    assert y_phase.shape[1] == WINDOW_SIZE

    # --------------------------------------------------------
    # ID range checks
    # --------------------------------------------------------

    assert X_event.min() >= 0
    assert X_event.max() < num_events

    assert X_type.min() >= 0
    assert X_type.max() < num_types

    assert X_room.min() >= 0
    assert X_room.max() < num_rooms

    assert X_state.min() >= 0
    assert X_state.max() < num_states

    # --------------------------------------------------------
    # Target checks
    # --------------------------------------------------------

    assert y_activity.min() >= 0
    assert y_activity.max() < num_classes

    assert np.all(
        np.isin(
            y_boundary,
            [0, 1]
        )
    )

    assert np.all(
        np.isin(
            y_phase,
            [0, 1, 2]
        )
    )

    # --------------------------------------------------------
    # NaN / infinity checks
    # --------------------------------------------------------

    assert np.isfinite(X_time).all()
    assert np.isfinite(X_context).all()

    print(
        f"{split_name}: PASS"
    )

    print(
        f"  windows      : {n:,}"
    )

    print(
        f"  X_event      : {X_event.shape}"
    )

    print(
        f"  X_time       : {X_time.shape}"
    )

    print(
        f"  X_context    : {X_context.shape}"
    )

    print(
        f"  y_activity   : {y_activity.shape}"
    )

    print(
        f"  activity IDs : "
        f"{y_activity.min()} - "
        f"{y_activity.max()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print(
        "\n=================================================="
    )
    print(
        "LOADING TRANSITION DATASET"
    )
    print(
        "=================================================="
    )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "split",
        "week_id",
        "event_token",
        "sensor_type",
        "room",
        "state_norm",
        "delta_t",
        "hour",
        "minute",
        "dayofweek",
        "label",
        "boundary",
        "activity_phase",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns:\n{missing}"
        )

    # --------------------------------------------------------
    # Basic dataset checks
    # --------------------------------------------------------

    print(
        "\n===== SPLIT DISTRIBUTION ====="
    )

    print(
        df["split"].value_counts()
    )

    print(
        "\n===== WEEK DISTRIBUTION ====="
    )

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        weeks = sorted(
            df.loc[
                df["split"] == split_name,
                "week_id"
            ]
            .astype(str)
            .unique()
        )

        print(
            f"{split_name}: "
            f"{len(weeks)} weeks"
        )

        print(
            weeks
        )

    # --------------------------------------------------------
    # Check class coverage before encoding
    # --------------------------------------------------------

    print(
        "\n===== CLASS COVERAGE ====="
    )

    all_labels = sorted(
        df["label"]
        .astype(str)
        .unique()
    )

    print(
        f"Total classes: "
        f"{len(all_labels)}"
    )

    print(
        all_labels
    )

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        labels = sorted(
            df.loc[
                df["split"] == split_name,
                "label"
            ]
            .astype(str)
            .unique()
        )

        missing_labels = sorted(
            set(all_labels) - set(labels)
        )

        print(
            f"\n{split_name}: "
            f"{len(labels)}/{len(all_labels)} classes"
        )

        if missing_labels:
            print(
                "MISSING:",
                missing_labels
            )
        else:
            print(
                "All classes present."
            )

    # --------------------------------------------------------
    # Build GLOBAL mappings
    #
    # Important:
    # We use the same mapping for train/val/test.
    # --------------------------------------------------------

    event_map = build_global_mapping(
        df["event_token"]
    )

    type_map = build_global_mapping(
        df["sensor_type"]
    )

    room_map = build_global_mapping(
        df["room"]
    )

    state_map = build_global_mapping(
        df["state_norm"]
    )

    label_map = build_global_mapping(
        df["label"]
    )

    num_events = len(event_map)
    num_types = len(type_map)
    num_rooms = len(room_map)
    num_states = len(state_map)
    num_classes = len(label_map)

    # --------------------------------------------------------
    # Print mappings
    # --------------------------------------------------------

    print(
        "\n===== VOCABULARY SIZES ====="
    )

    print(
        "events :",
        num_events
    )

    print(
        "types  :",
        num_types
    )

    print(
        "rooms  :",
        num_rooms
    )

    print(
        "states :",
        num_states
    )

    print(
        "classes:",
        num_classes
    )

    print(
        "\n===== LABEL MAPPING ====="
    )

    for label, idx in label_map.items():
        print(
            f"{idx:2d} -> {label}"
        )

    # --------------------------------------------------------
    # Build each split
    # --------------------------------------------------------

    all_arrays = {}

    metadata = {}

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        print(
            "\n=================================================="
        )

        print(
            f"BUILDING {split_name.upper()} WINDOWS"
        )

        print(
            "=================================================="
        )

        part = df[
            df["split"] == split_name
        ]

        built = build_split(
            part,
            split_name,
            event_map,
            type_map,
            room_map,
            state_map,
            label_map,
        )

        # ----------------------------------------------------
        # Validate immediately
        # ----------------------------------------------------

        check_split_windows(
            built,
            split_name,
            num_events,
            num_types,
            num_rooms,
            num_states,
            num_classes,
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_arrays.update(
            built
        )

        y = built[
            f"{split_name}_y_activity"
        ]

        class_counts = np.bincount(
            y,
            minlength=num_classes
        )

        metadata[split_name] = {
            "windows": len(y),
            "class_counts": class_counts.tolist(),
        }

        print(
            "\nClass counts:"
        )

        print(
            class_counts
        )

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    all_arrays.update(
        {
            "num_events":
                np.array(
                    num_events,
                    dtype=np.int32
                ),

            "num_types":
                np.array(
                    num_types,
                    dtype=np.int32
                ),

            "num_rooms":
                np.array(
                    num_rooms,
                    dtype=np.int32
                ),

            "num_states":
                np.array(
                    num_states,
                    dtype=np.int32
                ),

            "num_classes":
                np.array(
                    num_classes,
                    dtype=np.int32
                ),

            "max_window":
                np.array(
                    WINDOW_SIZE,
                    dtype=np.int32
                ),

            "stride":
                np.array(
                    STRIDE,
                    dtype=np.int32
                ),

            "time_dim":
                np.array(
                    5,
                    dtype=np.int32
                ),

            "context_dim":
                np.array(
                    7,
                    dtype=np.int32
                ),
        }
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FILE,
        **all_arrays
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n=================================================="
    )

    print(
        "SUCCESS"
    )

    print(
        "=================================================="
    )

    print(
        f"Saved:\n{OUTPUT_FILE}"
    )

    print(
        "\nDataset summary:"
    )

    total_windows = 0

    for split_name, info in metadata.items():

        print(
            f"{split_name:5s}: "
            f"{info['windows']:,} windows"
        )

        total_windows += info[
            "windows"
        ]

    print(
        f"\nTOTAL: "
        f"{total_windows:,} windows"
    )

    print(
        "\nAll validation checks passed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()