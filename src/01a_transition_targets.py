import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/aruba_events_with_group_split.csv"
OUTPUT_FILE = "data/processed/aruba_events_transition.csv"


def generate_targets():

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Cannot find {INPUT_FILE}.")
        return

    print(f"Loading dataset: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print("\n===== BEFORE =====")
    print(df["room"].value_counts())
    print(df[["sensor_id", "room"]].drop_duplicates().head(10))

    if "split" not in df.columns:
        raise ValueError(
            "Expected a 'split' column in aruba_events_with_split.csv"
        )

    if "label" in df.columns:
        act_col = "label"
    elif "Activity" in df.columns:
        act_col = "Activity"
    else:
        raise ValueError("Could not find activity label column.")

    print(f"Using activity column: {act_col}")

    # --------------------------------------------------
    # Calculate transition targets independently per week
    # --------------------------------------------------

    df["boundary"] = 0

    for (split_name, week_name), idx in df.groupby(
        ["split", "week_id"],
        sort=False
    ).groups.items():

        week_idx = list(idx)

        labels = df.loc[week_idx, act_col]

        boundary = labels.ne(labels.shift(1)).astype(np.int8)

        # First event of every week starts a new sequence
        boundary.iloc[0] = 1

        df.loc[week_idx, "boundary"] = boundary.to_numpy()


    # --------------------------------------------------
    # Segment ID must be independent per split AND week
    # --------------------------------------------------

    df["segment_id"] = (
        df.groupby(["split", "week_id"])["boundary"]
        .cumsum()
        .astype(np.int32)
    )


    # --------------------------------------------------
    # Distance from beginning of current activity
    # --------------------------------------------------

    df["distance_since_transition"] = (
        df.groupby(
            ["split", "week_id", "segment_id"]
        )
        .cumcount()
        .astype(np.int32)
    )


    # --------------------------------------------------
    # Activity phase
    # 0 = beginning
    # 1 = middle
    # 2 = ending
    # --------------------------------------------------

    segment_lengths = (
        df.groupby(
            ["split", "week_id", "segment_id"]
        )["segment_id"]
        .transform("count")
    )

    norm_dist = np.where(
        segment_lengths > 1,
        df["distance_since_transition"] /
        (segment_lengths - 1),
        0.0
    )

    conditions = [
        norm_dist <= 0.33,
        (norm_dist > 0.33) & (norm_dist <= 0.66),
        norm_dist > 0.66,
    ]

    df["activity_phase"] = np.select(
        conditions,
        [0, 1, 2],
        default=0
    ).astype(np.int8)

    # --------------------------------------------------
    # Transition density, independently inside each split
    # --------------------------------------------------

    df["window_transition_count"] = (
        df.groupby(["split", "week_id"])["boundary"]
        .transform(
            lambda s: s.rolling(
                window=30,
                min_periods=1
            ).sum()
        )
        .astype(np.int16)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    print("\n===== AFTER =====")
    print(df["room"].value_counts())
    print(df[["sensor_id", "room"]].drop_duplicates().head(10))

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSUCCESS: Saved to {OUTPUT_FILE}")

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nBoundary distribution by split:")
    print(
        df.groupby("split")["boundary"]
        .value_counts()
    )

    print("\nSample:")
    print(
        df[
            [
                "split",
                act_col,
                "boundary",
                "segment_id",
                "distance_since_transition",
                "activity_phase",
                "window_transition_count",
            ]
        ].head(15)
    )


if __name__ == "__main__":
    generate_targets()