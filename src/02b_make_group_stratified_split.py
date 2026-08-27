import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/aruba_events.csv"
OUTPUT_FILE = "data/processed/aruba_events_with_group_split.csv"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

LABEL_COL = "label"
GROUP_COL = "week_id"


def main():

    # =========================================================
    # 1. LOAD DATA
    # =========================================================

    print("=" * 70)
    print("LOADING ARUBA DATASET")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    required = [LABEL_COL, GROUP_COL]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df[GROUP_COL] = df[GROUP_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str)

    # Preserve original chronological order.
    df["_original_order"] = np.arange(len(df))


    # =========================================================
    # 2. COLLECT WEEK INFORMATION
    # =========================================================

    weeks = []

    for week, part in df.groupby(
        GROUP_COL,
        sort=False
    ):

        counts = part[LABEL_COL].value_counts()

        weeks.append(
            {
                "week": week,
                "rows": len(part),
                "labels": set(counts.index),
                "counts": counts.to_dict(),
            }
        )

    n_weeks = len(weeks)

    if n_weeks < 3:
        raise ValueError(
            "Need at least three weeks."
        )

    all_labels = set(
        df[LABEL_COL].unique()
    )

    labels_sorted = sorted(all_labels)

    print(f"\nTotal weeks: {n_weeks}")
    print(f"Total classes: {len(all_labels)}")
    print("Classes:")
    print(labels_sorted)


    # =========================================================
    # 3. DETERMINE NUMBER OF WEEKS PER SPLIT
    # =========================================================

    n_train = max(
        1,
        int(n_weeks * TRAIN_RATIO)
    )

    n_val = max(
        1,
        int(n_weeks * VAL_RATIO)
    )

    if n_train + n_val >= n_weeks:

        n_val = 1
        n_train = n_weeks - 2

    n_test = (
        n_weeks
        - n_train
        - n_val
    )

    print("\n" + "=" * 70)
    print("TARGET WEEK COUNTS")
    print("=" * 70)

    print(f"Train weeks: {n_train}")
    print(f"Val weeks  : {n_val}")
    print(f"Test weeks : {n_test}")


    # =========================================================
    # 4. INITIAL CHRONOLOGICAL SPLIT
    #
    # We initially keep the chronological ordering.
    # =========================================================

    week_names = [
        item["week"]
        for item in weeks
    ]

    split_weeks = {

        "train": week_names[
            :n_train
        ],

        "val": week_names[
            n_train:n_train + n_val
        ],

        "test": week_names[
            n_train + n_val:
        ],
    }


    # =========================================================
    # 5. HELPER FUNCTIONS
    # =========================================================

    def labels_for_week(week):

        part = df[
            df[GROUP_COL] == week
        ]

        return set(
            part[LABEL_COL].unique()
        )


    def labels_for_split(split_name):

        result = set()

        for week in split_weeks[
            split_name
        ]:

            result.update(
                labels_for_week(week)
            )

        return result


    def missing_labels(split_name):

        return (
            all_labels
            - labels_for_split(split_name)
        )


    def week_size(week):

        return len(
            df[
                df[GROUP_COL] == week
            ]
        )


    # =========================================================
    # 6. PRINT INITIAL SPLIT
    # =========================================================

    print("\n" + "=" * 70)
    print("INITIAL CHRONOLOGICAL SPLIT")
    print("=" * 70)

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        print(
            f"\n{split_name.upper()} "
            f"({len(split_weeks[split_name])} weeks)"
        )

        print(
            split_weeks[split_name]
        )


    # =========================================================
    # 7. REPAIR MISSING CLASSES
    #
    # The important case here is validation missing
    # Respirate.
    #
    # We exchange ENTIRE WEEKS.
    # We never split individual events from a week.
    # =========================================================

    print("\n" + "=" * 70)
    print("CLASS COVERAGE BEFORE REPAIR")
    print("=" * 70)

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        present = labels_for_split(
            split_name
        )

        missing = sorted(
            all_labels - present
        )

        print(
            f"{split_name}: "
            f"{len(present)}/{len(all_labels)} classes"
        )

        if missing:
            print(
                "  MISSING:",
                missing
            )


    # ---------------------------------------------------------
    # Try to repair validation first.
    # ---------------------------------------------------------

    for target_label in sorted(
        missing_labels("val")
    ):

        candidate_source = None
        candidate_source_week = None
        candidate_val_week = None

        # Look for a train/test week containing
        # the missing class.

        for source_split in [
            "train",
            "test",
        ]:

            for source_week in split_weeks[
                source_split
            ]:

                source_labels = labels_for_week(
                    source_week
                )

                if target_label not in source_labels:
                    continue

                # Try replacing one validation week.
                for val_week in split_weeks[
                    "val"
                ]:

                    # Do not remove a class from validation
                    # if that week is its ONLY occurrence
                    # inside validation.

                    val_week_labels = labels_for_week(
                        val_week
                    )

                    remaining_val_labels = set()

                    for other_week in split_weeks[
                        "val"
                    ]:

                        if other_week == val_week:
                            continue

                        remaining_val_labels.update(
                            labels_for_week(
                                other_week
                            )
                        )

                    safe = True

                    for label in val_week_labels:

                        if label not in remaining_val_labels:

                            safe = False
                            break

                    if not safe:
                        continue

                    candidate_source = source_split
                    candidate_source_week = source_week
                    candidate_val_week = val_week

                    break

                if candidate_source is not None:
                    break

            if candidate_source is not None:
                break


        # -----------------------------------------------------
        # Perform exchange.
        # -----------------------------------------------------

        if candidate_source is not None:

            print(
                "\nValidation repair:"
            )

            print(
                f"  Moving "
                f"{candidate_source_week} "
                f"from {candidate_source} -> val"
            )

            print(
                f"  Moving "
                f"{candidate_val_week} "
                f"from val -> {candidate_source}"
            )

            split_weeks[
                candidate_source
            ].remove(
                candidate_source_week
            )

            split_weeks[
                candidate_source
            ].append(
                candidate_val_week
            )

            split_weeks[
                "val"
            ].remove(
                candidate_val_week
            )

            split_weeks[
                "val"
            ].append(
                candidate_source_week
            )

        else:

            print(
                "\nWARNING:"
            )

            print(
                f"Could not repair validation "
                f"for class: {target_label}"
            )


    # =========================================================
    # 8. REPAIR TEST IF NECESSARY
    # =========================================================

    for target_label in sorted(
        missing_labels("test")
    ):

        candidate_source = None
        candidate_source_week = None
        candidate_test_week = None

        for source_split in [
            "train",
            "val",
        ]:

            for source_week in split_weeks[
                source_split
            ]:

                if target_label not in labels_for_week(
                    source_week
                ):
                    continue

                for test_week in split_weeks[
                    "test"
                ]:

                    test_week_labels = labels_for_week(
                        test_week
                    )

                    remaining_test_labels = set()

                    for other_week in split_weeks[
                        "test"
                    ]:

                        if other_week == test_week:
                            continue

                        remaining_test_labels.update(
                            labels_for_week(
                                other_week
                            )
                        )

                    safe = True

                    for label in test_week_labels:

                        if label not in remaining_test_labels:

                            safe = False
                            break

                    if not safe:
                        continue

                    candidate_source = source_split
                    candidate_source_week = source_week
                    candidate_test_week = test_week

                    break

                if candidate_source is not None:
                    break

            if candidate_source is not None:
                break


        if candidate_source is not None:

            print(
                "\nTest repair:"
            )

            print(
                f"  Moving "
                f"{candidate_source_week} "
                f"from {candidate_source} -> test"
            )

            print(
                f"  Moving "
                f"{candidate_test_week} "
                f"from test -> {candidate_source}"
            )

            split_weeks[
                candidate_source
            ].remove(
                candidate_source_week
            )

            split_weeks[
                candidate_source
            ].append(
                candidate_test_week
            )

            split_weeks[
                "test"
            ].remove(
                candidate_test_week
            )

            split_weeks[
                "test"
            ].append(
                candidate_source_week
            )

        else:

            print(
                "\nWARNING:"
            )

            print(
                f"Could not repair test "
                f"for class: {target_label}"
            )


    # =========================================================
    # 9. SORT WEEK LISTS CHRONOLOGICALLY
    # =========================================================

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        split_weeks[
            split_name
        ] = sorted(
            split_weeks[
                split_name
            ]
        )


    # =========================================================
    # 10. FINAL WEEK ALLOCATION
    # =========================================================

    print("\n" + "=" * 70)
    print("FINAL WEEK ALLOCATION")
    print("=" * 70)

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        print(
            f"\n{split_name.upper()} "
            f"({len(split_weeks[split_name])} weeks)"
        )

        print(
            split_weeks[split_name]
        )


    # =========================================================
    # 11. VERIFY NO WEEK OVERLAP
    # =========================================================

    train_set = set(
        split_weeks["train"]
    )

    val_set = set(
        split_weeks["val"]
    )

    test_set = set(
        split_weeks["test"]
    )

    assert train_set.isdisjoint(
        val_set
    )

    assert train_set.isdisjoint(
        test_set
    )

    assert val_set.isdisjoint(
        test_set
    )

    assert (
        len(
            train_set
            | val_set
            | test_set
        )
        == n_weeks
    )

    print("\nWeek leakage check: PASS")


    # =========================================================
    # 12. ASSIGN SPLIT TO EVERY EVENT
    # =========================================================

    def assign_split(week):

        if week in train_set:
            return "train"

        if week in val_set:
            return "val"

        if week in test_set:
            return "test"

        raise RuntimeError(
            f"Unassigned week: {week}"
        )


    df["split"] = df[
        GROUP_COL
    ].map(
        assign_split
    )


    # =========================================================
    # 13. FINAL CLASS COVERAGE
    # =========================================================

    print("\n" + "=" * 70)
    print("FINAL CLASS COVERAGE")
    print("=" * 70)

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        present = labels_for_split(
            split_name
        )

        missing = sorted(
            all_labels - present
        )

        print(
            f"{split_name}: "
            f"{len(present)}/{len(all_labels)} classes"
        )

        if missing:

            print(
                "  MISSING:",
                missing
            )

        else:

            print(
                "  All classes present."
            )


    # =========================================================
    # 14. EVENT COUNTS
    # =========================================================

    print("\n" + "=" * 70)
    print("EVENT COUNTS BY SPLIT")
    print("=" * 70)

    print(
        df["split"].value_counts()
    )


    # =========================================================
    # 15. ACTIVITY COUNTS
    # =========================================================

    print("\n" + "=" * 70)
    print("ACTIVITY COUNTS BY SPLIT")
    print("=" * 70)

    counts = pd.crosstab(
        df["split"],
        df[LABEL_COL]
    )

    print(
        counts.to_string()
    )


    # =========================================================
    # 16. ACTIVITY PERCENTAGES
    # =========================================================

    print("\n" + "=" * 70)
    print("ACTIVITY PERCENTAGES BY SPLIT")
    print("=" * 70)

    percentages = (
        pd.crosstab(
            df["split"],
            df[LABEL_COL],
            normalize="index"
        )
        * 100
    )

    print(
        percentages.round(2).to_string()
    )


    # =========================================================
    # 17. SAVE
    # =========================================================

    df = df.sort_values(
        "_original_order"
    )

    df = df.drop(
        columns=["_original_order"]
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        f"Saved:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()