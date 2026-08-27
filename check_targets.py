import pandas as pd

print("--- Loading Enriched Dataset ---")
df = pd.read_csv("data/processed/aruba_events_transition.csv")

print("\n--- Check 1: Number of activity transitions ---")
print("Total events:", len(df))
print("Total transitions:", int(df["boundary"].sum()))
print("Transition percentage:", round(100 * df["boundary"].mean(), 4), "%")

print("\n--- Check 2: Activity phase distribution ---")
print(df["activity_phase"].value_counts())

print("\n--- Check 3: Distance statistics ---")
print(df["distance_since_transition"].describe())
print("\n----------------------------------")
