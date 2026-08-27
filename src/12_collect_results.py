\
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main():
    rows = []
    for path in Path("results").glob("*_metrics.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(data)
    if not rows:
        print("No metrics JSON files found.")
        return
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df.to_csv("results/all_model_metrics.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
