\
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config_utils import load_config, set_global_seed, ensure_project_dirs


TIMESTAMP_CANDIDATES = ["timestamp", "datetime", "date_time", "time"]
SENSOR_CANDIDATES = ["sensor_id", "sensor", "sensorid", "device", "device_id"]
STATE_CANDIDATES = ["state", "status", "value", "sensor_state"]
ACTIVITY_CANDIDATES = ["activity", "label", "activity_label", "class"]


def first_existing(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def infer_sensor_type(sensor_id: str) -> str:
    s = str(sensor_id).upper()
    if s.startswith("M") or "PIR" in s or "MOTION" in s:
        return "motion"
    if s.startswith("D") or "DOOR" in s:
        return "door"
    if s.startswith("T") or "TEMP" in s:
        return "temperature"
    if s.startswith("L") or "LIGHT" in s:
        return "light"
    if s.startswith("I") or "ITEM" in s:
        return "item"
    return "other"


def load_room_map(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    data.pop("_comment", None)
    return {str(k): str(v) for k, v in data.items()}


def parse_legacy_aruba(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 4:
                continue
            date, time, sensor, state = parts[:4]
            extra = " ".join(parts[4:]).strip()
            activity = ""
            marker = ""
            lower = extra.lower()
            if lower.endswith(" begin"):
                activity = extra[:-6].strip()
                marker = "begin"
            elif lower.endswith(" end"):
                activity = extra[:-4].strip()
                marker = "end"
            elif extra:
                activity = extra
            rows.append({
                "timestamp": f"{date} {time}",
                "sensor_id": sensor,
                "state": state,
                "activity_raw": activity,
                "marker": marker,
            })
    return pd.DataFrame(rows)


def parse_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = first_existing(df.columns, TIMESTAMP_CANDIDATES)
    sensor_col = first_existing(df.columns, SENSOR_CANDIDATES)
    state_col = first_existing(df.columns, STATE_CANDIDATES)
    activity_col = first_existing(df.columns, ACTIVITY_CANDIDATES)

    if ts_col is None and {"date", "time"}.issubset({c.lower() for c in df.columns}):
        date_col = [c for c in df.columns if c.lower() == "date"][0]
        time_col = [c for c in df.columns if c.lower() == "time"][0]
        timestamp = df[date_col].astype(str) + " " + df[time_col].astype(str)
    elif ts_col is not None:
        timestamp = df[ts_col].astype(str)
    else:
        raise ValueError("CSV needs a timestamp column or separate date and time columns.")

    if sensor_col is None or state_col is None:
        raise ValueError("Could not identify sensor and state columns.")

    activity = df[activity_col].fillna("").astype(str) if activity_col else ""
    out = pd.DataFrame({
        "timestamp": timestamp,
        "sensor_id": df[sensor_col].astype(str),
        "state": df[state_col].astype(str),
        "activity_raw": activity,
        "marker": "",
    })

    # Handle activity strings containing begin/end.
    raw_activity = out["activity_raw"].fillna("").astype(str)
    begin_mask = raw_activity.str.lower().str.endswith(" begin")
    end_mask = raw_activity.str.lower().str.endswith(" end")
    out.loc[begin_mask, "marker"] = "begin"
    out.loc[end_mask, "marker"] = "end"
    out.loc[begin_mask, "activity_raw"] = raw_activity[begin_mask].str[:-6].str.strip()
    out.loc[end_mask, "activity_raw"] = raw_activity[end_mask].str[:-4].str.strip()
    return out


def fill_activity_intervals(df: pd.DataFrame, other_label: str) -> pd.Series:
    current = other_label
    labels = []
    for row in df.itertuples(index=False):
        raw = str(row.activity_raw).strip()
        marker = str(row.marker).strip().lower()
        if marker == "begin":
            current = raw if raw else other_label
            labels.append(current)
        elif marker == "end":
            labels.append(current)
            current = other_label
        elif raw and raw.lower() != "nan":
            labels.append(raw)
        else:
            labels.append(current)
    return pd.Series(labels, index=df.index, dtype="object")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg["seed"]))
    ensure_project_dirs()

    data_cfg = cfg["data"]
    raw_path = args.input or data_cfg["raw_path"]
    p = Path(raw_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Input not found: {p}. Place Aruba data there or use --input."
        )

    if p.suffix.lower() == ".csv":
        df = parse_csv(str(p))
    else:
        df = parse_legacy_aruba(str(p))

    if df.empty:
        raise ValueError("No valid rows were parsed.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    other_label = str(data_cfg.get("other_label", "Other"))
    df["label"] = fill_activity_intervals(df, other_label)

    if not bool(data_cfg.get("keep_other", True)):
        df = df[df["label"] != other_label].copy()

    room_map = load_room_map(data_cfg.get("room_map_json"))
    print("\n===== DEBUG ROOM MAP =====")
    print("Room map path:", data_cfg.get("room_map_json"))
    print("Loaded entries:", len(room_map))
    print("M003 ->", room_map.get("M003"))
    print("T002 ->", room_map.get("T002"))
    print("==========================\n")
    df["sensor_type"] = df["sensor_id"].map(infer_sensor_type)
    df["room"] = df["sensor_id"].map(room_map).fillna("UNKNOWN")
    print(df[["sensor_id", "room"]].head(20))
    df["state_norm"] = (
    df["state"]
    .astype(str)
    .str.upper()
    .str.strip())

    # Compress continuous sensor values
    df.loc[df["sensor_type"] == "temperature", "state_norm"] = "TEMP"
    df.loc[df["sensor_type"] == "light", "state_norm"] = "LIGHT"

    df["event_token"] = df["sensor_id"].astype(str) + "|" + df["state_norm"]
    df["semantic_token"] = (
        df["sensor_type"].astype(str) + "|" +
        df["room"].astype(str) + "|" +
        df["state_norm"].astype(str)
    )

    df["delta_t"] = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
    df["delta_t"] = df["delta_t"].clip(lower=0.0)

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date.astype(str)
    iso = df["timestamp"].dt.isocalendar()
    df["week_id"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    out_path = Path(data_cfg["processed_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Events: {len(df):,}")
    print(f"Sensors: {df['sensor_id'].nunique()}")
    print(f"Labels: {df['label'].nunique()}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
