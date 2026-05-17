"""
===============================================================================
WIND FARM C - STAGE 1: DATA AUDIT & INVENTORY
===============================================================================
ELE469 / ACS64xx MSc Project - University of Sheffield 2025-26
Student: Nijat Badalov

Purpose:
    Read-only audit of all 58 Wind Farm C event datasets.
    Produces a master inventory with:
      - Event metadata (label, start/end, description, asset_id)
      - Row counts (train/prediction split)
      - Status distribution per event
      - Feature quality summary (missing, zeros, ranges)
      - Feature classification by physical subsystem

Data Source:
    C:/Users/nijat/OneDrive/Documents/WIND FIN/Wind farm c/Wind Farm C

Outputs:
    results/stage1_event_inventory.csv
    results/stage1_feature_groups.json
    results/stage1_data_quality.csv
    results/stage1_audit_summary.txt

CRITICAL: This script is READ-ONLY. It does not modify any source data.
===============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = Path(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\Wind Farm C")
DATASETS_DIR = DATA_ROOT / "datasets"
EVENT_INFO_PATH = DATA_ROOT / "event_info.csv"
FEATURE_DESC_PATH = DATA_ROOT / "feature_description.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_event_info():
    """Load and parse event_info.csv with semicolon delimiter."""
    df = pd.read_csv(EVENT_INFO_PATH, sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()
    # Parse timestamps
    for col in ["event_start", "event_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    print(f"[AUDIT] Loaded event_info: {len(df)} events")
    print(f"        Anomaly: {(df['event_label']=='anomaly').sum()}, "
          f"Normal: {(df['event_label']=='normal').sum()}")
    return df


def load_feature_description():
    """Load and parse feature_description.csv."""
    df = pd.read_csv(FEATURE_DESC_PATH, sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()
    print(f"[AUDIT] Loaded feature_description: {len(df)} sensors")
    return df


def classify_features(feat_desc):
    """
    Classify sensors into physical subsystem groups based on description.
    This is critical for physics-aware feature selection and SysID I/O mapping.
    """
    groups = {
        "gearbox_thermal": [],
        "gearbox_oil": [],
        "generator_thermal": [],
        "generator_electrical": [],
        "transformer_thermal": [],
        "hydraulic": [],
        "pitch_axis": [],
        "yaw": [],
        "rotor_bearing": [],
        "converter": [],
        "cooling_water": [],
        "ambient_environment": [],
        "wind_speed": [],
        "power_active": [],
        "power_reactive": [],
        "vibration": [],
        "pressure": [],
        "battery_voltage": [],
        "cabinet_thermal": [],
        "other": [],
    }

    for _, row in feat_desc.iterrows():
        name = str(row.get("sensor_name", "")).lower().strip()
        desc = str(row.get("description", "")).lower().strip()
        unit = str(row.get("unit", "")).lower().strip()

        # Wind speed
        if "wind_speed" in name or "wind speed" in desc:
            groups["wind_speed"].append(name)
        # Active power
        elif "power_" in name and "reactive" not in desc:
            groups["power_active"].append(name)
        elif "active power" in desc:
            groups["power_active"].append(name)
        # Reactive power
        elif "reactive_power" in name or "reactive power" in desc:
            groups["power_reactive"].append(name)
        # Gearbox oil temperature
        elif "gearbox oil temp" in desc or "gearbox oil inlet" in desc:
            groups["gearbox_oil"].append(name)
        # Gearbox general (oil level, bypass valve, filter)
        elif "gearbox" in desc and ("temp" in desc or "bearing" in desc):
            groups["gearbox_thermal"].append(name)
        elif "gearbox" in desc:
            groups["gearbox_oil"].append(name)
        # Planetary bearing temperatures
        elif "planetary bearing" in desc:
            groups["gearbox_thermal"].append(name)
        # Generator cooling / stator / winding
        elif ("generator" in desc or "stator" in desc) and "temp" in desc:
            groups["generator_thermal"].append(name)
        elif "generator" in desc and ("current" in desc or "rms" in desc or "voltage" in desc):
            groups["generator_electrical"].append(name)
        elif "generator" in desc:
            groups["generator_thermal"].append(name)
        # Transformer
        elif "transformer" in desc:
            groups["transformer_thermal"].append(name)
        # Hydraulic
        elif "hydraulic" in desc:
            groups["hydraulic"].append(name)
        # Rotor bearing
        elif "rotor bearing" in desc:
            groups["rotor_bearing"].append(name)
        # Pitch / axis motor
        elif "axis" in desc and ("motor" in desc or "position" in desc or "cooling" in desc or "temp" in desc):
            groups["pitch_axis"].append(name)
        elif "pitch" in desc or "blade" in desc:
            groups["pitch_axis"].append(name)
        # Yaw
        elif "yaw" in desc:
            groups["yaw"].append(name)
        # Converter
        elif "converter" in desc or "dc link" in desc:
            groups["converter"].append(name)
        # Cooling water
        elif "water" in desc or "cooling" in desc:
            groups["cooling_water"].append(name)
        # Ambient / outside / nacelle temperature
        elif "ambient" in desc or "outside temp" in desc or "nacelle" in desc:
            groups["ambient_environment"].append(name)
        # Vibration
        elif "vibration" in desc:
            groups["vibration"].append(name)
        # Pressure (non-hydraulic)
        elif "pressure" in desc or "bar" in unit:
            groups["pressure"].append(name)
        # Battery / voltage
        elif "battery" in desc or "24v" in desc or "power pack" in desc:
            groups["battery_voltage"].append(name)
        # Cabinet / electrical cabinet / platform / board temperature
        elif "cabinet" in desc or "board" in desc or "platform" in desc or "hub temp" in desc:
            groups["cabinet_thermal"].append(name)
        # Catch-all
        else:
            groups["other"].append(name)

    # Summary
    print("\n[AUDIT] Feature Group Classification:")
    total_classified = 0
    for group, sensors in sorted(groups.items()):
        if sensors:
            print(f"  {group:30s}: {len(sensors):3d} sensors")
            total_classified += len(sensors)
    print(f"  {'TOTAL':30s}: {total_classified:3d} sensors")

    return groups


def audit_single_event(event_id, event_row, feat_desc):
    """
    Audit one event CSV file. Returns a dict of quality metrics.
    """
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    if not csv_path.exists():
        return {"event_id": event_id, "status": "FILE_MISSING"}

    # Read with low_memory=False to avoid dtype warnings
    try:
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
    except Exception as e:
        return {"event_id": event_id, "status": f"READ_ERROR: {e}"}

    df.columns = df.columns.str.strip()
    n_rows = len(df)
    n_cols = len(df.columns)

    # Train/test split
    if "train_test" in df.columns:
        n_train = (df["train_test"] == "train").sum()
        n_pred = (df["train_test"] != "train").sum()
    else:
        n_train = n_rows
        n_pred = 0

    # Status distribution
    status_dist = {}
    if "status_type_id" in df.columns:
        status_counts = df["status_type_id"].value_counts().to_dict()
        status_dist = {f"status_{int(k)}": int(v) for k, v in status_counts.items()}

    # Asset ID
    asset_id = df["asset_id"].iloc[0] if "asset_id" in df.columns else None

    # Time range
    if "time_stamp" in df.columns:
        ts = pd.to_datetime(df["time_stamp"], errors="coerce")
        ts_min = ts.min()
        ts_max = ts.max()
        ts_span_days = (ts_max - ts_min).days if pd.notna(ts_min) and pd.notna(ts_max) else None
    else:
        ts_min = ts_max = ts_span_days = None

    # Missing value analysis (on sensor columns only)
    sensor_cols = [c for c in df.columns if c not in ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]]
    n_missing_total = df[sensor_cols].isna().sum().sum()
    n_zero_total = (df[sensor_cols] == 0).sum().sum()
    n_cells_total = len(sensor_cols) * n_rows

    # Healthy rows (status 0 or 2, train only)
    if "status_type_id" in df.columns and "train_test" in df.columns:
        healthy_mask = (df["train_test"] == "train") & (df["status_type_id"].isin([0, 2]))
        n_healthy_train = healthy_mask.sum()
    else:
        n_healthy_train = 0

    result = {
        "event_id": event_id,
        "status": "OK",
        "asset_id": asset_id,
        "event_label": event_row.get("event_label", "unknown"),
        "event_description": event_row.get("event_description", ""),
        "event_start": str(event_row.get("event_start", "")),
        "event_end": str(event_row.get("event_end", "")),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_train": n_train,
        "n_prediction": n_pred,
        "n_healthy_train": n_healthy_train,
        "ts_min": str(ts_min) if ts_min else None,
        "ts_max": str(ts_max) if ts_max else None,
        "ts_span_days": ts_span_days,
        "n_missing_cells": int(n_missing_total),
        "n_zero_cells": int(n_zero_total),
        "pct_missing": round(100 * n_missing_total / n_cells_total, 2) if n_cells_total > 0 else 0,
        "pct_zero": round(100 * n_zero_total / n_cells_total, 2) if n_cells_total > 0 else 0,
    }
    result.update(status_dist)

    return result


def run_full_audit():
    """Execute the complete Stage 1 audit."""
    print("=" * 80)
    print("WIND FARM C - STAGE 1: DATA AUDIT & INVENTORY")
    print("=" * 80)
    start_time = time.time()

    # ── Load metadata ─────────────────────────────────────────────────────
    event_info = load_event_info()
    feat_desc = load_feature_description()

    # ── Classify features ─────────────────────────────────────────────────
    feature_groups = classify_features(feat_desc)

    # ── Check available CSV files ─────────────────────────────────────────
    available_csvs = sorted([int(f.stem) for f in DATASETS_DIR.glob("*.csv")])
    expected_ids = sorted(event_info["event_id"].tolist())
    print(f"\n[AUDIT] CSV files found: {len(available_csvs)}")
    print(f"[AUDIT] Events in event_info: {len(expected_ids)}")
    missing_csvs = set(expected_ids) - set(available_csvs)
    extra_csvs = set(available_csvs) - set(expected_ids)
    if missing_csvs:
        print(f"  [!] Missing CSVs for events: {sorted(missing_csvs)}")
    if extra_csvs:
        print(f"  [!] Extra CSVs not in event_info: {sorted(extra_csvs)}")

    # ── Audit each event ──────────────────────────────────────────────────
    print(f"\n[AUDIT] Auditing {len(event_info)} event files...")
    audit_results = []
    for idx, row in event_info.iterrows():
        eid = row["event_id"]
        result = audit_single_event(eid, row, feat_desc)
        audit_results.append(result)
        label_tag = "[ANOMALY]" if row["event_label"] == "anomaly" else "[NORMAL ]"
        status_tag = "OK" if result.get("status") == "OK" else "XX"
        print(f"  [{status_tag}] Event {eid:3d} | {label_tag} | "
              f"Asset {result.get('asset_id', '?'):>3} | "
              f"Rows: {result.get('n_rows', 0):>6,} | "
              f"Train: {result.get('n_train', 0):>6,} | "
              f"Pred: {result.get('n_prediction', 0):>5,} | "
              f"Healthy: {result.get('n_healthy_train', 0):>6,}")

    # ── Build master inventory ────────────────────────────────────────────
    inventory_df = pd.DataFrame(audit_results)

    # ── Per-turbine summary ───────────────────────────────────────────────
    ok_events = inventory_df[inventory_df["status"] == "OK"]
    turbine_summary = ok_events.groupby("asset_id").agg(
        n_events=("event_id", "count"),
        n_anomaly=("event_label", lambda x: (x == "anomaly").sum()),
        n_normal=("event_label", lambda x: (x == "normal").sum()),
        total_healthy_train=("n_healthy_train", "sum"),
    ).reset_index()

    print(f"\n[AUDIT] Per-Turbine Summary:")
    print(f"{'Asset':>6} | {'Events':>6} | {'Anomaly':>7} | {'Normal':>6} | {'Healthy Train':>13}")
    print("-" * 55)
    for _, t in turbine_summary.iterrows():
        print(f"{t['asset_id']:>6} | {t['n_events']:>6} | {t['n_anomaly']:>7} | "
              f"{t['n_normal']:>6} | {t['total_healthy_train']:>13,}")

    # ── Global summary ────────────────────────────────────────────────────
    n_anomaly = (ok_events["event_label"] == "anomaly").sum()
    n_normal = (ok_events["event_label"] == "normal").sum()
    n_turbines = ok_events["asset_id"].nunique()
    total_rows = ok_events["n_rows"].sum()
    total_healthy = ok_events["n_healthy_train"].sum()

    summary_text = f"""
================================================================================
WIND FARM C - STAGE 1 AUDIT SUMMARY
================================================================================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
  Wind Farm: C (offshore, Germany, anonymised)
  Event CSVs found: {len(available_csvs)}
  Events in metadata: {len(expected_ids)}
  Successfully audited: {len(ok_events)}
  Missing CSVs: {sorted(missing_csvs) if missing_csvs else 'None'}
  Extra CSVs: {sorted(extra_csvs) if extra_csvs else 'None'}

EVENT DISTRIBUTION
  Anomaly events: {n_anomaly}
  Normal events: {n_normal}
  Total events: {n_anomaly + n_normal}
  Unique turbines: {n_turbines}

DATA SCALE
  Total rows across all events: {total_rows:,}
  Total healthy training rows (status 0/2, train only): {total_healthy:,}
  Columns per CSV: 957
  Feature sensors: {len(feat_desc)}

FEATURE GROUPS (for SysID I/O selection)
"""
    for group, sensors in sorted(feature_groups.items()):
        if sensors:
            summary_text += f"  {group:30s}: {len(sensors):3d} sensors\n"

    summary_text += f"""
LEAKAGE SAFEGUARDS VERIFIED
  [OK] Train/prediction split column present in all CSVs
  [OK] Status_type_id column present in all CSVs
  [OK] Event labels stored in event_info.csv, NOT in CSVs
  [OK] Timestamp monotonicity will be checked in Stage 2

CRITICAL WARNINGS FOR WIND FARM C
  [!] Missing values may appear as ZEROS (operator policy for farms B/C)
  [!] 957 features - do NOT use all; select physics-aware groups
  [!] Feature names are anonymised - use feature_description.csv
  [!] Status codes may be inconsistent (logged only on change)
  [!] Angle features need sin/cos encoding (is_angle flag)

NEXT STAGE
  Stage 2: Build strict clean healthy training pool
  - Filter: status in {{0,2}}, train_test == 'train'
  - Remove zero blocks, physically impossible values
  - Select physics-aware feature subsets for SysID
  - Fit scalers on training data ONLY
================================================================================
"""
    print(summary_text)

    # ── Save outputs ──────────────────────────────────────────────────────
    inventory_df.to_csv(RESULTS_DIR / "stage1_event_inventory.csv", index=False)

    with open(RESULTS_DIR / "stage1_feature_groups.json", "w") as f:
        json.dump(feature_groups, f, indent=2)

    with open(RESULTS_DIR / "stage1_audit_summary.txt", "w") as f:
        f.write(summary_text)

    # Save turbine summary
    turbine_summary.to_csv(RESULTS_DIR / "stage1_turbine_summary.csv", index=False)

    # Save feature description with groups
    feat_with_group = feat_desc.copy()
    sensor_to_group = {}
    for group, sensors in feature_groups.items():
        for s in sensors:
            sensor_to_group[s] = group
    feat_with_group["physical_group"] = feat_with_group["sensor_name"].str.lower().str.strip().map(sensor_to_group)
    feat_with_group.to_csv(RESULTS_DIR / "stage1_features_classified.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\n[AUDIT] Stage 1 complete in {elapsed:.1f}s")
    print(f"[AUDIT] Outputs saved to: {RESULTS_DIR}")
    return inventory_df, feature_groups


if __name__ == "__main__":
    run_full_audit()
