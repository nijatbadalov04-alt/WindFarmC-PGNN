"""
Stage 2: Build strict clean healthy training pool for Wind Farm C.
Selects physics-aware features, filters by status, removes zero blocks,
fits scalers on training data ONLY, and saves cleaned data.
"""
import pandas as pd
import numpy as np
import json
import time
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = Path(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\Wind Farm C")
DATASETS_DIR = DATA_ROOT / "datasets"
EVENT_INFO_PATH = DATA_ROOT / "event_info.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── PHYSICS-AWARE FEATURE SELECTION ──────────────────────────────────────
# Based on Stage 1 feature_description audit. Selected for SysID I/O mapping.
# Strategy: thermal outputs + exogenous inputs + key operational signals.

THERMAL_OUTPUTS = {
    "gearbox": [
        "sensor_186_avg",  # Gearbox oil temperature 1
        "sensor_187_avg",  # Gearbox oil temperature 2
        "sensor_189_avg",  # Gearbox oil inlet temperature 1
        "sensor_190_avg",  # Gearbox oil inlet temperature 2
    ],
    "generator": [
        "sensor_199_avg",  # Stator winding U1
        "sensor_200_avg",  # Stator winding U2
        "sensor_201_avg",  # Stator winding V1
        "sensor_173_avg",  # Generator cooling air inlet 1
        "sensor_228_avg",  # Cooling water temp generator inlet 1
        "sensor_233_avg",  # Cooling water temp generator outlet 1
    ],
    "transformer": [
        "sensor_191_avg",  # Oil temperature 1 main transformer
        "sensor_192_avg",  # Oil temperature 2 main transformer
        "sensor_188_avg",  # Oil temperature EB transformer
    ],
    "hydraulic": [
        "sensor_178_avg",  # Hydraulic oil tank temperature 1
        "sensor_179_avg",  # Hydraulic oil tank temperature 2
    ],
    "rotor_bearing": [
        "sensor_196_avg",  # Rotor bearing temperature 1
        "sensor_197_avg",  # Rotor bearing temperature 2
        "sensor_198_avg",  # Rotor bearing temperature 3
    ],
}

EXOGENOUS_INPUTS = [
    "sensor_7_avg",          # Ambient temperature
    "sensor_41_avg",         # Outside temperature
    "wind_speed_236_avg",    # Wind speed 1+2 (combined)
    "power_6_avg",           # Active power HV grid
    "sensor_144_avg",        # Rotor speed 1
    "sensor_8_avg",          # Generator angle speed
]

# All selected feature columns (flat list)
ALL_THERMAL = []
for subsys, cols in THERMAL_OUTPUTS.items():
    ALL_THERMAL.extend(cols)
ALL_SELECTED = ALL_THERMAL + EXOGENOUS_INPUTS

# Context columns (metadata, not features)
META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]


def load_event_info():
    df = pd.read_csv(EVENT_INFO_PATH, sep=";")
    df.columns = df.columns.str.strip()
    for col in ["event_start", "event_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_single_event(event_id, event_row, scaler=None, fit_scaler=False):
    """
    Load one event CSV, apply strict cleaning, return cleaned DataFrames.
    Returns (train_clean, pred_clean, n_removed_dict)
    """
    csv_path = DATASETS_DIR / f"{event_id}.csv"
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    df.columns = df.columns.str.strip()

    # Check all selected columns exist
    missing_cols = [c for c in ALL_SELECTED if c not in df.columns]
    if missing_cols:
        print(f"  [WARN] Event {event_id}: missing columns {missing_cols[:5]}...")
        # Use available columns only
        available = [c for c in ALL_SELECTED if c in df.columns]
    else:
        available = ALL_SELECTED.copy()

    # Parse timestamps
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df = df.sort_values("time_stamp").reset_index(drop=True)

    # Split train/prediction
    train_mask = df["train_test"] == "train"
    pred_mask = ~train_mask

    n_total = len(df)
    n_train_raw = train_mask.sum()
    n_pred_raw = pred_mask.sum()

    # ── TRAINING DATA CLEANING ───────────────────────────────────────────
    train_df = df[train_mask].copy()
    removal_log = {"event_id": event_id, "n_train_raw": int(n_train_raw)}

    # 1. Status filter: keep only normal operation (0) and idling (2)
    status_mask = train_df["status_type_id"].isin([0, 2])
    n_status_removed = (~status_mask).sum()
    train_df = train_df[status_mask]
    removal_log["n_status_removed"] = int(n_status_removed)

    # 2. Remove rows where key signals are all-zero (likely missing data)
    key_signals = ["power_6_avg", "wind_speed_236_avg", "sensor_144_avg"]
    key_available = [c for c in key_signals if c in df.columns]
    if key_available:
        zero_mask = (train_df[key_available] == 0).all(axis=1)
        n_zero_removed = zero_mask.sum()
        train_df = train_df[~zero_mask]
    else:
        n_zero_removed = 0
    removal_log["n_zero_block_removed"] = int(n_zero_removed)

    # 3. Physical plausibility checks
    plaus_mask = pd.Series(True, index=train_df.index)
    if "wind_speed_236_avg" in train_df.columns:
        plaus_mask &= (train_df["wind_speed_236_avg"] >= 0) & (train_df["wind_speed_236_avg"] <= 40)
    if "sensor_7_avg" in train_df.columns:
        plaus_mask &= (train_df["sensor_7_avg"] >= -30) & (train_df["sensor_7_avg"] <= 50)
    # Temperatures should be in reasonable range
    for tc in [c for c in available if "sensor_1" in c or "sensor_2" in c]:
        if tc in train_df.columns:
            plaus_mask &= (train_df[tc] >= -50) & (train_df[tc] <= 250)

    n_plaus_removed = (~plaus_mask).sum()
    train_df = train_df[plaus_mask]
    removal_log["n_plausibility_removed"] = int(n_plaus_removed)

    # 4. Remove NaN rows in selected features
    n_before_na = len(train_df)
    train_df = train_df.dropna(subset=[c for c in available if c in train_df.columns])
    removal_log["n_nan_removed"] = int(n_before_na - len(train_df))

    removal_log["n_train_clean"] = int(len(train_df))

    # ── PREDICTION DATA (no cleaning, preserve as-is) ────────────────────
    pred_df = df[pred_mask].copy()

    # ── Extract feature matrices ─────────────────────────────────────────
    feat_cols = [c for c in available if c in df.columns]
    train_features = train_df[META_COLS + feat_cols].copy()
    pred_features = pred_df[META_COLS + feat_cols].copy()

    return train_features, pred_features, removal_log, feat_cols


def build_training_pool():
    """Build the global clean healthy training pool from normal events."""
    print("=" * 80)
    print("STAGE 2: BUILD CLEAN HEALTHY TRAINING POOL")
    print("=" * 80)
    start_time = time.time()

    event_info = load_event_info()

    # Identify normal events for baseline training
    normal_events = event_info[event_info["event_label"] == "normal"]
    anomaly_events = event_info[event_info["event_label"] == "anomaly"]
    print(f"[S2] Normal events for training pool: {len(normal_events)}")
    print(f"[S2] Anomaly events (reserved): {len(anomaly_events)}")

    # Process all events
    all_train_dfs = []
    all_removal_logs = []
    event_data = {}  # Store per-event data for later stages

    print(f"\n[S2] Processing {len(event_info)} events...")
    for _, row in event_info.iterrows():
        eid = row["event_id"]
        train_df, pred_df, removal_log, feat_cols = clean_single_event(eid, row)

        # Store for later use
        event_data[eid] = {
            "train": train_df,
            "pred": pred_df,
            "label": row["event_label"],
            "asset_id": row["asset_id"],
            "event_start": row["event_start"],
            "event_end": row["event_end"],
            "description": row.get("event_description", ""),
        }

        # Only normal events go into training pool
        if row["event_label"] == "normal":
            all_train_dfs.append(train_df)

        all_removal_logs.append(removal_log)
        pct = 100 * removal_log["n_train_clean"] / max(removal_log["n_train_raw"], 1)
        print(f"  Ev{eid:3d} | {row['event_label']:7s} | Asset {row['asset_id']:>3} | "
              f"Raw: {removal_log['n_train_raw']:>6,} -> Clean: {removal_log['n_train_clean']:>6,} "
              f"({pct:.0f}%) | Status-rm: {removal_log['n_status_removed']:>4,} "
              f"Zero-rm: {removal_log['n_zero_block_removed']:>4,}")

    # ── Combine normal training pool ─────────────────────────────────────
    if not all_train_dfs:
        print("[ERROR] No training data available!")
        return

    pool = pd.concat(all_train_dfs, ignore_index=True)
    print(f"\n[S2] Global healthy training pool:")
    print(f"  Total clean rows: {len(pool):,}")
    print(f"  From {len(normal_events)} normal events")
    print(f"  Unique turbines: {pool['asset_id'].nunique()}")
    print(f"  Feature columns: {len(feat_cols)}")

    # ── Fit scaler on TRAINING DATA ONLY ─────────────────────────────────
    scaler = StandardScaler()
    scaler_cols = [c for c in feat_cols if c in pool.columns]
    scaler.fit(pool[scaler_cols].values)
    print(f"[S2] StandardScaler fitted on {len(pool):,} training rows ({len(scaler_cols)} features)")
    print(f"     CRITICAL: Scaler fitted on NORMAL events only. No prediction data used.")

    # ── Save artifacts ───────────────────────────────────────────────────
    # Save scaler
    scaler_path = MODELS_DIR / "scaler_stage2.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "columns": scaler_cols}, f)
    print(f"[S2] Scaler saved to: {scaler_path}")

    # Save feature selection config
    config = {
        "thermal_outputs": THERMAL_OUTPUTS,
        "exogenous_inputs": EXOGENOUS_INPUTS,
        "all_selected": ALL_SELECTED,
        "scaler_columns": scaler_cols,
        "n_training_rows": len(pool),
        "n_normal_events": len(normal_events),
        "training_event_ids": normal_events["event_id"].tolist(),
    }
    with open(MODELS_DIR / "feature_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Save removal log
    pd.DataFrame(all_removal_logs).to_csv(RESULTS_DIR / "stage2_cleaning_log.csv", index=False)

    # Save event data index (lightweight - just metadata, not full DFs)
    event_index = []
    for eid, data in event_data.items():
        event_index.append({
            "event_id": eid,
            "label": data["label"],
            "asset_id": data["asset_id"],
            "event_start": str(data["event_start"]),
            "event_end": str(data["event_end"]),
            "description": data["description"],
            "n_train_rows": len(data["train"]),
            "n_pred_rows": len(data["pred"]),
        })
    pd.DataFrame(event_index).to_csv(RESULTS_DIR / "stage2_event_index.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\n[S2] Stage 2 complete in {elapsed:.1f}s")
    print(f"[S2] Ready for Stage 3: SysID/ARMAX baseline")

    return pool, scaler, scaler_cols, event_data, feat_cols


if __name__ == "__main__":
    build_training_pool()
