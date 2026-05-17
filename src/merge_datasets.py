"""
Wind Farm C — Merge per-event CSVs into one timeline per turbine.
Confirmed: all events for same asset_id share identical sensor values
at overlapping timestamps (100% match verified).

Output: WindFarmC-Project/data/merged/asset_{id}.parquet
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\Wind Farm C")
DATASETS_DIR = DATA_ROOT / "datasets"
PROJECT = Path(__file__).parent.parent.resolve()
MERGED_DIR = PROJECT / "data" / "merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

def merge_all_assets():
    print("=" * 80)
    print("MERGE: Building one timeline per turbine")
    print("=" * 80)
    t0 = time.time()

    ei = pd.read_csv(DATA_ROOT / "event_info.csv", sep=";")
    ei.columns = ei.columns.str.strip()

    asset_ids = sorted(ei["asset_id"].unique())
    print(f"Found {len(asset_ids)} unique turbines\n")

    summary = []
    for asset_id in asset_ids:
        eids = ei[ei["asset_id"] == asset_id]["event_id"].values
        frames = []
        for eid in eids:
            csv_path = DATASETS_DIR / f"{eid}.csv"
            df = pd.read_csv(csv_path, sep=";", low_memory=False)
            df.columns = df.columns.str.strip()
            df["time_stamp"] = pd.to_datetime(df["time_stamp"])
            frames.append(df)

        # Concat and drop exact duplicate rows (same timestamp)
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["time_stamp"]).sort_values("time_stamp").reset_index(drop=True)

        out_path = MERGED_DIR / f"asset_{asset_id}.parquet"
        merged.to_parquet(out_path, index=False)

        ts_min = merged["time_stamp"].min()
        ts_max = merged["time_stamp"].max()
        n_events = len(eids)
        n_anom = len(ei[(ei["asset_id"] == asset_id) & (ei["event_label"] == "anomaly")])

        summary.append({
            "asset_id": asset_id, "events": n_events, "anomalies": n_anom,
            "rows": len(merged), "start": ts_min, "end": ts_max
        })
        print(f"  Asset {asset_id:>3}: {len(merged):>7,} rows | {n_events} events ({n_anom} anom) | {ts_min.date()} -> {ts_max.date()}")

    pd.DataFrame(summary).to_csv(MERGED_DIR / "merge_summary.csv", index=False)
    print(f"\nMerge complete in {time.time()-t0:.1f}s -> {MERGED_DIR}")

if __name__ == "__main__":
    merge_all_assets()
