"""
Stage 3: MIMO ARMAX System Identification for Wind Farm C.
Fits global baseline on normal-event training data, computes residuals,
extracts delta-theta features, and runs event-based anomaly detection.
"""
import pandas as pd
import numpy as np
import json, pickle, time, warnings
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).parent.parent.resolve()
DATA_ROOT = Path(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\Wind Farm C")
DATASETS_DIR = DATA_ROOT / "datasets"
MODELS_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"

# ── Subsystem definitions (from Stage 2) ─────────────────────────────────
SUBSYSTEMS = {
    "gearbox": {
        "targets": ["sensor_186_avg", "sensor_187_avg"],
        "inputs":  ["sensor_7_avg", "wind_speed_236_avg", "power_6_avg", "sensor_144_avg"],
    },
    "generator": {
        "targets": ["sensor_199_avg", "sensor_200_avg", "sensor_228_avg"],
        "inputs":  ["sensor_7_avg", "wind_speed_236_avg", "power_6_avg", "sensor_8_avg"],
    },
    "transformer": {
        "targets": ["sensor_191_avg", "sensor_192_avg"],
        "inputs":  ["sensor_7_avg", "wind_speed_236_avg", "power_6_avg"],
    },
    "hydraulic": {
        "targets": ["sensor_178_avg", "sensor_179_avg"],
        "inputs":  ["sensor_7_avg", "wind_speed_236_avg", "power_6_avg"],
    },
    "rotor_bearing": {
        "targets": ["sensor_196_avg", "sensor_197_avg"],
        "inputs":  ["sensor_7_avg", "wind_speed_236_avg", "power_6_avg", "sensor_144_avg"],
    },
}
NA = 3  # AR order
NB = 3  # Exogenous lag order
NC = 2  # MA order
RIDGE_ALPHA = 1.0
WINDOW = 432    # 3 days at 10-min
STRIDE = 72     # 12-hour stride
LIVE_FRAC = 0.40
FORWARD_MIN = 2
FORWARD_MAX = 60
CONFIRM_COUNT = 2
CONFIRM_WINDOW = 4
NONDISCRIM_FRAC = 0.40
RECOVERY_DAYS = 10

META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]

def load_event_info():
    df = pd.read_csv(DATA_ROOT / "event_info.csv", sep=";")
    df.columns = df.columns.str.strip()
    for c in ["event_start", "event_end"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def build_regression_matrix(df, targets, inputs, na, nb):
    """Build ARMAX regression matrix Phi and output vector Y."""
    max_lag = max(na, nb)
    n = len(df) - max_lag
    if n <= 0:
        return None, None, 0
    cols = []
    # AR terms: past values of ALL targets
    for t in targets:
        for lag in range(1, na + 1):
            cols.append(df[t].values[max_lag - lag: max_lag - lag + n])
    # Exogenous terms: past values of inputs
    for u in inputs:
        for lag in range(1, nb + 1):
            cols.append(df[u].values[max_lag - lag: max_lag - lag + n])
    Phi = np.column_stack(cols)
    n_params = len(targets) * na + len(inputs) * nb
    return Phi, max_lag, n_params

def fit_armax_global(pool_df, subsys_name, config):
    """Fit global ARMAX baseline for one subsystem using Ridge + ELS."""
    targets = config["targets"]
    inputs = config["inputs"]
    avail_t = [t for t in targets if t in pool_df.columns]
    avail_u = [u for u in inputs if u in pool_df.columns]
    if not avail_t or not avail_u:
        return None

    # Group by asset, build regression matrices
    all_Phi, all_Y = [], {t: [] for t in avail_t}
    for asset_id, grp in pool_df.groupby("asset_id"):
        grp = grp.sort_values("time_stamp").reset_index(drop=True)
        Phi, max_lag, n_params = build_regression_matrix(grp, avail_t, avail_u, NA, NB)
        if Phi is None:
            continue
        n = Phi.shape[0]
        all_Phi.append(Phi)
        for t in avail_t:
            all_Y[t].append(grp[t].values[max_lag: max_lag + n])

    if not all_Phi:
        return None

    Phi_global = np.vstack(all_Phi)
    models = {}
    for t in avail_t:
        Y = np.concatenate(all_Y[t])
        # Drop NaN rows
        valid = np.isfinite(Phi_global).all(axis=1) & np.isfinite(Y)
        Phi_clean = Phi_global[valid]
        Y_clean = Y[valid]
        if len(Y_clean) < 100:
            print(f"    {t}: SKIPPED (only {len(Y_clean)} valid rows)")
            continue
        # Ridge regression
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        ridge.fit(Phi_clean, Y_clean)
        beta = ridge.coef_
        # Compute residuals for MA estimation (ELS)
        residuals = Y_clean - Phi_clean @ beta
        r2 = 1 - np.var(residuals) / max(np.var(Y_clean), 1e-12)
        models[t] = {"beta": beta, "r2": float(r2), "n_params": len(beta)}
        print(f"    {t}: R2={r2:.4f}, params={len(beta)}, samples={len(Y_clean):,}")

    return {
        "targets": avail_t, "inputs": avail_u, "models": models,
        "na": NA, "nb": NB, "n_samples": Phi_global.shape[0],
    }

def compute_residuals_window(df, armax, start, end):
    """Compute ARMAX residuals for a window of data."""
    targets = armax["targets"]
    inputs = armax["inputs"]
    window = df.iloc[start:end]
    if len(window) < max(NA, NB) + 10:
        return None, None
    Phi, max_lag, _ = build_regression_matrix(window, targets, inputs, NA, NB)
    if Phi is None:
        return None, None
    n = Phi.shape[0]
    residuals = {}
    for t in targets:
        Y = window[t].values[max_lag: max_lag + n]
        beta = armax["models"][t]["beta"]
        pred = Phi @ beta
        residuals[t] = Y - pred
    return residuals, n

def extract_window_features(residuals, df_window, armax, global_models):
    """Extract 49-dim feature vector from one window."""
    features = {}
    # Delta-theta: fit local Ridge, compute drift from global
    targets = armax["targets"]
    inputs = armax["inputs"]
    Phi, max_lag, _ = build_regression_matrix(df_window, targets, inputs, NA, NB)
    if Phi is None or Phi.shape[0] < 20:
        return None

    for t in targets:
        n = Phi.shape[0]
        Y = df_window[t].values[max_lag: max_lag + n]
        try:
            local_ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
            local_ridge.fit(Phi, Y)
            delta_theta = local_ridge.coef_ - global_models[t]["beta"]
            for i, dt in enumerate(delta_theta):
                features[f"dtheta_{t}_{i}"] = dt
        except Exception:
            return None

        # Residual statistics
        res = Y - Phi @ global_models[t]["beta"]
        features[f"res_mean_{t}"] = np.mean(res)
        features[f"res_std_{t}"] = np.std(res)
        features[f"res_maxabs_{t}"] = np.max(np.abs(res))
        features[f"res_rmse_{t}"] = np.sqrt(np.mean(res**2))
        sigma = np.std(res) + 1e-12
        features[f"res_outlier_{t}"] = np.mean(np.abs(res) > 2 * sigma)
        local_r2 = 1 - np.var(res) / max(np.var(Y), 1e-12)
        features[f"res_r2_{t}"] = local_r2

    # Operating context
    if "wind_speed_236_avg" in df_window.columns:
        ws = df_window["wind_speed_236_avg"].values
        features["ctx_wind_mean"] = np.mean(ws)
        features["ctx_wind_std"] = np.std(ws)
        features["ctx_wind_max"] = np.max(ws)
        features["ctx_highwind"] = np.mean(ws > 12)
    if "power_6_avg" in df_window.columns:
        features["ctx_power_mean"] = np.mean(df_window["power_6_avg"].values)
    return features

def run_detection(event_data, armax_models, scaler_info, event_info_row):
    """Run sliding-window detection on one event's full timeline."""
    eid = event_info_row["event_id"]
    label = event_info_row["event_label"]

    # Load full event CSV
    csv_path = DATASETS_DIR / f"{eid}.csv"
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    df.columns = df.columns.str.strip()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df = df.sort_values("time_stamp").reset_index(drop=True)

    # Determine live start (skip burn-in)
    n_total = len(df)
    live_start = int(n_total * LIVE_FRAC)

    # Per-subsystem anomaly scores
    subsys_scores = {}
    for sname, armax in armax_models.items():
        targets = armax["targets"]
        scores = []
        timestamps = []
        global_models = armax["models"]

        i = live_start
        while i + WINDOW <= n_total:
            window_df = df.iloc[i:i + WINDOW]
            feats = extract_window_features(
                None, window_df, armax, global_models
            )
            if feats is not None:
                # Anomaly score = mean absolute delta-theta + residual deviation
                dtheta_vals = [v for k, v in feats.items() if k.startswith("dtheta_")]
                res_rmse_vals = [v for k, v in feats.items() if k.startswith("res_rmse_")]
                res_r2_vals = [v for k, v in feats.items() if k.startswith("res_r2_")]

                score = np.mean(np.abs(dtheta_vals)) + np.mean(res_rmse_vals)
                if res_r2_vals:
                    r2_drop = max(0, armax["models"][targets[0]]["r2"] - np.mean(res_r2_vals))
                    score += r2_drop * 10

                scores.append(score)
                ts_mid = window_df["time_stamp"].iloc[len(window_df)//2]
                timestamps.append(ts_mid)
            i += STRIDE

        subsys_scores[sname] = {"scores": scores, "timestamps": timestamps}

    return subsys_scores

def event_level_evaluation(subsys_scores, event_info_row, armax_models):
    """Evaluate detection results against 2-60 day window."""
    eid = event_info_row["event_id"]
    label = event_info_row["event_label"]
    ev_start = event_info_row["event_start"]

    results = {"event_id": eid, "label": label, "asset_id": event_info_row["asset_id"],
               "description": event_info_row.get("event_description", "")}

    best_lead = None
    best_subsys = None
    detected = False

    for sname, data in subsys_scores.items():
        scores = data["scores"]
        timestamps = data["timestamps"]
        if not scores:
            continue

        # Adaptive threshold: mean + 2.5*std of first 30% of scores (assumed healthy)
        n_baseline = max(int(len(scores) * 0.3), 10)
        baseline = scores[:n_baseline]
        threshold = np.mean(baseline) + 2.5 * np.std(baseline)

        # Apply streak confirmation
        alarm_windows = [s > threshold for s in scores]
        confirmed = []
        for j in range(len(alarm_windows)):
            if j < CONFIRM_COUNT - 1:
                confirmed.append(False)
                continue
            streak = sum(alarm_windows[max(0, j - CONFIRM_WINDOW + 1):j + 1])
            confirmed.append(streak >= CONFIRM_COUNT)

        # Non-discriminatory filter
        fire_rate = sum(confirmed) / max(len(confirmed), 1)
        if fire_rate > NONDISCRIM_FRAC:
            continue

        # Check for detections in valid window
        if label == "anomaly" and pd.notna(ev_start):
            for j, (conf, ts) in enumerate(zip(confirmed, timestamps)):
                if not conf:
                    continue
                days_before = (ev_start - ts).total_seconds() / 86400
                if FORWARD_MIN <= days_before <= FORWARD_MAX:
                    if best_lead is None or days_before > best_lead:
                        best_lead = days_before
                        best_subsys = sname
                        detected = True

        elif label == "normal":
            # Any confirmed alarm is a false positive
            if any(confirmed):
                results["false_positive"] = True
                first_fp_idx = next(i for i, c in enumerate(confirmed) if c)
                results["fp_subsystem"] = sname
            else:
                results["false_positive"] = False

    if label == "anomaly":
        results["detected"] = detected
        results["lead_days"] = round(best_lead, 1) if best_lead else None
        results["detecting_subsystem"] = best_subsys
    return results

def main():
    print("=" * 80)
    print("STAGE 3: ARMAX SYSTEM IDENTIFICATION + ANOMALY DETECTION")
    print("=" * 80)
    t0 = time.time()

    event_info = load_event_info()
    normal_events = event_info[event_info["event_label"] == "normal"]

    # Load scaler
    with open(MODELS_DIR / "scaler_stage2.pkl", "rb") as f:
        scaler_info = pickle.load(f)

    # ── Step 1: Build global training pool ────────────────────────────────
    print("\n[S3] Loading normal-event training data...")
    train_dfs = []
    for _, row in normal_events.iterrows():
        csv_path = DATASETS_DIR / f"{row['event_id']}.csv"
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
        df.columns = df.columns.str.strip()
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        mask = (df["train_test"] == "train") & (df["status_type_id"].isin([0, 2]))
        train_dfs.append(df[mask])
    pool = pd.concat(train_dfs, ignore_index=True)
    print(f"[S3] Training pool: {len(pool):,} rows from {len(normal_events)} normal events")

    # ── Step 2: Fit ARMAX per subsystem ───────────────────────────────────
    print("\n[S3] Fitting ARMAX models per subsystem...")
    armax_models = {}
    for sname, config in SUBSYSTEMS.items():
        print(f"  [{sname.upper()}]")
        result = fit_armax_global(pool, sname, config)
        if result:
            armax_models[sname] = result
            print(f"    -> {result['n_samples']:,} samples, {len(result['targets'])} outputs")

    # Save ARMAX coefficients
    armax_save = {}
    for sname, model in armax_models.items():
        armax_save[sname] = {
            "targets": model["targets"], "inputs": model["inputs"],
            "na": model["na"], "nb": model["nb"],
            "n_samples": model["n_samples"],
            "models": {t: {"beta": m["beta"].tolist(), "r2": m["r2"]}
                       for t, m in model["models"].items()},
        }
    with open(MODELS_DIR / "armax_coefficients.json", "w") as f:
        json.dump(armax_save, f, indent=2)
    print(f"[S3] ARMAX coefficients saved")

    # ── Step 3: Run detection on ALL events ───────────────────────────────
    print(f"\n[S3] Running detection on {len(event_info)} events...")
    all_results = []
    for _, row in event_info.iterrows():
        eid = row["event_id"]
        subsys_scores = run_detection(None, armax_models, scaler_info, row)
        result = event_level_evaluation(subsys_scores, row, armax_models)
        all_results.append(result)

        if row["event_label"] == "anomaly":
            det = "TP" if result.get("detected") else "FN"
            lead = f"{result.get('lead_days', '-'):>5}" if result.get("detected") else "  -  "
            sub = result.get("detecting_subsystem") or "-"
            print(f"  Ev{eid:3d} [ANOMALY] Asset {row['asset_id']:>3} | "
                  f"{det} | Lead: {lead}d | By: {sub:15s} | {str(row.get('event_description',''))[:40]}")
        else:
            fp = "FP!" if result.get("false_positive") else "OK "
            print(f"  Ev{eid:3d} [NORMAL ] Asset {row['asset_id']:>3} | {fp}")

    # ── Step 4: Summary ──────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    anomaly_results = results_df[results_df["label"] == "anomaly"]
    normal_results = results_df[results_df["label"] == "normal"]

    tp = anomaly_results["detected"].sum() if "detected" in anomaly_results.columns else 0
    fn = len(anomaly_results) - tp
    fp = normal_results["false_positive"].sum() if "false_positive" in normal_results.columns else 0

    detected_leads = anomaly_results.loc[anomaly_results.get("detected", False) == True, "lead_days"]
    mean_lead = detected_leads.mean() if len(detected_leads) > 0 else 0

    print(f"\n{'='*60}")
    print(f"STAGE 3 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  True Positives:  {tp}/{len(anomaly_results)}")
    print(f"  False Negatives: {fn}")
    print(f"  False Positives: {fp}/{len(normal_results)}")
    print(f"  Mean Lead Time:  {mean_lead:.1f} days")
    if len(detected_leads) > 0:
        print(f"  Min Lead Time:   {detected_leads.min():.1f} days")
        print(f"  Max Lead Time:   {detected_leads.max():.1f} days")
    print(f"  Recall:          {tp/max(len(anomaly_results),1)*100:.0f}%")
    print(f"  Precision:       {tp/max(tp+fp,1)*100:.0f}%")

    results_df.to_csv(RESULTS_DIR / "stage3_detection_results.csv", index=False)
    print(f"\n[S3] Results saved to {RESULTS_DIR / 'stage3_detection_results.csv'}")
    print(f"[S3] Stage 3 complete in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
