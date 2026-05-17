"""
Stage 4: Random Forest classifier on ARMAX features to suppress false positives.
Extracts per-window features, trains RF on labeled windows, calibrates thresholds.
"""
import pandas as pd
import numpy as np
import json, pickle, time, warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).parent.parent.resolve()
DATA_ROOT = Path(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\Wind Farm C")
DATASETS_DIR = DATA_ROOT / "datasets"
MODELS_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"

NA = 3; NB = 3; RIDGE_ALPHA = 1.0
WINDOW = 432; STRIDE = 72
FORWARD_MIN = 2; FORWARD_MAX = 60
CONFIRM_COUNT = 3; CONFIRM_WINDOW = 5

def load_event_info():
    df = pd.read_csv(DATA_ROOT / "event_info.csv", sep=";")
    df.columns = df.columns.str.strip()
    for c in ["event_start", "event_end"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def load_armax():
    with open(MODELS_DIR / "armax_coefficients.json") as f:
        raw = json.load(f)
    models = {}
    for sname, data in raw.items():
        m = dict(data)
        for t in m["models"]:
            m["models"][t]["beta"] = np.array(m["models"][t]["beta"])
        models[sname] = m
    return models

def build_phi(df, targets, inputs, na, nb):
    mx = max(na, nb)
    n = len(df) - mx
    if n <= 0: return None, 0
    cols = []
    for t in targets:
        for lag in range(1, na+1):
            cols.append(df[t].values[mx-lag:mx-lag+n])
    for u in inputs:
        for lag in range(1, nb+1):
            cols.append(df[u].values[mx-lag:mx-lag+n])
    return np.column_stack(cols), mx

def extract_features(window_df, armax_model):
    """Extract feature vector from one window for one subsystem."""
    targets = armax_model["targets"]
    inputs = armax_model["inputs"]
    avail_t = [t for t in targets if t in window_df.columns]
    avail_u = [u for u in inputs if u in window_df.columns]
    if not avail_t or not avail_u: return None

    Phi, mx = build_phi(window_df, avail_t, avail_u, NA, NB)
    if Phi is None or Phi.shape[0] < 20: return None
    valid = np.isfinite(Phi).all(axis=1)
    if valid.sum() < 20: return None

    features = {}
    for t in avail_t:
        n = Phi.shape[0]
        Y = window_df[t].values[mx:mx+n]
        y_valid = np.isfinite(Y) & valid
        if y_valid.sum() < 20: return None
        Phi_c = Phi[y_valid]; Y_c = Y[y_valid]

        global_beta = armax_model["models"][t]["beta"]
        global_r2 = armax_model["models"][t]["r2"]

        # Global residuals
        res = Y_c - Phi_c @ global_beta
        features[f"res_mean_{t}"] = np.mean(res)
        features[f"res_std_{t}"] = np.std(res)
        features[f"res_rmse_{t}"] = np.sqrt(np.mean(res**2))
        features[f"res_maxabs_{t}"] = np.max(np.abs(res))
        sigma = np.std(res) + 1e-12
        features[f"res_kurtosis_{t}"] = float(np.mean(((res-np.mean(res))/sigma)**4))
        features[f"res_outlier_{t}"] = float(np.mean(np.abs(res) > 2*sigma))
        local_r2 = 1 - np.var(res)/max(np.var(Y_c), 1e-12)
        features[f"r2_local_{t}"] = local_r2
        features[f"r2_drop_{t}"] = max(0, global_r2 - local_r2)

        # Delta-theta (local Ridge vs global)
        try:
            lr = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
            lr.fit(Phi_c, Y_c)
            dt = lr.coef_ - global_beta
            features[f"dtheta_norm_{t}"] = float(np.linalg.norm(dt))
            features[f"dtheta_max_{t}"] = float(np.max(np.abs(dt)))
            features[f"dtheta_mean_{t}"] = float(np.mean(np.abs(dt)))
        except Exception:
            return None

    # Context
    for col, prefix in [("wind_speed_236_avg", "ws"), ("power_6_avg", "pw")]:
        if col in window_df.columns:
            v = window_df[col].dropna().values
            if len(v) > 0:
                features[f"{prefix}_mean"] = float(np.mean(v))
                features[f"{prefix}_std"] = float(np.std(v))
    return features

def build_labeled_dataset(event_info, armax_models):
    """Build labeled feature dataset from all events."""
    print("[S4] Building labeled window features...")
    all_rows = []
    for _, ev in event_info.iterrows():
        eid = ev["event_id"]; label = ev["event_label"]
        ev_start = ev["event_start"]

        csv_path = DATASETS_DIR / f"{eid}.csv"
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
        df.columns = df.columns.str.strip()
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.sort_values("time_stamp").reset_index(drop=True)

        n = len(df)
        live_start = int(n * 0.40)
        i = live_start
        while i + WINDOW <= n:
            window_df = df.iloc[i:i+WINDOW]
            ts_mid = window_df["time_stamp"].iloc[len(window_df)//2]

            # Combine features from ALL subsystems
            combined = {"event_id": eid, "ts_mid": ts_mid}
            valid = True
            for sname, armax in armax_models.items():
                feats = extract_features(window_df, armax)
                if feats is None:
                    valid = False; break
                for k, v in feats.items():
                    combined[f"{sname}__{k}"] = v
            if not valid:
                i += STRIDE; continue

            # Label this window
            if label == "anomaly" and pd.notna(ev_start):
                days_before = (ev_start - ts_mid).total_seconds() / 86400
                if FORWARD_MIN <= days_before <= FORWARD_MAX:
                    combined["window_label"] = 1  # Pre-fault
                else:
                    combined["window_label"] = 0  # Normal
            else:
                combined["window_label"] = 0  # Normal

            all_rows.append(combined)
            i += STRIDE

        tag = "ANOM" if label == "anomaly" else "NORM"
        print(f"  Ev{eid:3d} [{tag}] -> {sum(1 for r in all_rows if r['event_id']==eid)} windows")

    feat_df = pd.DataFrame(all_rows)
    print(f"\n[S4] Total windows: {len(feat_df):,}")
    print(f"  Label=1 (pre-fault): {(feat_df['window_label']==1).sum():,}")
    print(f"  Label=0 (normal):    {(feat_df['window_label']==0).sum():,}")
    return feat_df

def train_classifier(feat_df):
    """Train RF classifier with cross-validation."""
    feat_cols = [c for c in feat_df.columns if c not in ["event_id", "ts_mid", "window_label"]]
    X = feat_df[feat_cols].values
    y = feat_df["window_label"].values

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n[S4] Training RF on {X.shape[0]:,} windows, {X.shape[1]} features")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        pred = rf.predict(X[te])
        f1 = f1_score(y[te], pred)
        f1s.append(f1)
        print(f"  Fold {fold+1}: F1={f1:.4f}")
    print(f"  Mean F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Final model on all data
    final_rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
    final_rf.fit(X, y)

    # Feature importance
    importances = pd.Series(final_rf.feature_importances_, index=feat_cols)
    top20 = importances.nlargest(20)
    print(f"\n[S4] Top 20 features:")
    for fname, imp in top20.items():
        print(f"  {fname:50s}: {imp:.4f}")

    return final_rf, feat_cols, importances

def run_event_detection(event_info, armax_models, classifier, feat_cols):
    """Run final event-level detection using trained RF."""
    print(f"\n[S4] Running RF-enhanced detection on {len(event_info)} events...")
    all_results = []

    for _, ev in event_info.iterrows():
        eid = ev["event_id"]; label = ev["event_label"]
        ev_start = ev["event_start"]

        csv_path = DATASETS_DIR / f"{eid}.csv"
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
        df.columns = df.columns.str.strip()
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.sort_values("time_stamp").reset_index(drop=True)

        n = len(df)
        live_start = int(n * 0.40)
        predictions = []; timestamps = []

        i = live_start
        while i + WINDOW <= n:
            window_df = df.iloc[i:i+WINDOW]
            ts_mid = window_df["time_stamp"].iloc[len(window_df)//2]

            combined = {}
            valid = True
            for sname, armax in armax_models.items():
                feats = extract_features(window_df, armax)
                if feats is None:
                    valid = False; break
                for k, v in feats.items():
                    combined[f"{sname}__{k}"] = v

            if valid:
                x = np.array([[combined.get(c, 0.0) for c in feat_cols]])
                x = np.nan_to_num(x, nan=0.0)
                prob = classifier.predict_proba(x)[0][1]
                predictions.append(prob)
                timestamps.append(ts_mid)
            i += STRIDE

        # Apply threshold + streak confirmation
        threshold = 0.5
        alarms = [p > threshold for p in predictions]
        confirmed = []
        for j in range(len(alarms)):
            if j < CONFIRM_COUNT - 1:
                confirmed.append(False); continue
            streak = sum(alarms[max(0, j-CONFIRM_WINDOW+1):j+1])
            confirmed.append(streak >= CONFIRM_COUNT)

        result = {"event_id": eid, "label": label, "asset_id": ev["asset_id"],
                  "description": ev.get("event_description", "")}

        if label == "anomaly" and pd.notna(ev_start):
            best_lead = None
            for j, (conf, ts) in enumerate(zip(confirmed, timestamps)):
                if not conf: continue
                days = (ev_start - ts).total_seconds() / 86400
                if FORWARD_MIN <= days <= FORWARD_MAX:
                    if best_lead is None or days > best_lead:
                        best_lead = days
            result["detected"] = best_lead is not None
            result["lead_days"] = round(best_lead, 1) if best_lead else None
        else:
            result["false_positive"] = any(confirmed)

        all_results.append(result)
        if label == "anomaly":
            det = "TP" if result.get("detected") else "FN"
            ld = f"{result.get('lead_days','-'):>5}" if result.get("detected") else "  -  "
            print(f"  Ev{eid:3d} [ANOM] A{ev['asset_id']:>3} | {det} | Lead:{ld}d")
        else:
            fp = "FP!" if result.get("false_positive") else "OK "
            print(f"  Ev{eid:3d} [NORM] A{ev['asset_id']:>3} | {fp}")

    return pd.DataFrame(all_results)

def main():
    print("=" * 80)
    print("STAGE 4: ML CLASSIFIER (RF) ON ARMAX FEATURES")
    print("=" * 80)
    t0 = time.time()

    event_info = load_event_info()
    armax_models = load_armax()

    # Build labeled dataset
    feat_df = build_labeled_dataset(event_info, armax_models)
    feat_df.to_csv(RESULTS_DIR / "stage4_window_features.csv", index=False)

    # Train classifier
    classifier, feat_cols, importances = train_classifier(feat_df)
    with open(MODELS_DIR / "rf_classifier.pkl", "wb") as f:
        pickle.dump({"model": classifier, "columns": feat_cols}, f)
    importances.to_csv(RESULTS_DIR / "stage4_feature_importance.csv")

    # Event-level detection
    results_df = run_event_detection(event_info, armax_models, classifier, feat_cols)
    results_df.to_csv(RESULTS_DIR / "stage4_detection_results.csv", index=False)

    # Summary
    anom = results_df[results_df["label"] == "anomaly"]
    norm = results_df[results_df["label"] == "normal"]
    tp = anom["detected"].sum() if "detected" in anom.columns else 0
    fn = len(anom) - tp
    fp = norm["false_positive"].sum() if "false_positive" in norm.columns else 0
    leads = anom.loc[anom.get("detected", False)==True, "lead_days"]
    ml = leads.mean() if len(leads) > 0 else 0

    print(f"\n{'='*60}")
    print(f"STAGE 4 RESULTS (ARMAX + RF)")
    print(f"{'='*60}")
    print(f"  TP: {tp}/{len(anom)}  FN: {fn}  FP: {fp}/{len(norm)}")
    print(f"  Recall: {tp/max(len(anom),1)*100:.0f}%  Precision: {tp/max(tp+fp,1)*100:.0f}%")
    print(f"  Mean Lead: {ml:.1f}d")
    print(f"\n[S4] Complete in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
