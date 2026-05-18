"""
CARE-COMPLIANT DETECTION v3 — Normal Behavior Model
====================================================
Based on CWD2017 (Tautz-Weinert): Train a model of normal behavior
on Part 1, predict on Part 2, flag when residuals exceed threshold.

Key improvements:
- NBM uses operational context (wind speed, power) to predict temperatures
- Residual-based detection eliminates seasonal shift problem
- Status filtering (status_type_id=0 only)
- CUSUM on residuals with h=20 (much less sensitive)
- Proper CARE scoring with tc=72
"""
import pandas as pd, numpy as np, os, warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
anomaly_ids = sorted(ei[ei["event_label"]=="anomaly"]["event_id"].tolist())
normal_ids = sorted(ei[ei["event_label"]=="normal"]["event_id"].tolist())
all_ids = sorted(anomaly_ids + normal_ids)

# Context features (inputs to NBM)
CONTEXT = ["wind_speed_avg","active_power_avg","reactive_power_avg"]

# Target sensors to model (temperatures, pressures, currents)
TARGETS = {
    "gearbox_temp": "sensor_186_avg",
    "generator_temp": "sensor_173_avg",
    "bearing_temp": "sensor_194_avg",
    "transformer_temp": "sensor_191_avg",
    "hydraulic_temp": "sensor_178_avg",
    "cabinet_temp": "sensor_39_avg",
    "pitch_temp": "sensor_62_avg",
    "hydraulic_pressure": "sensor_48_avg",
    "oil_level": "sensor_74_avg",
    "24v_current": "sensor_25_avg",
    "battery_ax1": "sensor_12_avg",
    "battery_ax2": "sensor_13_avg",
    "battery_ax3": "sensor_14_avg",
    "mains_freq": "sensor_47_avg",
    "abb_voltage": "sensor_58_avg",
    "filter_pressure": "sensor_109_avg",
    "oil_container_a1": "sensor_44_avg",
    "gear_pump_current": "sensor_87_avg",
    "reactive_power_hv": "sensor_75_avg",
    "bearing2_temp": "sensor_195_avg",
}

def cusum_on_residuals(residuals, h=20.0, k=1.0):
    """CUSUM on normalized residuals. More conservative than raw CUSUM."""
    n = len(residuals)
    pos = np.zeros(n); neg = np.zeros(n)
    alarm = np.zeros(n, dtype=int)
    for i in range(1, n):
        pos[i] = max(0, pos[i-1] + residuals[i] - k)
        neg[i] = max(0, neg[i-1] - residuals[i] - k)
        if pos[i] > h or neg[i] > h:
            alarm[i] = 1
    return alarm

def care_criticality(predictions, status, tc=72):
    n = len(predictions)
    crit = np.zeros(n+1)
    for i in range(n):
        if status[i] == 0:  # Normal status only
            if predictions[i] == 1:
                crit[i+1] = crit[i] + 1
            else:
                crit[i+1] = max(crit[i] - 1, 0)
        else:
            crit[i+1] = crit[i]
    return np.max(crit[1:]), crit[1:]

def f_beta(tp, fp, fn, beta=0.5):
    if tp == 0: return 0.0
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)

def weighted_score(predictions, status, event_start_idx, event_end_idx):
    if event_end_idx <= event_start_idx: return 0.0
    M = event_end_idx - event_start_idx
    total_w = 0.0; weighted_sum = 0.0
    for i in range(M):
        idx = event_start_idx + i
        if status[idx] != 0: continue  # Skip abnormal status
        rel_pos = i / max(M-1, 1)
        w = 1.0 if rel_pos <= 0.5 else max(0, 2.0 * (1.0 - rel_pos))
        total_w += w
        if predictions[idx] == 1:
            weighted_sum += w
    return weighted_sum / total_w if total_w > 0 else 0.0

print("="*100)
print("CARE v3: Normal Behavior Model + CUSUM Residuals")
print("="*100)

results = []
nbm_stats = []

for eid in all_ids:
    f1 = os.path.join(DATA, f"event_{eid}_part1.pkl")
    f2 = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if not os.path.exists(f1) or not os.path.exists(f2): continue
    
    df_train = pd.read_pickle(f1)
    df_pred = pd.read_pickle(f2)
    is_anomaly = eid in anomaly_ids
    label = "anomaly" if is_anomaly else "normal"
    n_pred = len(df_pred)
    
    status = df_pred["status_type_id"].values if "status_type_id" in df_pred.columns else np.zeros(n_pred)
    
    # Event timing
    event_start_idx = None; event_end_idx = None
    if is_anomaly:
        ev = ei[ei["event_id"]==eid].iloc[0]
        esi = int(ev["event_start_id"]) if pd.notna(ev["event_start_id"]) else None
        eei = int(ev["event_end_id"]) if pd.notna(ev["event_end_id"]) else None
        if esi and "row_id" in df_pred.columns:
            m = df_pred["row_id"] >= esi
            if m.any(): event_start_idx = m.values.argmax()
        if eei and "row_id" in df_pred.columns:
            m = df_pred["row_id"] >= eei
            if m.any(): event_end_idx = m.values.argmax()
    
    # ---- NBM TRAINING (Part 1, normal-status only) ----
    train_mask = df_train["status_type_id"].values == 0 if "status_type_id" in df_train.columns else np.ones(len(df_train), dtype=bool)
    
    # Find available context features
    avail_ctx = [c for c in CONTEXT if c in df_train.columns and c in df_pred.columns]
    if not avail_ctx:
        # Fallback: use first 3 numeric columns as context
        avail_ctx = [c for c in df_train.columns if "_avg" in c][:3]
    
    predictions = np.zeros(n_pred, dtype=int)
    channel_alarms = {}
    n_models = 0
    
    for ch_name, sensor in TARGETS.items():
        if sensor not in df_train.columns or sensor not in df_pred.columns:
            continue
        
        # Prepare features
        feat_cols = [c for c in avail_ctx if c != sensor]
        if not feat_cols: continue
        
        # Training data (normal status only)
        X_train = df_train.loc[train_mask, feat_cols].values
        y_train = df_train.loc[train_mask, sensor].values
        
        # Remove NaN/inf
        valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[valid]; y_train = y_train[valid]
        if len(X_train) < 100: continue
        
        # Scale
        scaler_x = StandardScaler().fit(X_train)
        scaler_y_mean = y_train.mean(); scaler_y_std = y_train.std()
        if scaler_y_std < 1e-6: continue
        
        X_train_s = scaler_x.transform(X_train)
        y_train_s = (y_train - scaler_y_mean) / scaler_y_std
        
        # Train Ridge regression (simple, robust NBM as per CWD2017)
        model = Ridge(alpha=1.0).fit(X_train_s, y_train_s)
        
        # Predict on Part 2
        X_pred = df_pred[feat_cols].fillna(0).values
        valid_pred = np.isfinite(X_pred).all(axis=1)
        X_pred_s = scaler_x.transform(np.where(np.isfinite(X_pred), X_pred, 0))
        y_pred_actual = df_pred[sensor].fillna(scaler_y_mean).values
        y_pred_model = model.predict(X_pred_s) * scaler_y_std + scaler_y_mean
        
        # Compute residuals
        residuals = y_pred_actual - y_pred_model
        
        # Compute residual statistics from training
        y_train_pred = model.predict(X_train_s) * scaler_y_std + scaler_y_mean
        train_residuals = y_train - y_train_pred
        res_mean = train_residuals.mean()
        res_std = train_residuals.std()
        if res_std < 1e-6: continue
        
        # Normalized residuals
        norm_res = (residuals - res_mean) / res_std
        
        n_models += 1
        
        # Method 1: CUSUM on normalized residuals (h=20, k=1.0)
        cusum_alarm = cusum_on_residuals(norm_res, h=20.0, k=1.0)
        
        # Method 2: Rolling window sigma check (3-day window)
        W = 432; S = 144
        sigma_alarm = np.zeros(n_pred, dtype=int)
        for wi in range(0, n_pred - W + 1, S):
            wm = np.mean(norm_res[wi:wi+W])
            if abs(wm) > 4.0:  # 4-sigma threshold on residuals
                sigma_alarm[wi:wi+W] = 1
        
        ch_alarm = np.maximum(cusum_alarm, sigma_alarm)
        channel_alarms[ch_name] = ch_alarm
        predictions = np.maximum(predictions, ch_alarm)
    
    # ---- CARE SCORING ----
    normal_mask = (status == 0)
    
    if is_anomaly:
        # Ground truth: 1 during event window
        gt = np.zeros(n_pred, dtype=int)
        if event_start_idx is not None:
            end_idx = event_end_idx if event_end_idx else n_pred
            gt[event_start_idx:end_idx] = 1
        
        gt_f = gt[normal_mask]; pred_f = predictions[normal_mask]
        tp = np.sum((gt_f==1)&(pred_f==1))
        fp = np.sum((gt_f==0)&(pred_f==1))
        fn = np.sum((gt_f==1)&(pred_f==0))
        tn = np.sum((gt_f==0)&(pred_f==0))
        coverage = f_beta(tp, fp, fn, beta=0.5)
        
        ws = 0.0
        if event_start_idx is not None:
            end_idx = event_end_idx if event_end_idx else n_pred
            ws = weighted_score(predictions, status, event_start_idx, end_idx)
        
        max_crit, crit_arr = care_criticality(predictions, status, tc=72)
        detected = max_crit >= 72
        
        lead_days = None
        if detected and event_start_idx:
            crit_thresh_idx = np.argmax(crit_arr >= 72)
            if crit_thresh_idx < event_start_idx:
                lead_days = (event_start_idx - crit_thresh_idx) * 10 / (60 * 24)
        
        n_ch = sum(1 for v in channel_alarms.values() if np.sum(v) > 0)
        pct_alarm = np.mean(predictions[normal_mask]) * 100 if normal_mask.sum() > 0 else 0
        
        results.append({
            "event_id": eid, "label": label, "detected": detected,
            "max_crit": max_crit, "coverage_f05": coverage,
            "weighted_score": ws, "lead_days": lead_days,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_ch": n_ch, "n_models": n_models,
            "pct_alarm": pct_alarm,
        })
    else:
        pred_f = predictions[normal_mask]
        fp = int(np.sum(pred_f == 1))
        tn = int(np.sum(pred_f == 0))
        acc = tn / (fp + tn) if (fp + tn) > 0 else 1.0
        
        max_crit, _ = care_criticality(predictions, status, tc=72)
        false_alarm = max_crit >= 72
        
        n_ch = sum(1 for v in channel_alarms.values() if np.sum(v) > 0)
        pct_alarm = np.mean(predictions[normal_mask]) * 100 if normal_mask.sum() > 0 else 0
        
        results.append({
            "event_id": eid, "label": label,
            "max_crit": max_crit, "accuracy": acc,
            "false_alarm": false_alarm,
            "fp": fp, "tn": tn, "n_ch": n_ch, "n_models": n_models,
            "pct_alarm": pct_alarm,
        })

# ================================================================
# REPORT
# ================================================================
df_res = pd.DataFrame(results)
df_anom = df_res[df_res["label"]=="anomaly"]
df_norm = df_res[df_res["label"]=="normal"]

print(f"\n{'='*100}")
print("CARE-COMPLIANT RESULTS (NBM + CUSUM h=20)")
print("="*100)

# Anomaly datasets
tp_events = df_anom["detected"].sum()
fn_events = len(df_anom) - tp_events
print(f"\n--- ANOMALY DATASETS ({len(df_anom)}) ---")
print(f"Events detected: {tp_events}/{len(df_anom)} ({tp_events/len(df_anom)*100:.1f}%)")
if fn_events > 0:
    missed = df_anom[~df_anom["detected"]]
    for _, r in missed.iterrows():
        print(f"  MISSED Event {int(r['event_id'])}: max_crit={r['max_crit']:.0f} ({r['n_ch']} channels, {r['pct_alarm']:.1f}% alarm)")

mean_cov = df_anom["coverage_f05"].mean()
mean_ws = df_anom["weighted_score"].mean()
leads = df_anom.dropna(subset=["lead_days"])
mean_lead = leads["lead_days"].mean() if len(leads) > 0 else 0

print(f"\nCoverage (mean F_0.5): {mean_cov:.4f}")
print(f"Earliness (mean WS): {mean_ws:.4f}")
print(f"Lead time: {mean_lead:.1f}d mean ({len(leads)} events with lead)")

# Normal datasets
print(f"\n--- NORMAL DATASETS ({len(df_norm)}) ---")
fp_events = df_norm["false_alarm"].sum()
tn_events = len(df_norm) - fp_events
mean_acc = df_norm["accuracy"].mean()
print(f"False alarms: {fp_events}/{len(df_norm)}")
print(f"Accuracy (mean): {mean_acc:.4f}")

if fp_events > 0:
    fa_ids = df_norm[df_norm["false_alarm"]]["event_id"].tolist()
    print(f"False alarm IDs: {fa_ids}")

# CARE composite
reliability = f_beta(tp_events, fp_events, fn_events, beta=0.5)
if tp_events > 0 and mean_acc >= 0.5:
    CARE = (mean_cov + mean_acc + reliability + mean_ws) / 4
else:
    CARE = 0.0

print(f"\n{'='*60}")
print(f"CARE SCORE BREAKDOWN")
print(f"{'='*60}")
print(f"  C (Coverage):     {mean_cov:.4f}")
print(f"  A (Accuracy):     {mean_acc:.4f}")
print(f"  R (Reliability):  {reliability:.4f}")
print(f"  E (Earliness):    {mean_ws:.4f}")
print(f"  CARE Score:       {CARE:.4f}")

# Per-event detail
print(f"\n--- PER-ANOMALY EVENT DETAIL ---")
print(f"{'ID':>4} {'Det':>4} {'Crit':>6} {'CovF05':>8} {'WS':>6} {'Lead':>6} {'Ch':>3} {'FP':>6} {'Alarm%':>7}")
for _, r in df_anom.sort_values("event_id").iterrows():
    det = "Y" if r["detected"] else "N"
    lead = f"{r['lead_days']:.0f}d" if pd.notna(r.get("lead_days")) else "-"
    print(f"{int(r['event_id']):>4} {det:>4} {r['max_crit']:>6.0f} {r['coverage_f05']:>8.4f} {r.get('weighted_score',0):>6.3f} {lead:>6} {r['n_ch']:>3} {r['fp']:>6} {r['pct_alarm']:>6.1f}%")

print(f"\n--- PER-NORMAL EVENT DETAIL ---")
print(f"{'ID':>4} {'FA':>4} {'Crit':>6} {'Acc':>8} {'Ch':>3} {'FP':>6} {'Alarm%':>7}")
for _, r in df_norm.sort_values("event_id").iterrows():
    fa = "Y" if r["false_alarm"] else "N"
    print(f"{int(r['event_id']):>4} {fa:>4} {r['max_crit']:>6.0f} {r['accuracy']:>8.4f} {r['n_ch']:>3} {r['fp']:>6} {r['pct_alarm']:>6.1f}%")

df_res.to_csv(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\care_v3_results.csv", index=False)
