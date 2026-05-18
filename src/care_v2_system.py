"""
CARE-COMPLIANT ANOMALY DETECTION SYSTEM v2.0
============================================
Fixed ALL 10 critical issues from self-critique:
1. Training on Part 1, predicting on Part 2 (no data leakage)
2. Evaluating ALL 58 datasets (27 anomaly + 31 normal)
3. Proper CARE criticality algorithm (tc=72, status filtering)
4. Computing all 4 CARE sub-scores
5. Status-filtered pointwise scoring
6. Proper lead time (detection to event_start)
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
FEAT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\feature_description.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
feat = pd.read_csv(FEAT_CSV, sep=";")
anomaly_ids = sorted(ei[ei["event_label"]=="anomaly"]["event_id"].tolist())
normal_ids = sorted(ei[ei["event_label"]=="normal"]["event_id"].tolist())
all_ids = sorted(anomaly_ids + normal_ids)

print(f"Dataset: {len(anomaly_ids)} anomaly + {len(normal_ids)} normal = {len(all_ids)} total")

# Get sensor groups
sensor_cols = [c for c in pd.read_pickle(os.path.join(DATA, "event_4_part2.pkl")).columns if "_avg" in c]
print(f"Sensors: {len(sensor_cols)}")

# CUSUM detection function (per-timestamp, CARE-compliant)
def cusum_detect(data, baseline_mean, baseline_std, k=0.5, h=5.0):
    """Two-sided CUSUM on individual timestamps. Returns binary anomaly array."""
    if baseline_std < 1e-6: return np.zeros(len(data))
    z = (data - baseline_mean) / baseline_std
    n = len(z)
    pos = np.zeros(n); neg = np.zeros(n)
    alarm = np.zeros(n, dtype=int)
    for i in range(1, n):
        pos[i] = max(0, pos[i-1] + z[i] - k)
        neg[i] = max(0, neg[i-1] - z[i] - k)
        if pos[i] > h or neg[i] > h:
            alarm[i] = 1
    return alarm

def sigma_drift(window_mean, baseline_mean, baseline_std, threshold=3.0):
    """Simple sigma-drift detector."""
    if baseline_std < 1e-6: return 0
    return 1 if abs(window_mean - baseline_mean) > threshold * baseline_std else 0

# CARE criticality algorithm (Algorithm 1 from paper)
def care_criticality(predictions, status, tc=72):
    """Exact CARE criticality algorithm.
    predictions: binary array (1=anomaly detected)
    status: status_id array (0=normal, else=abnormal) 
    tc: threshold (default 72 = 12 hours of consecutive anomalies)
    Returns: max_criticality, criticality_array
    """
    n = len(predictions)
    crit = np.zeros(n+1)
    for i in range(n):
        st = status[i]
        p = predictions[i]
        if st == 0:  # Normal status - scoring counts
            if p == 1:
                crit[i+1] = crit[i] + 1
            else:
                crit[i+1] = max(crit[i] - 1, 0)
        else:  # Abnormal status - ignore
            crit[i+1] = crit[i]
    crit = crit[1:]
    return np.max(crit), crit

# CARE F_beta score
def f_beta(tp, fp, fn, beta=0.5):
    if tp == 0: return 0.0
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)

# CARE Weighted Score for earliness
def weighted_score(predictions, event_start_idx, event_end_idx):
    """WS with linear weight function: 1.0 in first half, linear to 0 in second half."""
    if event_end_idx <= event_start_idx: return 0.0
    M = event_end_idx - event_start_idx
    total_w = 0.0; weighted_sum = 0.0
    for i in range(M):
        rel_pos = i / max(M-1, 1)  # 0 to 1
        if rel_pos <= 0.5:
            w = 1.0
        else:
            w = max(0, 2.0 * (1.0 - rel_pos))
        total_w += w
        if predictions[event_start_idx + i] == 1:
            weighted_sum += w
    return weighted_sum / total_w if total_w > 0 else 0.0

# ================================================================
# MAIN DETECTION PIPELINE (CARE-compliant)
# ================================================================
# Key sensors for multi-channel detection
CHANNEL_SENSORS = {
    "gearbox": ["sensor_186_avg"],
    "generator": ["sensor_173_avg"],
    "hydraulic": ["sensor_178_avg", "sensor_48_avg"],
    "pitch": ["sensor_62_avg", "sensor_12_avg", "sensor_13_avg", "sensor_14_avg"],
    "transformer": ["sensor_191_avg"],
    "cabinet": ["sensor_39_avg"],
    "bearing": ["sensor_194_avg"],
    "electrical": ["sensor_25_avg", "sensor_47_avg", "sensor_58_avg"],
    "converter": ["sensor_75_avg", "sensor_109_avg"],
    "oil_level": ["sensor_74_avg", "sensor_44_avg"],
}

ALL_DETECT_SENSORS = list(set(s for sl in CHANNEL_SENSORS.values() for s in sl))

results = []

for eid in all_ids:
    f1 = os.path.join(DATA, f"event_{eid}_part1.pkl")
    f2 = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if not os.path.exists(f1) or not os.path.exists(f2): continue
    
    df_train = pd.read_pickle(f1)  # Part 1 = training (normal behavior)
    df_pred = pd.read_pickle(f2)   # Part 2 = prediction (contains event)
    
    is_anomaly = eid in anomaly_ids
    label = "anomaly" if is_anomaly else "normal"
    
    # Get event timing (for anomaly datasets)
    event_start_idx = None; event_end_idx = None
    if is_anomaly:
        ev = ei[ei["event_id"]==eid].iloc[0]
        event_start_id = int(ev["event_start_id"]) if pd.notna(ev["event_start_id"]) else None
        event_end_id = int(ev["event_end_id"]) if pd.notna(ev["event_end_id"]) else None
        if event_start_id and "row_id" in df_pred.columns:
            mask_start = df_pred["row_id"] >= event_start_id
            event_start_idx = mask_start.idxmax() if mask_start.any() else None
            if event_start_idx is not None:
                event_start_idx = df_pred.index.get_loc(event_start_idx) if event_start_idx in df_pred.index else None
        if event_end_id and "row_id" in df_pred.columns:
            mask_end = df_pred["row_id"] >= event_end_id
            event_end_idx = mask_end.idxmax() if mask_end.any() else None
            if event_end_idx is not None:
                event_end_idx = df_pred.index.get_loc(event_end_idx) if event_end_idx in df_pred.index else None
    
    n_pred = len(df_pred)
    
    # Get status array
    if "status_id" in df_pred.columns:
        status = df_pred["status_id"].values
    else:
        status = np.zeros(n_pred)  # Assume normal if no status
    
    # ---- DETECTION: Multi-channel CUSUM + Sigma-drift ----
    # Train baseline from Part 1 (CARE-compliant)
    predictions = np.zeros(n_pred, dtype=int)
    channel_alarms = {}
    
    for ch_name, sensors in CHANNEL_SENSORS.items():
        ch_pred = np.zeros(n_pred, dtype=int)
        for s in sensors:
            if s not in df_train.columns or s not in df_pred.columns: continue
            train_data = df_train[s].dropna()
            if len(train_data) < 100: continue
            bl_mean = train_data.mean()
            bl_std = train_data.std()
            if bl_std < 1e-6: continue
            
            pred_data = df_pred[s].fillna(bl_mean).values
            
            # Method 1: CUSUM on raw timestamps
            cusum_alarm = cusum_detect(pred_data, bl_mean, bl_std, k=0.5, h=5.0)
            
            # Method 2: Sliding window sigma-drift (3-day window, 1-day step)
            W = 432  # 3 days
            S = 144  # 1 day step
            sigma_alarm = np.zeros(n_pred, dtype=int)
            for wi in range(0, n_pred - W + 1, S):
                wm = np.mean(pred_data[wi:wi+W])
                if abs(wm - bl_mean) > 3.0 * bl_std:
                    sigma_alarm[wi:wi+W] = 1
            
            # Combine: OR logic
            ch_pred = np.maximum(ch_pred, np.maximum(cusum_alarm, sigma_alarm))
        
        channel_alarms[ch_name] = ch_pred
        predictions = np.maximum(predictions, ch_pred)
    
    # ---- CARE SCORING ----
    # Filter by normal status (status_id == 0)
    normal_mask = (status == 0)
    
    if is_anomaly:
        # Pointwise F_0.5 (Coverage)
        gt = np.zeros(n_pred, dtype=int)
        if event_start_idx is not None:
            end_idx = event_end_idx if event_end_idx else n_pred
            gt[event_start_idx:end_idx] = 1
        
        # Only score normal-status points
        gt_filtered = gt[normal_mask]
        pred_filtered = predictions[normal_mask]
        
        tp = np.sum((gt_filtered == 1) & (pred_filtered == 1))
        fp = np.sum((gt_filtered == 0) & (pred_filtered == 1))
        fn = np.sum((gt_filtered == 1) & (pred_filtered == 0))
        tn = np.sum((gt_filtered == 0) & (pred_filtered == 0))
        
        coverage = f_beta(tp, fp, fn, beta=0.5)
        
        # Weighted score (Earliness)
        ws = 0.0
        if event_start_idx is not None:
            end_idx = event_end_idx if event_end_idx else n_pred
            ws = weighted_score(predictions, event_start_idx, end_idx)
        
        # Event-level: did we detect? (criticality >= tc)
        max_crit, crit_arr = care_criticality(predictions, status, tc=72)
        detected = max_crit >= 72
        
        # Lead time (from first detection to event_start)
        lead_days = None
        if detected and event_start_idx:
            first_alarm = np.argmax(crit_arr >= 72)
            if first_alarm < event_start_idx:
                lead_days = (event_start_idx - first_alarm) * 10 / (60 * 24)
        
        results.append({
            "event_id": eid, "label": label, "detected": detected,
            "max_crit": max_crit, "coverage_f05": coverage,
            "weighted_score": ws, "lead_days": lead_days,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_channels": sum(1 for v in channel_alarms.values() if np.sum(v) > 0),
            "pct_normal_status": np.mean(normal_mask)*100,
        })
    else:
        # Normal dataset: Accuracy = TN/(FP+TN)
        pred_filtered = predictions[normal_mask]
        fp = np.sum(pred_filtered == 1)
        tn = np.sum(pred_filtered == 0)
        acc = tn / (fp + tn) if (fp + tn) > 0 else 1.0
        
        max_crit, _ = care_criticality(predictions, status, tc=72)
        false_alarm = max_crit >= 72
        
        results.append({
            "event_id": eid, "label": label, "detected": False,
            "max_crit": max_crit, "accuracy": acc,
            "false_alarm": false_alarm,
            "fp": fp, "tn": tn,
            "n_channels": sum(1 for v in channel_alarms.values() if np.sum(v) > 0),
            "pct_normal_status": np.mean(normal_mask)*100,
        })

# ================================================================
# RESULTS
# ================================================================
df_res = pd.DataFrame(results)
df_anom = df_res[df_res["label"]=="anomaly"]
df_norm = df_res[df_res["label"]=="normal"]

print("\n" + "="*100)
print("CARE-COMPLIANT RESULTS")
print("="*100)

# Coverage
tp_events = df_anom["detected"].sum()
fn_events = len(df_anom) - tp_events
print(f"\n--- ANOMALY DATASETS ({len(df_anom)}) ---")
print(f"Events detected (crit>=72): {tp_events}/{len(df_anom)} = {tp_events/len(df_anom)*100:.1f}%")
print(f"Events missed: {fn_events}")
if fn_events > 0:
    missed = df_anom[~df_anom["detected"]]["event_id"].tolist()
    print(f"Missed event IDs: {missed}")
    for _, r in df_anom[~df_anom["detected"]].iterrows():
        print(f"  Event {r['event_id']}: max_crit={r['max_crit']:.0f} (need 72)")

# Mean F_0.5 Coverage
mean_cov = df_anom["coverage_f05"].mean()
print(f"\nMean Coverage (F_0.5): {mean_cov:.4f}")

# Mean WS (Earliness)
mean_ws = df_anom["weighted_score"].mean()
print(f"Mean Weighted Score (Earliness): {mean_ws:.4f}")

# Lead times
leads = df_anom.dropna(subset=["lead_days"])
if len(leads) > 0:
    print(f"Mean lead time (detection → event_start): {leads['lead_days'].mean():.1f} days")
    print(f"Median lead time: {leads['lead_days'].median():.1f} days")

# Normal datasets
print(f"\n--- NORMAL DATASETS ({len(df_norm)}) ---")
fp_events = df_norm["false_alarm"].sum()
tn_events = len(df_norm) - fp_events
print(f"False alarms (crit>=72): {fp_events}/{len(df_norm)}")
print(f"Correctly quiet: {tn_events}/{len(df_norm)}")
mean_acc = df_norm["accuracy"].mean()
print(f"Mean Accuracy (TN/(FP+TN)): {mean_acc:.4f}")

if fp_events > 0:
    print(f"False alarm event IDs: {df_norm[df_norm['false_alarm']]['event_id'].tolist()}")

# CARE composite score
print(f"\n--- CARE SCORE ---")
# Reliability: event-based F_0.5
ef_tp = tp_events
ef_fp = fp_events
ef_fn = fn_events
reliability = f_beta(ef_tp, ef_fp, ef_fn, beta=0.5)

# Final CARE (if any detected and Acc >= 0.5)
if tp_events > 0 and mean_acc >= 0.5:
    CARE = (mean_cov + mean_acc + reliability + mean_ws) / 4
else:
    CARE = 0.0

print(f"  C (Coverage F_0.5):     {mean_cov:.4f}")
print(f"  A (Accuracy):           {mean_acc:.4f}")
print(f"  R (Reliability EF_0.5): {reliability:.4f}")
print(f"  E (Earliness WS):       {mean_ws:.4f}")
print(f"  ─────────────────────")
print(f"  CARE Score:             {CARE:.4f}")

# Per-event details
print(f"\n--- PER-EVENT DETAIL (Anomaly) ---")
print(f"{'Event':>6} {'Detected':>10} {'MaxCrit':>8} {'CovF05':>8} {'WS':>6} {'Lead':>8} {'Chs':>4} {'FP':>5} {'FN':>6}")
for _, r in df_anom.sort_values("event_id").iterrows():
    det = "YES" if r["detected"] else "NO"
    lead = f"{r['lead_days']:.0f}d" if pd.notna(r.get("lead_days")) else "-"
    print(f"{int(r['event_id']):>6} {det:>10} {r['max_crit']:>8.0f} {r['coverage_f05']:>8.4f} {r.get('weighted_score',0):>6.3f} {lead:>8} {r['n_channels']:>4} {r['fp']:>5} {r.get('fn',0):>6}")

# Save
df_res.to_csv(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\care_v2_results.csv", index=False)
print(f"\nResults saved to results/care_v2_results.csv")
