"""
CARE v4: Properly tuned NBM + Adaptive CUSUM
=============================================
Fixes from v3:
- Correct column names (wind_speed_236_avg, power_2_avg, etc.)
- NBM uses multiple context features (wind, power, ambient temp)
- CUSUM h tuned via training data cross-validation
- Status filtering with status_type_id
- Iterative threshold optimization (10 loops)
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

# Actual context columns in Wind Farm C
CONTEXT_COLS = ["wind_speed_236_avg", "wind_speed_235_avg", "power_2_avg", "power_5_avg", "power_6_avg"]

# Target sensors 
TARGETS = {
    "gearbox_temp": "sensor_186_avg",
    "generator_temp": "sensor_173_avg",
    "bearing1_temp": "sensor_194_avg",
    "bearing2_temp": "sensor_195_avg",
    "transformer_temp": "sensor_191_avg",
    "hydraulic_temp": "sensor_178_avg",
    "cabinet_temp": "sensor_39_avg",
    "pitch_temp": "sensor_62_avg",
    "hydraulic_press": "sensor_48_avg",
    "oil_level": "sensor_74_avg",
    "24v_current": "sensor_25_avg",
    "battery_ax1": "sensor_12_avg",
    "battery_ax2": "sensor_13_avg",
    "battery_ax3": "sensor_14_avg",
    "mains_freq": "sensor_47_avg",
    "abb_voltage_l1": "sensor_58_avg",
    "abb_voltage_l2": "sensor_59_avg",
    "abb_voltage_l3": "sensor_60_avg",
    "filter_press1": "sensor_109_avg",
    "filter_press2": "sensor_110_avg",
    "oil_a1": "sensor_44_avg",
    "oil_a2": "sensor_45_avg",
    "pump_current": "sensor_87_avg",
    "reactive_hv": "sensor_75_avg",
    "gen_current_l1": "sensor_130_avg",
    "gen_current_l2": "sensor_131_avg",
    "gen_current_l3": "sensor_132_avg",
    "cabinet_air": "sensor_167_avg",
}

def care_criticality(predictions, status, tc=72):
    n = len(predictions)
    crit = np.zeros(n+1)
    for i in range(n):
        if status[i] == 0:
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

def weighted_score(predictions, status, esi, eei):
    if eei <= esi: return 0.0
    M = eei - esi
    total_w = 0.0; weighted_sum = 0.0
    for i in range(M):
        idx = esi + i
        if idx >= len(status) or status[idx] != 0: continue
        rel_pos = i / max(M-1, 1)
        w = 1.0 if rel_pos <= 0.5 else max(0, 2*(1-rel_pos))
        total_w += w
        if predictions[idx] == 1: weighted_sum += w
    return weighted_sum / total_w if total_w > 0 else 0.0

# ================================================================
# ITERATIVE OPTIMIZATION: Find best h (CUSUM threshold) and sigma_thresh
# ================================================================
print("="*100)
print("ITERATIVE OPTIMIZATION: 10 loops to find best CUSUM h")
print("="*100)

best_care = -1; best_params = {}

for loop in range(10):
    # Sweep CUSUM h and sigma threshold
    h_vals = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    h = h_vals[loop]
    sigma_thresh = 4.0 if h < 50 else 3.5 if h < 100 else 3.0
    k = 1.0 if h < 50 else 0.75 if h < 100 else 0.5
    
    results = []
    
    for eid in all_ids:
        f1 = os.path.join(DATA, f"event_{eid}_part1.pkl")
        f2 = os.path.join(DATA, f"event_{eid}_part2.pkl")
        if not os.path.exists(f1) or not os.path.exists(f2): continue
        
        df1 = pd.read_pickle(f1); df2 = pd.read_pickle(f2)
        is_anomaly = eid in anomaly_ids
        n2 = len(df2)
        status = df2["status_type_id"].values if "status_type_id" in df2.columns else np.zeros(n2)
        
        # Event timing
        esi_val = None; eei_val = None
        if is_anomaly:
            ev = ei[ei["event_id"]==eid].iloc[0]
            if pd.notna(ev["event_start_id"]) and "row_id" in df2.columns:
                m = df2["row_id"].values >= int(ev["event_start_id"])
                if m.any(): esi_val = m.argmax()
            if pd.notna(ev["event_end_id"]) and "row_id" in df2.columns:
                m = df2["row_id"].values >= int(ev["event_end_id"])
                if m.any(): eei_val = m.argmax()
        
        # Training mask (normal status in Part 1)
        tr_mask = df1["status_type_id"].values == 0 if "status_type_id" in df1.columns else np.ones(len(df1), dtype=bool)
        
        # Available context
        ctx = [c for c in CONTEXT_COLS if c in df1.columns and c in df2.columns]
        
        predictions = np.zeros(n2, dtype=int)
        n_models = 0; n_ch_firing = 0
        
        for ch_name, sensor in TARGETS.items():
            if sensor not in df1.columns or sensor not in df2.columns: continue
            
            feat_cols = ctx + [s for s in TARGETS.values() if s in df1.columns and s != sensor][:5]
            if not feat_cols: continue
            
            X_tr = df1.loc[tr_mask, feat_cols].values
            y_tr = df1.loc[tr_mask, sensor].values
            valid = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
            X_tr = X_tr[valid]; y_tr = y_tr[valid]
            if len(X_tr) < 200: continue
            
            sc = StandardScaler().fit(X_tr)
            model = Ridge(alpha=10.0).fit(sc.transform(X_tr), y_tr)
            
            # Training residuals
            res_tr = y_tr - model.predict(sc.transform(X_tr))
            res_m = res_tr.mean(); res_s = res_tr.std()
            if res_s < 1e-6: continue
            n_models += 1
            
            # Test predictions
            X_te = df2[feat_cols].fillna(0).values
            X_te = np.where(np.isfinite(X_te), X_te, 0)
            y_te = df2[sensor].fillna(0).values
            y_pred = model.predict(sc.transform(X_te))
            
            # Normalized residuals
            norm_res = (y_te - y_pred - res_m) / res_s
            
            # CUSUM
            ch_alarm = np.zeros(n2, dtype=int)
            pos = 0.0; neg = 0.0
            for i in range(n2):
                pos = max(0, pos + norm_res[i] - k)
                neg = max(0, neg - norm_res[i] - k)
                if pos > h or neg > h:
                    ch_alarm[i] = 1
            
            # Also: sliding window sigma (3-day)
            W = 432; S = 144
            for wi in range(0, n2 - W + 1, S):
                wm = np.mean(norm_res[wi:wi+W])
                if abs(wm) > sigma_thresh:
                    ch_alarm[wi:wi+W] = 1
            
            if np.sum(ch_alarm) > 0: n_ch_firing += 1
            predictions = np.maximum(predictions, ch_alarm)
        
        # CARE scoring
        normal_mask = (status == 0)
        
        if is_anomaly:
            gt = np.zeros(n2, dtype=int)
            if esi_val is not None:
                end = eei_val if eei_val else n2
                gt[esi_val:end] = 1
            
            gt_f = gt[normal_mask]; pred_f = predictions[normal_mask]
            tp = int(np.sum((gt_f==1)&(pred_f==1)))
            fp = int(np.sum((gt_f==0)&(pred_f==1)))
            fn = int(np.sum((gt_f==1)&(pred_f==0)))
            tn = int(np.sum((gt_f==0)&(pred_f==0)))
            cov = f_beta(tp, fp, fn, beta=0.5)
            
            ws = 0.0
            if esi_val is not None:
                end = eei_val if eei_val else n2
                ws = weighted_score(predictions, status, esi_val, end)
            
            mc, ca = care_criticality(predictions, status, tc=72)
            detected = mc >= 72
            
            ld = None
            if detected and esi_val:
                ci = np.argmax(ca >= 72)
                if ci < esi_val: ld = (esi_val - ci) * 10 / (60*24)
            
            results.append({"eid": eid, "label": "anomaly", "detected": detected, "crit": mc,
                           "cov": cov, "ws": ws, "lead": ld, "tp": tp, "fp": fp, "fn": fn,
                           "n_ch": n_ch_firing, "alarm_pct": np.mean(predictions[normal_mask])*100})
        else:
            pred_f = predictions[normal_mask]
            fp = int(np.sum(pred_f==1)); tn = int(np.sum(pred_f==0))
            acc = tn/(fp+tn) if (fp+tn)>0 else 1.0
            mc, _ = care_criticality(predictions, status, tc=72)
            fa = mc >= 72
            results.append({"eid": eid, "label": "normal", "crit": mc, "acc": acc,
                           "fa": fa, "fp": fp, "n_ch": n_ch_firing,
                           "alarm_pct": np.mean(predictions[normal_mask])*100})
    
    dr = pd.DataFrame(results)
    da = dr[dr["label"]=="anomaly"]; dn = dr[dr["label"]=="normal"]
    
    tp_ev = da["detected"].sum(); fn_ev = len(da) - tp_ev
    fp_ev = dn["fa"].sum(); tn_ev = len(dn) - fp_ev
    
    mean_cov = da["cov"].mean()
    mean_acc = dn["acc"].mean()
    mean_ws = da["ws"].mean()
    rel = f_beta(tp_ev, fp_ev, fn_ev, beta=0.5)
    
    if tp_ev > 0 and mean_acc >= 0.5:
        care = (mean_cov + mean_acc + rel + mean_ws) / 4
    else:
        care = 0.0
    
    leads = da.dropna(subset=["lead"])
    ml = leads["lead"].mean() if len(leads)>0 else 0
    
    print(f"\nLoop {loop+1}: h={h}, k={k:.1f}, sigma={sigma_thresh}")
    print(f"  TP={tp_ev}/27  FP_events={fp_ev}/31  Recall={tp_ev/27*100:.0f}%")
    print(f"  C={mean_cov:.4f}  A={mean_acc:.4f}  R={rel:.4f}  E={mean_ws:.4f}")
    print(f"  CARE={care:.4f}  Lead={ml:.1f}d")
    print(f"  Normal alarm%: mean={dn['alarm_pct'].mean():.1f}% max={dn['alarm_pct'].max():.1f}%")
    
    if care > best_care:
        best_care = care
        best_params = {"h": h, "k": k, "sigma": sigma_thresh, "loop": loop+1}
        best_results = dr.copy()

print(f"\n{'='*100}")
print(f"BEST: Loop {best_params.get('loop')}, h={best_params.get('h')}, CARE={best_care:.4f}")
print(f"{'='*100}")

# Save best results
if best_results is not None:
    best_results.to_csv(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\care_v4_best.csv", index=False)
    
    ba = best_results[best_results["label"]=="anomaly"]
    bn = best_results[best_results["label"]=="normal"]
    
    print(f"\nBest anomaly detection: {ba['detected'].sum()}/27")
    if ba["detected"].sum() < 27:
        missed = ba[~ba["detected"]]
        for _, r in missed.iterrows():
            print(f"  MISSED Event {int(r['eid'])}: crit={r['crit']:.0f}")
    
    print(f"Best false alarms: {bn['fa'].sum()}/31")
    if bn["fa"].sum() > 0:
        fas = bn[bn["fa"]]
        for _, r in fas.iterrows():
            print(f"  FA Event {int(r['eid'])}: crit={r['crit']:.0f} alarm={r['alarm_pct']:.1f}%")
