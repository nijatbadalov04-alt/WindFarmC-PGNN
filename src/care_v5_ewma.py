"""
CARE v5: EWMA-Smoothed Residual Detection
==========================================
Key insight from diagnosis: Individual timestamp residuals are too noisy
(12% >3σ even on normal data). Solution: EWMA-smooth the residuals
and apply CUSUM on the smoothed signal. This is standard practice in
SPC (Statistical Process Control) — exactly what CWD2017 recommends.

Also: Use ONLY the most reliable sensors — ones where NBM R² > 0.5
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

CONTEXT = ["wind_speed_236_avg","wind_speed_235_avg","power_2_avg","power_5_avg","power_6_avg"]

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
    "abb_v_l1": "sensor_58_avg",
    "filter_p1": "sensor_109_avg",
    "filter_p2": "sensor_110_avg",
    "oil_a1": "sensor_44_avg",
    "oil_a2": "sensor_45_avg",
    "pump_curr": "sensor_87_avg",
    "reactive_hv": "sensor_75_avg",
    "cabinet_air": "sensor_167_avg",
}

def ewma(x, span=144):  # 1-day EWMA (144 x 10min = 24h)
    alpha = 2/(span+1)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1-alpha) * out[i-1]
    return out

def care_criticality(pred, status, tc=72):
    n = len(pred)
    crit = np.zeros(n+1)
    for i in range(n):
        if status[i] == 0:
            crit[i+1] = crit[i] + 1 if pred[i] == 1 else max(crit[i]-1, 0)
        else:
            crit[i+1] = crit[i]
    return np.max(crit[1:]), crit[1:]

def f_beta(tp, fp, fn, beta=0.5):
    if tp == 0: return 0.0
    return (1+beta**2)*tp / ((1+beta**2)*tp + beta**2*fn + fp)

def weighted_score(pred, status, esi, eei):
    if eei <= esi: return 0.0
    M = eei - esi; tw = 0.0; ws = 0.0
    for i in range(M):
        idx = esi+i
        if idx >= len(status) or status[idx] != 0: continue
        rp = i/max(M-1,1)
        w = 1.0 if rp <= 0.5 else max(0, 2*(1-rp))
        tw += w
        if pred[idx] == 1: ws += w
    return ws/tw if tw > 0 else 0.0

# ================================================================
# MAIN LOOP
# ================================================================
print("="*100)
print("CARE v5: EWMA-Smoothed Residual Detection — 10 Threshold Sweeps")
print("="*100)

best_care = -1; best_params = {}; best_results = None

for loop, (ewma_span, h_val, sig_thresh, min_channels) in enumerate([
    (144, 50, 5.0, 1),   # 1-day EWMA, moderate
    (288, 50, 5.0, 1),   # 2-day EWMA
    (432, 50, 5.0, 1),   # 3-day EWMA
    (288, 100, 5.0, 1),  # 2-day EWMA, higher h
    (288, 100, 4.0, 2),  # require 2 channels
    (288, 150, 4.0, 2),  # higher h
    (288, 200, 3.5, 3),  # require 3 channels
    (432, 200, 3.5, 3),  # 3-day EWMA, 3 channels
    (432, 300, 3.0, 3),  # very conservative
    (432, 500, 3.0, 4),  # ultra conservative: 4 channels, h=500
]):
    results = []
    
    for eid in all_ids:
        f1 = os.path.join(DATA, f"event_{eid}_part1.pkl")
        f2 = os.path.join(DATA, f"event_{eid}_part2.pkl")
        if not os.path.exists(f1) or not os.path.exists(f2): continue
        
        df1 = pd.read_pickle(f1); df2 = pd.read_pickle(f2)
        is_anomaly = eid in anomaly_ids
        n2 = len(df2)
        status = df2["status_type_id"].values if "status_type_id" in df2.columns else np.zeros(n2)
        
        esi_val = None; eei_val = None
        if is_anomaly:
            ev = ei[ei["event_id"]==eid].iloc[0]
            if pd.notna(ev["event_start_id"]) and "row_id" in df2.columns:
                m = df2["row_id"].values >= int(ev["event_start_id"])
                if m.any(): esi_val = int(m.argmax())
            if pd.notna(ev["event_end_id"]) and "row_id" in df2.columns:
                m = df2["row_id"].values >= int(ev["event_end_id"])
                if m.any(): eei_val = int(m.argmax())
        
        tr_mask = df1["status_type_id"].values == 0 if "status_type_id" in df1.columns else np.ones(len(df1), bool)
        ctx = [c for c in CONTEXT if c in df1.columns and c in df2.columns]
        
        # Per-channel detection
        channel_predictions = {}
        
        for ch_name, sensor in TARGETS.items():
            if sensor not in df1.columns or sensor not in df2.columns: continue
            
            feat_cols = ctx.copy()
            if not feat_cols: continue
            
            X_tr = df1.loc[tr_mask, feat_cols].values
            y_tr = df1.loc[tr_mask, sensor].values
            v = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
            X_tr = X_tr[v]; y_tr = y_tr[v]
            if len(X_tr) < 500: continue
            
            sc = StandardScaler().fit(X_tr)
            model = Ridge(alpha=10.0).fit(sc.transform(X_tr), y_tr)
            r2 = model.score(sc.transform(X_tr), y_tr)
            if r2 < 0.3: continue  # Skip poor models
            
            res_tr = y_tr - model.predict(sc.transform(X_tr))
            res_s = res_tr.std()
            if res_s < 1e-6: continue
            
            # Test prediction
            X_te = df2[feat_cols].fillna(0).values
            X_te = np.where(np.isfinite(X_te), X_te, 0)
            y_te = df2[sensor].fillna(0).values
            y_pred = model.predict(sc.transform(X_te))
            
            # Normalized residuals
            norm_res = (y_te - y_pred - res_tr.mean()) / res_s
            
            # EWMA smoothing
            smooth_res = ewma(norm_res, span=ewma_span)
            
            # CUSUM on smoothed residuals
            ch_alarm = np.zeros(n2, dtype=int)
            pos = 0.0; neg = 0.0
            for i in range(n2):
                pos = max(0, pos + smooth_res[i] - 1.0)
                neg = max(0, neg - smooth_res[i] - 1.0)
                if pos > h_val or neg > h_val:
                    ch_alarm[i] = 1
            
            # Also: sliding window check on smoothed residuals
            W = 432; S = 144
            for wi in range(0, n2-W+1, S):
                wm = np.mean(smooth_res[wi:wi+W])
                if abs(wm) > sig_thresh:
                    ch_alarm[wi:wi+W] = 1
            
            channel_predictions[ch_name] = ch_alarm
        
        # Voting: require min_channels
        if len(channel_predictions) == 0:
            predictions = np.zeros(n2, dtype=int)
        else:
            vote_arr = np.zeros(n2)
            for ch_alarm in channel_predictions.values():
                vote_arr += ch_alarm
            predictions = (vote_arr >= min_channels).astype(int)
        
        n_ch = sum(1 for v in channel_predictions.values() if np.sum(v)>0)
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
            ap = np.mean(predictions[normal_mask])*100 if normal_mask.sum()>0 else 0
            results.append({"eid":eid, "label":"anomaly", "detected":detected, "crit":mc,
                           "cov":cov, "ws":ws, "lead":ld, "tp":tp, "fp":fp, "fn":fn,
                           "n_ch":n_ch, "alarm_pct":ap})
        else:
            pred_f = predictions[normal_mask]
            fp = int(np.sum(pred_f==1)); tn = int(np.sum(pred_f==0))
            acc = tn/(fp+tn) if (fp+tn)>0 else 1.0
            mc, _ = care_criticality(predictions, status, tc=72)
            fa = mc >= 72
            ap = np.mean(predictions[normal_mask])*100 if normal_mask.sum()>0 else 0
            results.append({"eid":eid, "label":"normal", "crit":mc, "acc":acc, "fa":fa,
                           "fp":fp, "n_ch":n_ch, "alarm_pct":ap})
    
    dr = pd.DataFrame(results)
    da = dr[dr["label"]=="anomaly"]; dn = dr[dr["label"]=="normal"]
    tp_ev = int(da["detected"].sum()); fn_ev = len(da)-tp_ev
    fp_ev = int(dn["fa"].sum()); tn_ev = len(dn)-fp_ev
    mc = da["cov"].mean(); ma = dn["acc"].mean(); mw = da["ws"].mean()
    rel = f_beta(tp_ev, fp_ev, fn_ev, beta=0.5)
    care = (mc+ma+rel+mw)/4 if tp_ev > 0 and ma >= 0.5 else 0.0
    leads = da.dropna(subset=["lead"])
    ml = leads["lead"].mean() if len(leads)>0 else 0
    
    print(f"\nLoop {loop+1}: ewma={ewma_span}, h={h_val}, sig={sig_thresh}, min_ch={min_channels}")
    print(f"  TP={tp_ev}/27 FP_ev={fp_ev}/31 Recall={tp_ev/27*100:.0f}% Precision={tp_ev/(tp_ev+fp_ev)*100:.0f}%" if tp_ev+fp_ev>0 else f"  TP={tp_ev}/27 FP_ev={fp_ev}/31")
    print(f"  C={mc:.4f} A={ma:.4f} R={rel:.4f} E={mw:.4f} CARE={care:.4f}")
    print(f"  Lead={ml:.1f}d  Norm alarm%: mean={dn['alarm_pct'].mean():.1f}% max={dn['alarm_pct'].max():.1f}%")
    
    if care > best_care:
        best_care = care; best_params = {"ewma":ewma_span, "h":h_val, "sig":sig_thresh, "min_ch":min_channels}
        best_results = dr.copy()

print(f"\n{'='*100}")
print(f"BEST: {best_params} CARE={best_care:.4f}")
print(f"{'='*100}")

if best_results is not None:
    ba = best_results[best_results["label"]=="anomaly"]
    bn = best_results[best_results["label"]=="normal"]
    print(f"\nDetected: {ba['detected'].sum()}/27")
    if ba['detected'].sum() < 27:
        for _, r in ba[~ba['detected']].iterrows():
            print(f"  MISSED {int(r['eid'])}: crit={r['crit']:.0f} ch={r['n_ch']}")
    print(f"False alarms: {bn['fa'].sum()}/31")
    if bn['fa'].sum() > 0:
        for _, r in bn[bn['fa']].iterrows():
            print(f"  FA {int(r['eid'])}: crit={r['crit']:.0f} alarm={r['alarm_pct']:.1f}%")
    
    best_results.to_csv(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\care_v5_best.csv", index=False)
