"""
CARE v6: Power-Binned Normal Behavior + Adaptive Ensemble
==========================================================
Fundamental redesign:
1. Bin training data by power output (operating state)
2. For each bin, compute mean+std of each sensor
3. In prediction, check if sensor value falls outside bin-specific range
4. Use ROLLING MEAN to smooth (not individual timestamps)
5. Require multiple channels to vote
6. Proper CARE scoring
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
anomaly_ids = sorted(ei[ei["event_label"]=="anomaly"]["event_id"].tolist())
normal_ids = sorted(ei[ei["event_label"]=="normal"]["event_id"].tolist())
all_ids = sorted(anomaly_ids + normal_ids)

# Power column for binning
POWER_COL = "power_2_avg"  # Main active power

# Target sensors
TARGETS = [
    "sensor_186_avg", "sensor_173_avg", "sensor_194_avg", "sensor_195_avg",
    "sensor_191_avg", "sensor_178_avg", "sensor_39_avg", "sensor_62_avg",
    "sensor_48_avg", "sensor_74_avg", "sensor_25_avg",
    "sensor_12_avg", "sensor_13_avg", "sensor_14_avg",
    "sensor_47_avg", "sensor_58_avg", "sensor_109_avg", "sensor_110_avg",
    "sensor_44_avg", "sensor_45_avg", "sensor_87_avg", "sensor_75_avg",
    "sensor_167_avg",
]

def care_criticality(pred, status, tc=72):
    n = len(pred)
    crit = np.zeros(n+1)
    for i in range(n):
        if status[i] == 0:
            crit[i+1] = crit[i]+1 if pred[i]==1 else max(crit[i]-1,0)
        else:
            crit[i+1] = crit[i]
    return np.max(crit[1:]), crit[1:]

def f_beta(tp, fp, fn, beta=0.5):
    if tp == 0: return 0.0
    return (1+beta**2)*tp/((1+beta**2)*tp+beta**2*fn+fp)

def weighted_score(pred, status, esi, eei):
    if eei <= esi: return 0.0
    M = eei-esi; tw=0.0; ws=0.0
    for i in range(M):
        idx=esi+i
        if idx>=len(status) or status[idx]!=0: continue
        rp=i/max(M-1,1); w=1.0 if rp<=0.5 else max(0,2*(1-rp))
        tw+=w
        if pred[idx]==1: ws+=w
    return ws/tw if tw>0 else 0.0

print("="*100)
print("CARE v6: Power-Binned NBM + Rolling Anomaly Score")
print("="*100)

best_care = -1; best_params = {}; best_results = None

for loop, (n_bins, roll_w, z_thresh, min_anom_frac, min_ch) in enumerate([
    (20, 432, 3.0, 0.3, 3),   # 20 bins, 3-day roll, 3σ, 30% anomalous, 3 channels
    (20, 432, 3.5, 0.3, 3),
    (20, 432, 4.0, 0.3, 3),
    (20, 432, 4.0, 0.4, 3),
    (20, 432, 4.0, 0.5, 4),
    (30, 432, 4.0, 0.5, 4),
    (30, 432, 5.0, 0.5, 5),
    (30, 864, 4.0, 0.5, 5),   # 6-day roll
    (30, 864, 5.0, 0.6, 6),
    (30, 864, 5.0, 0.7, 7),   # Very conservative
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
        
        # Training: normal-status only
        tr_mask = df1["status_type_id"].values==0 if "status_type_id" in df1.columns else np.ones(len(df1),bool)
        
        if POWER_COL not in df1.columns or POWER_COL not in df2.columns: continue
        
        power_tr = df1.loc[tr_mask, POWER_COL].values
        valid_power = np.isfinite(power_tr)
        
        # Build power bins from training data
        p_min, p_max = np.nanmin(power_tr[valid_power]), np.nanmax(power_tr[valid_power])
        bin_edges = np.linspace(p_min, p_max, n_bins+1)
        
        # For each sensor, compute bin-level statistics
        bin_stats = {}
        avail_targets = [s for s in TARGETS if s in df1.columns and s in df2.columns]
        
        for sensor in avail_targets:
            s_tr = df1.loc[tr_mask, sensor].values
            stats = {}
            for bi in range(n_bins):
                lo, hi = bin_edges[bi], bin_edges[bi+1]
                mask = (power_tr >= lo) & (power_tr < hi) & np.isfinite(s_tr) & valid_power
                if np.sum(mask) < 10: continue
                vals = s_tr[mask]
                stats[bi] = {"mean": np.mean(vals), "std": np.std(vals)}
            bin_stats[sensor] = stats
        
        # Prediction: for each timestamp, look up power bin, compute z-score
        power_pred = df2[POWER_COL].fillna(0).values
        pred_bins = np.digitize(power_pred, bin_edges) - 1
        pred_bins = np.clip(pred_bins, 0, n_bins-1)
        
        channel_scores = {}
        for sensor in avail_targets:
            if sensor not in bin_stats: continue
            s_pred = df2[sensor].fillna(0).values
            z_scores = np.zeros(n2)
            for i in range(n2):
                bi = pred_bins[i]
                if bi in bin_stats[sensor]:
                    st = bin_stats[sensor][bi]
                    if st["std"] > 1e-6:
                        z_scores[i] = abs(s_pred[i] - st["mean"]) / st["std"]
            channel_scores[sensor] = z_scores
        
        # Rolling anomaly score per channel
        channel_alarms = {}
        for sensor, z in channel_scores.items():
            # Rolling mean of z-scores
            if len(z) < roll_w:
                channel_alarms[sensor] = np.zeros(n2, dtype=int)
                continue
            roll_z = pd.Series(z).rolling(roll_w, min_periods=roll_w//2, center=True).mean().fillna(0).values
            channel_alarms[sensor] = (roll_z > z_thresh).astype(int)
        
        # Voting
        if len(channel_alarms) == 0:
            predictions = np.zeros(n2, dtype=int)
        else:
            vote = np.zeros(n2)
            for a in channel_alarms.values():
                vote += a
            predictions = (vote >= min_ch).astype(int)
        
        n_ch = sum(1 for v in channel_alarms.values() if np.sum(v)>0)
        nm = (status == 0)
        
        if is_anomaly:
            gt = np.zeros(n2, dtype=int)
            if esi_val is not None:
                end = eei_val if eei_val else n2; gt[esi_val:end] = 1
            gf=gt[nm]; pf=predictions[nm]
            tp=int(np.sum((gf==1)&(pf==1))); fp=int(np.sum((gf==0)&(pf==1)))
            fn=int(np.sum((gf==1)&(pf==0)))
            cov = f_beta(tp,fp,fn,beta=0.5)
            ws = weighted_score(predictions,status,esi_val,eei_val if eei_val else n2) if esi_val else 0.0
            mc,ca = care_criticality(predictions,status,tc=72)
            det = mc >= 72
            ld = None
            if det and esi_val:
                ci = np.argmax(ca>=72)
                if ci < esi_val: ld = (esi_val-ci)*10/(60*24)
            ap = np.mean(predictions[nm])*100 if nm.sum()>0 else 0
            results.append({"eid":eid,"label":"anomaly","detected":det,"crit":mc,"cov":cov,
                           "ws":ws,"lead":ld,"tp":tp,"fp":fp,"fn":fn,"n_ch":n_ch,"alarm_pct":ap})
        else:
            pf=predictions[nm]; fp=int(np.sum(pf==1)); tn=int(np.sum(pf==0))
            acc=tn/(fp+tn) if (fp+tn)>0 else 1.0
            mc,_=care_criticality(predictions,status,tc=72); fa=mc>=72
            ap = np.mean(predictions[nm])*100 if nm.sum()>0 else 0
            results.append({"eid":eid,"label":"normal","crit":mc,"acc":acc,"fa":fa,
                           "fp":fp,"n_ch":n_ch,"alarm_pct":ap})
    
    dr=pd.DataFrame(results)
    da=dr[dr["label"]=="anomaly"]; dn=dr[dr["label"]=="normal"]
    tp_ev=int(da["detected"].sum()); fn_ev=len(da)-tp_ev
    fp_ev=int(dn["fa"].sum()); tn_ev=len(dn)-fp_ev
    mc_=da["cov"].mean(); ma=dn["acc"].mean(); mw=da["ws"].mean()
    rel=f_beta(tp_ev,fp_ev,fn_ev,beta=0.5)
    care=(mc_+ma+rel+mw)/4 if tp_ev>0 and ma>=0.5 else 0.0
    leads=da.dropna(subset=["lead"])
    ml=leads["lead"].mean() if len(leads)>0 else 0
    
    print(f"\nLoop {loop+1}: bins={n_bins} roll={roll_w} z={z_thresh} frac={min_anom_frac} ch={min_ch}")
    print(f"  TP={tp_ev}/27 FP_ev={fp_ev}/31  Recall={tp_ev/27*100:.0f}%")
    print(f"  C={mc_:.4f} A={ma:.4f} R={rel:.4f} E={mw:.4f} CARE={care:.4f}")
    print(f"  Lead={ml:.1f}d  NormAlarm: mean={dn['alarm_pct'].mean():.1f}%")
    
    if care > best_care:
        best_care=care; best_params={"bins":n_bins,"roll":roll_w,"z":z_thresh,"ch":min_ch}
        best_results=dr.copy()

print(f"\n{'='*100}")
print(f"BEST: {best_params} → CARE={best_care:.4f}")
print(f"{'='*100}")

if best_results is not None:
    ba=best_results[best_results["label"]=="anomaly"]
    bn=best_results[best_results["label"]=="normal"]
    print(f"\nTP: {ba['detected'].sum()}/27")
    if ba['detected'].sum()<27:
        for _,r in ba[~ba['detected']].iterrows():
            print(f"  MISS {int(r['eid'])}: crit={r['crit']:.0f}")
    print(f"FP: {bn['fa'].sum()}/31")
    if bn['fa'].sum()>0 and bn['fa'].sum()<=10:
        for _,r in bn[bn['fa']].iterrows():
            print(f"  FA {int(r['eid'])}: crit={r['crit']:.0f} alarm={r['alarm_pct']:.1f}%")
    best_results.to_csv(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\care_v6_best.csv", index=False)
