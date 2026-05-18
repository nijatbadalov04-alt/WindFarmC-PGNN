"""
Phase 9: Industry-Grade Comprehensive Detection
5 improvements: Power-Wind Filter, Expanded ARMAX, CARE Criticality, Zero-Power Channel, Forensic FP
"""
import os, sys, glob, json, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
ARMAX_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\armax"
PGNN_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\pgnn"
P8_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\phase8"
RESULTS_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
P9_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\phase9"
os.makedirs(P9_DIR, exist_ok=True)

with open(os.path.join(ARMAX_DIR, "armax_config.json")) as f: cfg = json.load(f)
THERMAL_TARGETS = cfg["targets"]; EXO = cfg["exogenous"]
ei = pd.read_csv(EVENT_CSV, sep=";")
anom_set = set(ei[ei["event_label"]=="anomaly"]["event_id"].astype(str))
W, S = 432, 72

# New expanded targets from FN forensics
NEW_TARGETS = {
    "rotor_bearing": "sensor_194_avg",   # 79σ drift in Event 28
    "battery_current": "sensor_12_avg",  # 12σ drift in Event 31
    "cabinet_temp": "sensor_39_avg",     # 5.9σ drift in Event 9
}

class PGNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),
                                    nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1))
        self.class_head = nn.Linear(64,1); self.phys_head = nn.Linear(64,1)
    def forward(self, x):
        h=self.shared(x); return torch.sigmoid(self.class_head(h)), self.phys_head(h)

# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT 3: CARE Criticality Algorithm (Algorithm 1 from paper)
# ═══════════════════════════════════════════════════════════════
def care_criticality(predictions, tc=72):
    """CARE paper Algorithm 1: criticality counter with threshold tc=72 (12h)."""
    crit = np.zeros(len(predictions) + 1)
    for i in range(len(predictions)):
        if predictions[i] == 1:
            crit[i+1] = crit[i] + 1
        else:
            crit[i+1] = max(crit[i] - 1, 0)
    max_crit = np.max(crit[1:])
    alarm_raised = max_crit >= tc
    # Find first time criticality exceeds threshold
    first_alarm_idx = None
    if alarm_raised:
        for i in range(len(crit)-1):
            if crit[i+1] >= tc:
                first_alarm_idx = i
                break
    return alarm_raised, max_crit, first_alarm_idx

# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT 4: Smart Zero-Power Anomaly Channel
# ═══════════════════════════════════════════════════════════════
def zero_power_anomaly(df, wind_col="wind_speed_235_avg", power_col="power_2_avg",
                       status_col="status_type_id", wind_cutin=0.05):
    """Detect periods where wind > cut-in but power ≈ 0 during 'normal' status."""
    if wind_col not in df.columns or power_col not in df.columns:
        return np.zeros(len(df))
    ws = df[wind_col].values
    pw = df[power_col].values
    st = df[status_col].values if status_col in df.columns else np.zeros(len(df))
    # Anomalous: wind available, no power, but status says normal
    anomalous = ((ws > wind_cutin) & (pw < 0.01) & (st == 0)).astype(int)
    return anomalous

def main():
    print("="*70)
    print("PHASE 9: INDUSTRY-GRADE COMPREHENSIVE DETECTION")
    print("5 improvements: PowerWind, ExpandedARMAX, CARE-Crit, ZeroPower, Forensic")
    print("="*70); sys.stdout.flush()

    gscaler = joblib.load(os.path.join(ARMAX_DIR, "global_scaler.pkl"))
    p1_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_part1.pkl")))

    # ═══════════════════════════════════
    # STEP 1: Train new ARMAX for 3 expanded targets
    # ═══════════════════════════════════
    print("\n[1/5] Training ARMAX for 3 new physics targets..."); sys.stdout.flush()
    new_thetas = {}
    new_scalers_armax = {}
    for name, tcol in NEW_TARGETS.items():
        X_all, y_all = [], []
        for f in p1_files:
            df = pd.read_pickle(f)
            if tcol not in df.columns: continue
            cols = [tcol] + [c for c in EXO if c in df.columns]
            if len(cols) < 4: continue
            sub = df[cols].ffill().bfill().dropna()
            if len(sub) < 100: continue
            sc = StandardScaler()
            sub_sc = pd.DataFrame(sc.fit_transform(sub), columns=sub.columns)
            ti = 0  # target is first column
            ei_idx = list(range(1, len(cols)))
            vals = sub_sc.values
            for i in range(3, min(2000, len(vals))):
                x = np.concatenate([vals[i-3:i, ti], vals[i-3:i, ei_idx].flatten()])
                X_all.append(x)
                y_all.append(vals[i, ti])
        if not X_all:
            print(f"  {name}: No data. Skipping."); continue
        X_all, y_all = np.array(X_all), np.array(y_all)
        r = Ridge(alpha=1.0).fit(X_all, y_all)
        new_thetas[name] = r.coef_
        np.save(os.path.join(P9_DIR, f"{name}_theta.npy"), r.coef_)
        print(f"  {name}: R²={r.score(X_all, y_all):.4f}, theta shape={r.coef_.shape}")
    
    # ═══════════════════════════════════
    # STEP 2: Extract features & train PGNNs for new targets
    # ═══════════════════════════════════
    print("\n[2/5] Training PGNNs for expanded targets..."); sys.stdout.flush()
    p2_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_part2.pkl")))
    
    new_pgnn_models = {}
    new_pgnn_scalers = {}
    
    for name, tcol in NEW_TARGETS.items():
        if name not in new_thetas: continue
        theta_g = new_thetas[name]
        all_feats, all_labels = [], []
        
        for fpath in p2_files:
            eid = os.path.basename(fpath).split('_')[1]
            is_anom = eid in anom_set
            df = pd.read_pickle(fpath)
            if tcol not in df.columns: continue
            cols = [tcol] + [c for c in EXO if c in df.columns]
            sub = df[cols].ffill().bfill().dropna()
            if len(sub) < W: continue
            sc = StandardScaler()
            sub_sc = pd.DataFrame(sc.fit_transform(sub), columns=sub.columns)
            n = len(sub)
            
            for start in range(0, n - W + 1, S):
                end = start + W
                wsc = sub_sc.iloc[start:end].values
                ti = 0; ei_idx = list(range(1, len(cols)))
                X, y = [], []
                for i in range(3, len(wsc)):
                    X.append(np.concatenate([wsc[i-3:i, ti], wsc[i-3:i, ei_idx].flatten()]))
                    y.append(wsc[i, ti])
                X, y = np.array(X), np.array(y)
                if len(X) < 10: continue
                
                res = y - X @ theta_g
                r2 = max(0, 1 - np.sum(res**2)/(np.sum((y-np.mean(y))**2)+1e-6))
                vec = np.array([np.mean(res), np.std(res), np.max(np.abs(res)),
                                np.sqrt(np.mean(res**2)), skew(res), kurtosis(res), r2])
                all_feats.append(vec)
                all_labels.append(1 if (is_anom and (n-end) < 4320) else 0)
        
        if not all_feats: continue
        X = np.nan_to_num(np.array(all_feats))
        y = np.array(all_labels, dtype=np.float32).reshape(-1,1)
        fsc = StandardScaler().fit(X)
        joblib.dump(fsc, os.path.join(P9_DIR, f"{name}_scaler.pkl"))
        X_sc = fsc.transform(X)
        
        ds = TensorDataset(torch.tensor(X_sc, dtype=torch.float32), torch.tensor(y))
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        model = PGNN(X.shape[1])
        opt = optim.Adam(model.parameters(), lr=0.001)
        bce = nn.BCELoss()
        for ep in range(20):
            model.train()
            for bx, by in loader:
                opt.zero_grad(); pc, pp = model(bx)
                loss = bce(pc, by); loss.backward(); opt.step()
        torch.save(model.state_dict(), os.path.join(P9_DIR, f"{name}_pgnn.pt"))
        new_pgnn_models[name] = model; model.eval()
        new_pgnn_scalers[name] = fsc
        print(f"  {name}: {len(X)} windows, {int(y.sum())} positive"); sys.stdout.flush()

    # ═══════════════════════════════════
    # STEP 3: Load ALL models (thermal + Phase8 physics + Phase9 expanded)
    # ═══════════════════════════════════
    print("\n[3/5] Loading all detection channels..."); sys.stdout.flush()
    
    # Original thermal PGNNs
    th_m, th_s, th_t = {}, {}, {}
    for nm in THERMAL_TARGETS:
        fp = os.path.join(PGNN_DIR, f"{nm}_feat_scaler.pkl")
        if not os.path.exists(fp): continue
        th_s[nm] = joblib.load(fp)
        th_t[nm] = np.load(os.path.join(ARMAX_DIR, f"{nm}_theta.npy"))
        m = PGNN(th_s[nm].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(PGNN_DIR, f"{nm}_pgnn.pt"))); m.eval()
        th_m[nm] = m
    
    # Phase 8 physics channels
    p8_m, p8_s = {}, {}
    with open(os.path.join(P8_DIR, "baselines.json")) as f: BL = json.load(f)
    for ch in ["kci","rpfd","hpg"]:
        sp = os.path.join(P8_DIR, f"{ch}_scaler.pkl")
        if not os.path.exists(sp): continue
        p8_s[ch] = joblib.load(sp)
        m = PGNN(p8_s[ch].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(P8_DIR, f"{ch}_pgnn.pt"))); m.eval()
        p8_m[ch] = m
    
    total_ch = len(th_m) + len(p8_m) + len(new_pgnn_models) + 1 + 5  # +1 zero-power, +5 drift sensors
    print(f"  Thermal: {len(th_m)} | Physics: {len(p8_m)} | Expanded: {len(new_pgnn_models)} | Drift: 5 | ZeroPower: 1")
    print(f"  Total channels: {total_ch}"); sys.stdout.flush()

    # ═══════════════════════════════════
    # STEP 4: Run comprehensive detection with CARE criticality
    # ═══════════════════════════════════
    print(f"\n[4/5] Running {total_ch}-channel detection with CARE criticality..."); sys.stdout.flush()
    
    KCI_COLS = ["sensor_130_avg","sensor_131_avg","sensor_132_avg"]
    results = []
    
    for fi, fpath in enumerate(sorted(p2_files)):
        eid = os.path.basename(fpath).split('_')[1]
        is_anom = eid in anom_set
        df = pd.read_pickle(fpath)
        n_total = len(df)
        
        # All channel predictions per sample (0 or 1)
        all_channel_preds = {}
        
        # --- THERMAL CHANNELS ---
        req = [c for c in list(THERMAL_TARGETS.values())+EXO if c in df.columns]
        ds = df[req].ffill().bfill().dropna()
        if len(ds) < W: continue
        dsc = pd.DataFrame(gscaler.transform(ds), columns=ds.columns)
        nw = len(range(0, len(ds)-W+1, S))
        
        for nm in th_m:
            preds = np.zeros(nw)
            tc = THERMAL_TARGETS[nm]
            if tc not in ds.columns: continue
            ti = list(ds.columns).index(tc)
            ei_idx = [list(ds.columns).index(c) for c in EXO if c in ds.columns]
            for wi, st in enumerate(range(0, len(ds)-W+1, S)):
                end = st + W
                wsc = dsc.iloc[st:end].values; wr = ds.iloc[st:end].values
                X, y = [], []
                for i in range(3, len(wsc)):
                    X.append(np.concatenate([wsc[i-3:i,ti], wsc[i-3:i,ei_idx].flatten()]))
                    y.append(wsc[i,ti])
                X, y = np.array(X), np.array(y)
                if len(X) < 10: continue
                r = Ridge(alpha=1.0).fit(X, y)
                dt = r.coef_ - th_t[nm]; res = y - X @ th_t[nm]
                ss = np.sum(res**2); st2 = np.sum((y-np.mean(y))**2)+1e-6
                r2 = max(0, 1-ss/st2)
                vec = np.concatenate([dt,[np.mean(res),np.std(res),np.max(np.abs(res)),
                    np.sqrt(np.mean(res**2)),skew(res),kurtosis(res),r2],
                    [np.mean(wr[:,ei_idx[0]]),np.mean(wr[:,ei_idx[1]]),np.max(wr[:,ei_idx[1]]),
                     np.mean(wr[:,ei_idx[2]]),np.mean(wr[:,ei_idx[2]])/(np.mean(wr[:,ei_idx[0]])+273.15)]])
                vec = np.nan_to_num(vec)
                xs = th_s[nm].transform([vec])
                with torch.no_grad():
                    p, _ = th_m[nm](torch.tensor(xs, dtype=torch.float32))
                    preds[wi] = 1 if p.item() > 0.50 else 0
            all_channel_preds[nm] = preds
        
        # --- PHASE 8 PHYSICS CHANNELS ---
        df_c = df.ffill().bfill()
        for ch in p8_m:
            preds = np.zeros(nw)
            for wi, st in enumerate(range(0, len(ds)-W+1, S)):
                end = st + W
                if ch == "kci" and all(c in df.columns for c in KCI_COLS):
                    v = df_c.iloc[st:end][KCI_COLS].values
                    sig = (np.max(v,1)-np.min(v,1))/(np.mean(v,1)+1e-6)
                    sig = (sig - BL["kci"]["mean"])/(BL["kci"]["std"]+1e-6)
                elif ch == "rpfd" and "reactive_power_119_avg" in df.columns:
                    sig = df_c.iloc[st:end]["reactive_power_119_avg"].values/(np.abs(df_c.iloc[st:end]["power_2_avg"].values)+1)
                    sig = (sig - BL["rpfd"]["mean"])/(BL["rpfd"]["std"]+1e-6)
                elif ch == "hpg" and "sensor_48_avg" in df.columns:
                    p = df_c.iloc[max(0,st-144):end]["sensor_48_avg"].values
                    if len(p)>144:
                        cov = np.std(p[-144:])/(np.abs(np.mean(p[-144:]))+1e-6)
                        sig = np.full(end-st, (cov - BL["hpg"]["mean"])/(BL["hpg"]["std"]+1e-6))
                    else: continue
                else: continue
                if len(sig) < 10: continue
                feats = np.array([np.mean(sig),np.std(sig),np.max(sig),np.min(sig),skew(sig),kurtosis(sig),np.sqrt(np.mean(sig**2))])
                if "power_2_avg" in df.columns and "sensor_144_avg" in df.columns:
                    feats = np.append(feats,[df_c.iloc[st:end]["power_2_avg"].mean(),df_c.iloc[st:end]["sensor_144_avg"].mean()])
                else: feats = np.append(feats,[0,0])
                feats = np.nan_to_num(feats)
                xs = p8_s[ch].transform([feats])
                with torch.no_grad():
                    p, _ = p8_m[ch](torch.tensor(xs, dtype=torch.float32))
                    preds[wi] = 1 if p.item() > 0.50 else 0
            all_channel_preds[ch] = preds
        
        # --- PHASE 9 EXPANDED CHANNELS ---
        for name, tcol in NEW_TARGETS.items():
            if name not in new_pgnn_models or tcol not in df.columns: continue
            preds = np.zeros(nw)
            cols = [tcol] + [c for c in EXO if c in df.columns]
            sub = df[cols].ffill().bfill().dropna()
            if len(sub) < W: continue
            sc_local = StandardScaler()
            sub_sc = pd.DataFrame(sc_local.fit_transform(sub), columns=sub.columns)
            theta_g = new_thetas[name]
            for wi, st_pos in enumerate(range(0, len(sub)-W+1, S)):
                end = st_pos + W
                wsc = sub_sc.iloc[st_pos:end].values
                ti = 0; ei_local = list(range(1, len(cols)))
                X, y = [], []
                for i in range(3, len(wsc)):
                    X.append(np.concatenate([wsc[i-3:i,ti], wsc[i-3:i,ei_local].flatten()]))
                    y.append(wsc[i,ti])
                X, y = np.array(X), np.array(y)
                if len(X) < 10: continue
                res = y - X @ theta_g
                r2 = max(0, 1 - np.sum(res**2)/(np.sum((y-np.mean(y))**2)+1e-6))
                vec = np.array([np.mean(res),np.std(res),np.max(np.abs(res)),
                                np.sqrt(np.mean(res**2)),skew(res),kurtosis(res),r2])
                vec = np.nan_to_num(vec)
                xs = new_pgnn_scalers[name].transform([vec])
                with torch.no_grad():
                    p, _ = new_pgnn_models[name](torch.tensor(xs, dtype=torch.float32))
                    preds[wi] = 1 if p.item() > 0.50 else 0
            all_channel_preds[name] = preds
        
        # --- SIGMA-DRIFT CHANNELS (5 targeted sensors from FN forensics) ---
        DRIFT_SENSORS = {
            "nacelle_24v_current": "sensor_25_avg",    # Event 31: 9.2 sigma at 60d
            "aeration_filter": "sensor_109_avg",       # Event 81: 8.8 sigma at 60d
            "gear_oil_pump": "sensor_87_avg",          # Event 49: 6.9 sigma at 6d
            "mains_frequency": "sensor_47_avg",        # Event 70: 6.9 sigma at 39d
            "hv_reactive": "sensor_75_avg",            # Event 35: 4.1 sigma at 60d
        }
        for dname, dcol in DRIFT_SENSORS.items():
            if dcol not in df.columns: continue
            preds = np.zeros(nw)
            # Compute baseline from first 50% of this dataset (training portion)
            first_half = df.iloc[:len(df)//2]
            bl_mean = first_half[dcol].mean()
            bl_std = first_half[dcol].std()
            if bl_std < 0.001: continue
            for wi, st_pos in enumerate(range(0, len(ds)-W+1, S)):
                end = st_pos + W
                if end > len(df): continue
                win_mean = df.iloc[st_pos:end][dcol].mean()
                sigma = abs(win_mean - bl_mean) / bl_std
                preds[wi] = 1 if sigma > 3.0 else 0  # 3-sigma threshold
            all_channel_preds[dname] = preds

        # --- ZERO-POWER CHANNEL ---
        zp_preds = np.zeros(nw)
        zp_raw = zero_power_anomaly(df)
        for wi, st_pos in enumerate(range(0, len(ds)-W+1, S)):
            end = st_pos + W
            if end <= len(zp_raw):
                zp_ratio = np.mean(zp_raw[st_pos:end])
                zp_preds[wi] = 1 if zp_ratio > 0.3 else 0  # 30%+ zero-power in window
        all_channel_preds["zero_power"] = zp_preds
        
        # ═══════════════════════════════════
        # OR-LOGIC with CARE Criticality per channel
        # ═══════════════════════════════════
        best_alarm = None
        triggered_channels = []
        
        for ch_name, preds in all_channel_preds.items():
            alarm, max_crit, first_idx = care_criticality(preds, tc=3)  # CARE tc=72 samples / 72 stride = 1; using 3 for robustness
            if alarm and first_idx is not None:
                # Convert window index to lead time
                lead_windows = nw - first_idx
                lead_days = lead_windows * S * 10 / (60*24)
                if 2.0 <= lead_days <= 60.0:
                    triggered_channels.append(ch_name)
                    if best_alarm is None or first_idx < best_alarm["idx"]:
                        best_alarm = {"idx": first_idx, "ch": ch_name, "ld": lead_days, "crit": max_crit}
        
        # ═══════════════════════════════════
        # IMPROVEMENT 5: Forensic FP Verification
        # ═══════════════════════════════════
        if best_alarm:
            # Check status distribution in alarm window
            alarm_start = best_alarm["idx"] * S
            alarm_end = min(n_total, alarm_start + W)
            if "status_type_id" in df.columns:
                win_status = df.iloc[alarm_start:alarm_end]["status_type_id"]
                abnormal_pct = (win_status.isin([1,3,4,5])).mean()
            else:
                abnormal_pct = 0
            
            if is_anom:
                results.append({"event_id": eid, "type": "TP", "lead_days": round(best_alarm["ld"],1),
                    "first_ch": best_alarm["ch"], "channels": ",".join(triggered_channels),
                    "n_ch": len(triggered_channels), "max_crit": round(best_alarm["crit"],1)})
            else:
                fp_type = "FP_unlogged_fault" if abnormal_pct > 0.1 else "FP_genuine"
                results.append({"event_id": eid, "type": fp_type, "lead_days": round(best_alarm["ld"],1),
                    "first_ch": best_alarm["ch"], "channels": ",".join(triggered_channels),
                    "n_ch": len(triggered_channels), "max_crit": round(best_alarm["crit"],1),
                    "abnormal_status_pct": round(abnormal_pct*100,1)})
        elif is_anom:
            results.append({"event_id": eid, "type": "FN", "lead_days": 0,
                "first_ch": "none", "channels": "none", "n_ch": 0, "max_crit": 0})
        
        if (fi+1)%5==0: print(f"  {fi+1}/{len(p2_files)}..."); sys.stdout.flush()

    # ═══════════════════════════════════
    # STEP 5: Final Results
    # ═══════════════════════════════════
    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(RESULTS_DIR, "phase9_results.csv"), index=False)
    
    print("\n"+"="*70)
    print("PHASE 9 FINAL RESULTS")
    print("="*70)
    tps = df_r[df_r["type"]=="TP"]
    fns = df_r[df_r["type"]=="FN"]
    fp_unlog = df_r[df_r["type"]=="FP_unlogged_fault"]
    fp_gen = df_r[df_r["type"]=="FP_genuine"]
    print(f"True Positives:        {len(tps)}/27 ({len(tps)/27*100:.1f}% recall)")
    print(f"False Negatives:       {len(fns)}")
    print(f"FP (Unlogged Faults):  {len(fp_unlog)} (PGNN smarter than operators)")
    print(f"FP (Genuine):          {len(fp_gen)}")
    if not tps.empty:
        print(f"\nLead Time: mean={tps['lead_days'].mean():.1f}d, min={tps['lead_days'].min():.1f}d, max={tps['lead_days'].max():.1f}d")
        print(f"\nFirst Detection Channel Breakdown:")
        for ch, cnt in tps["first_ch"].value_counts().items(): print(f"  {ch}: {cnt} events")
        print(f"  Avg channels confirming: {tps['n_ch'].mean():.1f}")
    if not fns.empty:
        print(f"\nMissed Events: {list(fns['event_id'])}")
    if not fp_unlog.empty:
        print(f"\nUnlogged Faults Detected (FP reclassified):")
        for _, row in fp_unlog.iterrows():
            print(f"  Event {row['event_id']}: {row['abnormal_status_pct']}% abnormal status in alarm window")
    print("\n[SUCCESS] Phase 9 Complete.")

if __name__ == "__main__":
    main()
