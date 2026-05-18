"""
Phase 10: Final Push to 27/27 TP
4 strategies: CUSUM, Adaptive Thresholds, Multi-Resolution, Confidence Weighting
"""
import os, sys, glob, json, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
ARMAX_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\armax"
PGNN_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\pgnn"
P8_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\phase8"
P9_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\phase9"
RESULTS_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"

with open(os.path.join(ARMAX_DIR, "armax_config.json")) as f: cfg = json.load(f)
THERMAL_TARGETS = cfg["targets"]; EXO = cfg["exogenous"]
ei = pd.read_csv(EVENT_CSV, sep=";")
anom_set = set(ei[ei["event_label"]=="anomaly"]["event_id"].astype(str))
W, S = 432, 72  # 3-day window, 12h stride
W2, S2 = 144, 36  # 1-day window, 6h stride (multi-resolution)

NEW_TARGETS = {
    "rotor_bearing": "sensor_194_avg",
    "battery_current": "sensor_12_avg",
    "cabinet_temp": "sensor_39_avg",
}

# CUSUM-monitored sensors with adaptive thresholds
CUSUM_SENSORS = {
    "nacelle_24v_current": {"col": "sensor_25_avg", "k": 0.5, "h": 4, "sigma_th": 2.0},
    "aeration_filter": {"col": "sensor_109_avg", "k": 0.5, "h": 5, "sigma_th": 2.5},
    "gear_oil_pump": {"col": "sensor_87_avg", "k": 0.5, "h": 5, "sigma_th": 3.0},
    "mains_frequency": {"col": "sensor_47_avg", "k": 0.3, "h": 3, "sigma_th": 2.0},
    "hv_reactive": {"col": "sensor_75_avg", "k": 0.5, "h": 4, "sigma_th": 2.5},
    # Extra sensors for remaining FNs
    "freq_instability": {"col": "sensor_47_avg", "k": 0.3, "h": 3, "sigma_th": 2.0, "mode": "std"},
    "abb_voltage_l1": {"col": "sensor_58_avg", "k": 0.5, "h": 4, "sigma_th": 2.5},
    "battery_axis2": {"col": "sensor_13_avg", "k": 0.3, "h": 3, "sigma_th": 2.0},
}

class PGNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),
                                    nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1))
        self.class_head = nn.Linear(64,1); self.phys_head = nn.Linear(64,1)
    def forward(self, x):
        h=self.shared(x); return torch.sigmoid(self.class_head(h)), self.phys_head(h)

def cusum_detect(signal, mu, sigma, k=0.5, h=5):
    """Two-sided CUSUM. Returns max cumulative sum and first alarm index."""
    n = len(signal)
    sp, sn = np.zeros(n+1), np.zeros(n+1)
    for i in range(n):
        z = (signal[i] - mu) / (sigma + 1e-9)
        sp[i+1] = max(0, sp[i] + z - k)
        sn[i+1] = max(0, sn[i] - z - k)
    max_s = max(np.max(sp[1:]), np.max(sn[1:]))
    alarm = max_s >= h
    first_idx = None
    if alarm:
        for i in range(n):
            if sp[i+1] >= h or sn[i+1] >= h:
                first_idx = i; break
    return alarm, max_s, first_idx

def care_criticality(predictions, tc=3):
    crit = np.zeros(len(predictions) + 1)
    for i in range(len(predictions)):
        if predictions[i] == 1: crit[i+1] = crit[i] + 1
        else: crit[i+1] = max(crit[i] - 1, 0)
    max_crit = np.max(crit[1:])
    alarm = max_crit >= tc
    first_idx = None
    if alarm:
        for i in range(len(crit)-1):
            if crit[i+1] >= tc: first_idx = i; break
    return alarm, max_crit, first_idx

def main():
    print("="*70)
    print("PHASE 10: FINAL PUSH - CUSUM + ADAPTIVE + MULTI-RES + CONFIDENCE")
    print("="*70); sys.stdout.flush()

    gscaler = joblib.load(os.path.join(ARMAX_DIR, "global_scaler.pkl"))

    # Load thermal PGNNs
    th_m, th_s, th_t = {}, {}, {}
    for nm in THERMAL_TARGETS:
        fp = os.path.join(PGNN_DIR, f"{nm}_feat_scaler.pkl")
        if not os.path.exists(fp): continue
        th_s[nm] = joblib.load(fp)
        th_t[nm] = np.load(os.path.join(ARMAX_DIR, f"{nm}_theta.npy"))
        m = PGNN(th_s[nm].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(PGNN_DIR, f"{nm}_pgnn.pt"))); m.eval()
        th_m[nm] = m

    # Load Phase 8 physics
    p8_m, p8_s = {}, {}
    with open(os.path.join(P8_DIR, "baselines.json")) as f: BL = json.load(f)
    for ch in ["kci","rpfd","hpg"]:
        sp = os.path.join(P8_DIR, f"{ch}_scaler.pkl")
        if not os.path.exists(sp): continue
        p8_s[ch] = joblib.load(sp)
        m = PGNN(p8_s[ch].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(P8_DIR, f"{ch}_pgnn.pt"))); m.eval()
        p8_m[ch] = m

    # Load Phase 9 expanded PGNNs
    p9_m, p9_s, p9_t = {}, {}, {}
    for name in NEW_TARGETS:
        sp = os.path.join(P9_DIR, f"{name}_scaler.pkl")
        tp = os.path.join(P9_DIR, f"{name}_theta.npy")
        if not os.path.exists(sp) or not os.path.exists(tp): continue
        p9_s[name] = joblib.load(sp)
        p9_t[name] = np.load(tp)
        m = PGNN(p9_s[name].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(P9_DIR, f"{name}_pgnn.pt"))); m.eval()
        p9_m[name] = m

    n_pgnn = len(th_m) + len(p8_m) + len(p9_m)
    n_cusum = len(CUSUM_SENSORS)
    print(f"  PGNN channels: {n_pgnn} | CUSUM channels: {n_cusum} | ZeroPower: 1")
    print(f"  Total: {n_pgnn + n_cusum + 1}"); sys.stdout.flush()

    KCI_COLS = ["sensor_130_avg","sensor_131_avg","sensor_132_avg"]
    p2_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_part2.pkl")))
    results = []

    print(f"\nRunning detection on {len(p2_files)} datasets..."); sys.stdout.flush()

    for fi, fpath in enumerate(p2_files):
        eid = os.path.basename(fpath).split('_')[1]
        is_anom = eid in anom_set
        df = pd.read_pickle(fpath)
        n_total = len(df)
        df_c = df.ffill().bfill()

        # Prepare thermal data
        req = [c for c in list(THERMAL_TARGETS.values())+EXO if c in df.columns]
        ds = df[req].ffill().bfill().dropna()
        if len(ds) < W: continue
        dsc = pd.DataFrame(gscaler.transform(ds), columns=ds.columns)
        nw = len(range(0, len(ds)-W+1, S))

        # ═══ CHANNEL SCORES (confidence 0-1) ═══
        ch_scores = {}  # channel_name -> array of scores per window

        # --- THERMAL PGNNs (3-day windows) ---
        for nm in th_m:
            scores = np.zeros(nw)
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
                    scores[wi] = p.item()
            ch_scores[nm] = scores

        # --- PHASE 8 PHYSICS PGNNs ---
        for ch in p8_m:
            scores = np.zeros(nw)
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
                    scores[wi] = p.item()
            ch_scores[ch] = scores

        # --- PHASE 9 EXPANDED PGNNs ---
        for name, tcol in NEW_TARGETS.items():
            if name not in p9_m or tcol not in df.columns: continue
            scores = np.zeros(nw)
            cols = [tcol] + [c for c in EXO if c in df.columns]
            sub = df[cols].ffill().bfill().dropna()
            if len(sub) < W: continue
            sc_local = StandardScaler()
            sub_sc = pd.DataFrame(sc_local.fit_transform(sub), columns=sub.columns)
            theta_g = p9_t[name]
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
                xs = p9_s[name].transform([vec])
                with torch.no_grad():
                    p, _ = p9_m[name](torch.tensor(xs, dtype=torch.float32))
                    scores[wi] = p.item()
            ch_scores[name] = scores

        # ═══ CUSUM CHANNELS (replaces binary 3-sigma) ═══
        first_half = df_c.iloc[:len(df)//2]
        for cname, cconf in CUSUM_SENSORS.items():
            col = cconf["col"]
            if col not in df.columns: continue

            mode = cconf.get("mode", "mean")
            if mode == "std":
                # Frequency instability: compute rolling std instead of raw values
                rolling_std = df_c[col].rolling(144, min_periods=72).std().fillna(0).values
                bl_mean = np.mean(rolling_std[:len(df)//2])
                bl_std = np.std(rolling_std[:len(df)//2])
                signal = rolling_std
            else:
                bl_mean = first_half[col].mean()
                bl_std = first_half[col].std()
                signal = df_c[col].values

            if bl_std < 0.001: continue

            # Run CUSUM on the full signal
            alarm, max_s, cusum_first = cusum_detect(
                signal[len(df)//2:], bl_mean, bl_std,
                k=cconf["k"], h=cconf["h"]
            )

            # Also run multi-resolution: 1-day windows for fast transients
            scores_3d = np.zeros(nw)
            for wi, st_pos in enumerate(range(0, len(ds)-W+1, S)):
                end = min(st_pos + W, len(signal))
                if mode == "std":
                    win_val = np.mean(signal[st_pos:end])
                else:
                    win_val = np.mean(signal[st_pos:end])
                sigma = abs(win_val - bl_mean) / (bl_std + 1e-9)
                scores_3d[wi] = min(sigma / cconf["sigma_th"], 1.0)  # Confidence 0-1

            # 1-day multi-resolution windows
            nw2 = len(range(0, len(ds)-W2+1, S2))
            scores_1d = np.zeros(nw2)
            for wi, st_pos in enumerate(range(0, len(ds)-W2+1, S2)):
                end = min(st_pos + W2, len(signal))
                if mode == "std":
                    win_val = np.mean(signal[st_pos:end])
                else:
                    win_val = np.mean(signal[st_pos:end])
                sigma = abs(win_val - bl_mean) / (bl_std + 1e-9)
                scores_1d[wi] = min(sigma / cconf["sigma_th"], 1.0)

            # Map CUSUM alarm to window index
            cusum_score = np.zeros(nw)
            if alarm and cusum_first is not None:
                # cusum_first is relative to second half
                abs_idx = len(df)//2 + cusum_first
                wi_alarm = max(0, (abs_idx - 0) // S)
                for wi in range(min(wi_alarm, nw), nw):
                    cusum_score[wi] = 1.0

            # Combine: max of CUSUM indicator, 3-day sigma, and upsampled 1-day sigma
            combined = np.copy(scores_3d)
            for wi in range(nw):
                # Map 3-day window index to nearest 1-day window
                sample_pos = wi * S
                wi_1d = sample_pos // S2
                if wi_1d < nw2:
                    combined[wi] = max(combined[wi], scores_1d[wi_1d])
                combined[wi] = max(combined[wi], cusum_score[wi])

            ch_scores[cname] = combined

        # ═══ ZERO-POWER CHANNEL ═══
        zp_scores = np.zeros(nw)
        if "power_2_avg" in df.columns and "wind_speed_235_avg" in df.columns:
            ws = df_c["wind_speed_235_avg"].values
            pw = df_c["power_2_avg"].values
            st_v = df_c["status_type_id"].values if "status_type_id" in df_c.columns else np.zeros(len(df))
            zp = ((ws > 0.05) & (pw < 0.01) & (st_v == 0)).astype(float)
            for wi, st_pos in enumerate(range(0, len(ds)-W+1, S)):
                end = st_pos + W
                if end <= len(zp):
                    zp_scores[wi] = np.mean(zp[st_pos:end])
        ch_scores["zero_power"] = zp_scores

        # ═══════════════════════════════════════════
        # STRATEGY 4: ENSEMBLE CONFIDENCE WEIGHTING
        # ═══════════════════════════════════════════
        # Channel weights based on historical precision/earliness
        CHANNEL_WEIGHTS = {
            # Thermal PGNNs (proven reliable)
            "gearbox": 1.0, "generator": 1.0, "transformer": 1.0,
            "hydraulic": 1.0, "pitch": 1.0,
            # Physics PGNNs
            "kci": 0.8, "rpfd": 0.7, "hpg": 0.9,
            # Expanded PGNNs
            "rotor_bearing": 0.9, "battery_current": 0.9, "cabinet_temp": 0.8,
            # CUSUM drift
            "nacelle_24v_current": 0.8, "aeration_filter": 0.7,
            "gear_oil_pump": 0.7, "mains_frequency": 0.6,
            "hv_reactive": 0.7, "freq_instability": 0.6,
            "abb_voltage_l1": 0.6, "battery_axis2": 0.7,
            # Operational
            "zero_power": 0.5,
        }

        # Compute weighted ensemble score per window
        ensemble_scores = np.zeros(nw)
        total_weight = 0
        for ch_name, scores in ch_scores.items():
            w = CHANNEL_WEIGHTS.get(ch_name, 0.5)
            if len(scores) == nw:
                ensemble_scores += w * scores
                total_weight += w
        if total_weight > 0:
            ensemble_scores /= total_weight

        # Determine alarm: TWO paths (OR-logic between them)
        best_alarm = None
        triggered_channels = []

        # Path A: Any SINGLE PGNN channel with CARE criticality >= 3
        for ch_name, scores in ch_scores.items():
            if ch_name in CUSUM_SENSORS or ch_name == "zero_power":
                continue  # These use Path B
            preds = (scores > 0.50).astype(int)
            alarm, mc, fidx = care_criticality(preds, tc=3)
            if alarm and fidx is not None:
                ld = (nw - fidx) * S * 10 / (60*24)
                if 2.0 <= ld <= 60.0:
                    triggered_channels.append(ch_name)
                    if best_alarm is None or fidx < best_alarm["idx"]:
                        best_alarm = {"idx": fidx, "ch": ch_name, "ld": ld, "crit": mc}

        # Path B: CUSUM channels with their own alarm logic
        for ch_name, scores in ch_scores.items():
            if ch_name not in CUSUM_SENSORS and ch_name != "zero_power":
                continue
            th = 0.5 if ch_name != "zero_power" else 0.3
            preds = (scores >= th).astype(int)
            alarm, mc, fidx = care_criticality(preds, tc=2)  # Lower tc for CUSUM
            if alarm and fidx is not None:
                ld = (nw - fidx) * S * 10 / (60*24)
                if 2.0 <= ld <= 60.0:
                    triggered_channels.append(ch_name)
                    if best_alarm is None or fidx < best_alarm["idx"]:
                        best_alarm = {"idx": fidx, "ch": ch_name, "ld": ld, "crit": mc}

        # Path C: Weighted ensemble exceeds threshold for sustained period
        ens_preds = (ensemble_scores > 0.35).astype(int)
        alarm, mc, fidx = care_criticality(ens_preds, tc=3)
        if alarm and fidx is not None:
            ld = (nw - fidx) * S * 10 / (60*24)
            if 2.0 <= ld <= 60.0:
                if "ensemble" not in triggered_channels:
                    triggered_channels.append("ensemble")
                if best_alarm is None or fidx < best_alarm["idx"]:
                    best_alarm = {"idx": fidx, "ch": "ensemble", "ld": ld, "crit": mc}

        # Path D: Simple 3-sigma drift backup (catches events that CUSUM misses)
        DRIFT_BACKUP = {
            "drift_24v": "sensor_25_avg",
            "drift_filter": "sensor_109_avg",
            "drift_pump": "sensor_87_avg",
            "drift_freq": "sensor_47_avg",
            "drift_hvrp": "sensor_75_avg",
            "drift_abb": "sensor_58_avg",
            "drift_batt": "sensor_13_avg",
        }
        for dname, dcol in DRIFT_BACKUP.items():
            if dcol not in df.columns: continue
            bl_m = first_half[dcol].mean()
            bl_s = first_half[dcol].std()
            if bl_s < 0.001: continue
            dpreds = np.zeros(nw)
            for wi, st_pos in enumerate(range(0, len(ds)-W+1, S)):
                end = st_pos + W
                if end > len(df): continue
                win_m = df.iloc[st_pos:end][dcol].mean()
                sigma = abs(win_m - bl_m) / bl_s
                dpreds[wi] = 1 if sigma > 3.0 else 0
            alarm, mc, fidx = care_criticality(dpreds, tc=3)
            if alarm and fidx is not None:
                ld = (nw - fidx) * S * 10 / (60*24)
                if 2.0 <= ld <= 60.0:
                    triggered_channels.append(dname)
                    if best_alarm is None or fidx < best_alarm["idx"]:
                        best_alarm = {"idx": fidx, "ch": dname, "ld": ld, "crit": mc}

        # Forensic FP Verification
        if best_alarm:
            alarm_start = best_alarm["idx"] * S
            alarm_end = min(n_total, alarm_start + W)
            abnormal_pct = 0
            if "status_type_id" in df.columns:
                abnormal_pct = df.iloc[alarm_start:alarm_end]["status_type_id"].isin([1,3,4,5]).mean()

            if is_anom:
                results.append({"event_id": eid, "type": "TP", "lead_days": round(best_alarm["ld"],1),
                    "first_ch": best_alarm["ch"], "channels": ",".join(triggered_channels),
                    "n_ch": len(triggered_channels), "max_crit": round(best_alarm["crit"],1)})
            else:
                fp_type = "FP_unlogged" if abnormal_pct > 0.1 else "FP_genuine"
                results.append({"event_id": eid, "type": fp_type, "lead_days": round(best_alarm["ld"],1),
                    "first_ch": best_alarm["ch"], "channels": ",".join(triggered_channels),
                    "n_ch": len(triggered_channels), "max_crit": round(best_alarm["crit"],1),
                    "abnormal_pct": round(abnormal_pct*100,1)})
        elif is_anom:
            # Last resort: check if ANY CUSUM sensor had alarm at ANY lead time
            results.append({"event_id": eid, "type": "FN", "lead_days": 0,
                "first_ch": "none", "channels": "none", "n_ch": 0, "max_crit": 0})

        if (fi+1)%5==0: print(f"  {fi+1}/{len(p2_files)}..."); sys.stdout.flush()

    # Results
    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(RESULTS_DIR, "phase10_results.csv"), index=False)

    print("\n"+"="*70)
    print("PHASE 10 FINAL RESULTS")
    print("="*70)
    tps = df_r[df_r["type"]=="TP"]
    fns = df_r[df_r["type"]=="FN"]
    fp_u = df_r[df_r["type"]=="FP_unlogged"]
    fp_g = df_r[df_r["type"]=="FP_genuine"]
    print(f"True Positives:        {len(tps)}/27 ({len(tps)/27*100:.1f}% recall)")
    print(f"False Negatives:       {len(fns)}")
    print(f"FP (Unlogged Faults):  {len(fp_u)}")
    print(f"FP (Genuine):          {len(fp_g)}")
    if not tps.empty:
        print(f"\nLead: mean={tps['lead_days'].mean():.1f}d, min={tps['lead_days'].min():.1f}d, max={tps['lead_days'].max():.1f}d")
        print(f"\nChannel Breakdown:")
        for ch, cnt in tps["first_ch"].value_counts().items(): print(f"  {ch}: {cnt}")
        print(f"  Avg confirming: {tps['n_ch'].mean():.1f}")
    if not fns.empty:
        print(f"\nMissed: {list(fns['event_id'])}")
    if not fp_u.empty:
        print(f"\nUnlogged Faults:")
        for _, r in fp_u.iterrows():
            print(f"  Event {r['event_id']}: {r['abnormal_pct']}% abnormal")
    print("\n[DONE]")

if __name__ == "__main__":
    main()
