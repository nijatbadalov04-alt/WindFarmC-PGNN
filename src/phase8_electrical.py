"""
Phase 8: Physics-Guided Electrical & Hydraulic Fault Detection
Adds 3 new physics channels: KCI (Kirchhoff), RPFD (Power Factor), HPG (Hydraulic)
Uses OR-logic ensemble: ANY confirmed channel alarm = detection
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
RESULTS_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
P8_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\phase8"
os.makedirs(P8_DIR, exist_ok=True)

with open(os.path.join(ARMAX_DIR, "armax_config.json")) as f: config = json.load(f)
THERMAL_TARGETS = config["targets"]
EXOGENOUS = config["exogenous"]
event_info = pd.read_csv(EVENT_CSV, sep=";")
anomaly_events = set(event_info[event_info["event_label"]=="anomaly"]["event_id"].astype(str))

WINDOW, STRIDE, RECOVERY = 432, 72, 20
MIN_LEAD = (2*24*60)//(10*STRIDE)
MAX_LEAD = (60*24*60)//(10*STRIDE)

# New physics sensor groups
KCI_COLS = ["sensor_130_avg", "sensor_131_avg", "sensor_132_avg"]  # Gen RMS current L1/L2/L3
RPFD_COLS = ["reactive_power_119_avg", "power_2_avg"]
HPG_COLS = ["sensor_48_avg"]  # Hydraulic aggregate pressure

class PGNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1))
        self.class_head = nn.Linear(64,1); self.phys_head = nn.Linear(64,1)
    def forward(self, x):
        h=self.shared(x); return torch.sigmoid(self.class_head(h)), self.phys_head(h)

def compute_kci(df):
    """Kirchhoff Current Imbalance: (I_max - I_min) / I_avg per row"""
    vals = df[KCI_COLS].values
    i_max = np.max(vals, axis=1)
    i_min = np.min(vals, axis=1)
    i_avg = np.mean(vals, axis=1) + 1e-6
    return (i_max - i_min) / i_avg

def compute_rpfd(df):
    """Reactive Power Factor Drift: Q / (|P| + 1)"""
    return df["reactive_power_119_avg"].values / (np.abs(df["power_2_avg"].values) + 1.0)

def compute_hpg(df, window=144):
    """Hydraulic Pressure Gradient: rolling CoV of pressure"""
    p = df["sensor_48_avg"].values
    cov = np.zeros(len(p))
    for i in range(window, len(p)):
        seg = p[i-window:i]
        mu = np.mean(seg)
        cov[i] = np.std(seg) / (np.abs(mu) + 1e-6)
    return cov

def extract_window_features(signal, window_start, window_end):
    """Extract 7 statistical features from a physics signal window"""
    seg = signal[window_start:window_end]
    if len(seg) < 10: return None
    return np.array([np.mean(seg), np.std(seg), np.max(seg), np.min(seg),
                     skew(seg), kurtosis(seg), np.sqrt(np.mean(seg**2))])

def main():
    print("="*70)
    print("PHASE 8: ELECTRICAL + HYDRAULIC PHYSICS CHANNELS")
    print("="*70); sys.stdout.flush()

    # ═══════════════════════════════════════════
    # STEP 1: Build physics baselines from Part 1
    # ═══════════════════════════════════════════
    print("\n[1/4] Building physics baselines from normal operation..."); sys.stdout.flush()
    p1_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_part1.pkl")))

    kci_normals, rpfd_normals, hpg_normals = [], [], []
    for f in p1_files:
        df = pd.read_pickle(f)
        if all(c in df.columns for c in KCI_COLS):
            kci = compute_kci(df)
            kci_normals.append(kci[np.isfinite(kci)])
        if all(c in df.columns for c in RPFD_COLS):
            rpfd = compute_rpfd(df)
            rpfd_normals.append(rpfd[np.isfinite(rpfd)])
        if all(c in df.columns for c in HPG_COLS):
            hpg = compute_hpg(df)
            hpg_normals.append(hpg[144:])  # skip first window

    kci_baseline = {"mean": float(np.mean(np.concatenate(kci_normals))),
                    "std": float(np.std(np.concatenate(kci_normals)))}
    rpfd_baseline = {"mean": float(np.mean(np.concatenate(rpfd_normals))),
                     "std": float(np.std(np.concatenate(rpfd_normals)))}
    hpg_baseline = {"mean": float(np.mean(np.concatenate(hpg_normals))),
                    "std": float(np.std(np.concatenate(hpg_normals)))}

    print(f"  KCI baseline:  mean={kci_baseline['mean']:.4f}, std={kci_baseline['std']:.4f}")
    print(f"  RPFD baseline: mean={rpfd_baseline['mean']:.6f}, std={rpfd_baseline['std']:.6f}")
    print(f"  HPG baseline:  mean={hpg_baseline['mean']:.4f}, std={hpg_baseline['std']:.4f}")

    baselines = {"kci": kci_baseline, "rpfd": rpfd_baseline, "hpg": hpg_baseline}
    with open(os.path.join(P8_DIR, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=4)

    # ═══════════════════════════════════════════
    # STEP 2: Train PGNNs for new channels
    # ═══════════════════════════════════════════
    print("\n[2/4] Training PGNNs for 3 new physics channels..."); sys.stdout.flush()
    p2_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_part2.pkl")))
    np.random.seed(42)
    idx = np.arange(len(p2_files)); np.random.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]
    train_files = [p2_files[i] for i in train_idx]
    test_files = [p2_files[i] for i in test_idx]

    with open(os.path.join(P8_DIR, "test_files.json"), "w") as f:
        json.dump([os.path.basename(tf) for tf in test_files], f)

    # Extract features for each new channel
    for ch_name, compute_fn, cols, bl in [
        ("kci", compute_kci, KCI_COLS, kci_baseline),
        ("rpfd", compute_rpfd, RPFD_COLS, rpfd_baseline),
        ("hpg", lambda df: compute_hpg(df), HPG_COLS, hpg_baseline)
    ]:
        print(f"\n  Training PGNN for {ch_name.upper()}...")
        all_feats, all_labels = [], []

        for fpath in train_files:
            eid = os.path.basename(fpath).split('_')[1]
            is_anom = eid in anomaly_events
            df = pd.read_pickle(fpath)
            if not all(c in df.columns for c in cols): continue
            df_clean = df.ffill().bfill()

            signal = compute_fn(df_clean)
            n = len(signal)

            # Normalize signal using baseline
            signal_norm = (signal - bl["mean"]) / (bl["std"] + 1e-6)

            for start in range(0, n - WINDOW + 1, STRIDE):
                end = start + WINDOW
                feats = extract_window_features(signal_norm, start, end)
                if feats is None: continue

                # Add contextual features: power mean, rotor speed mean
                if "power_2_avg" in df.columns and "sensor_144_avg" in df.columns:
                    pw = df_clean.iloc[start:end]["power_2_avg"].mean()
                    rs = df_clean.iloc[start:end]["sensor_144_avg"].mean()
                    feats = np.append(feats, [pw, rs])
                else:
                    feats = np.append(feats, [0, 0])

                label = 1 if (is_anom and (n - end) < 4320) else 0
                all_feats.append(feats)
                all_labels.append(label)

        X = np.nan_to_num(np.array(all_feats))
        y = np.array(all_labels, dtype=np.float32).reshape(-1,1)

        if len(X) == 0:
            print(f"    No features for {ch_name}. Skipping.")
            continue

        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, os.path.join(P8_DIR, f"{ch_name}_scaler.pkl"))
        X_sc = scaler.transform(X)

        dataset = TensorDataset(torch.tensor(X_sc, dtype=torch.float32), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        model = PGNN(X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        bce = nn.BCELoss()

        for epoch in range(20):
            model.train()
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pc, pp = model(bx)
                loss = bce(pc, by)
                loss.backward(); optimizer.step()
                total_loss += loss.item()

        torch.save(model.state_dict(), os.path.join(P8_DIR, f"{ch_name}_pgnn.pt"))
        pos = y.sum(); neg = len(y) - pos
        print(f"    Trained: {len(X)} windows, {int(pos)} positive, loss={total_loss/len(loader):.4f}")

    # ═══════════════════════════════════════════
    # STEP 3: Load ALL models (5 thermal + 3 new)
    # ═══════════════════════════════════════════
    print("\n[3/4] Loading all 8 detection channels..."); sys.stdout.flush()
    global_scaler = joblib.load(os.path.join(ARMAX_DIR, "global_scaler.pkl"))

    # Thermal PGNNs
    thermal_models, thermal_scalers, thermal_thetas = {}, {}, {}
    for nm in THERMAL_TARGETS:
        fp = os.path.join(PGNN_DIR, f"{nm}_feat_scaler.pkl")
        if not os.path.exists(fp): continue
        thermal_scalers[nm] = joblib.load(fp)
        thermal_thetas[nm] = np.load(os.path.join(ARMAX_DIR, f"{nm}_theta.npy"))
        m = PGNN(thermal_scalers[nm].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(PGNN_DIR, f"{nm}_pgnn.pt")))
        m.eval(); thermal_models[nm] = m

    # New physics PGNNs
    new_models, new_scalers = {}, {}
    for ch in ["kci", "rpfd", "hpg"]:
        sp = os.path.join(P8_DIR, f"{ch}_scaler.pkl")
        mp = os.path.join(P8_DIR, f"{ch}_pgnn.pt")
        if not os.path.exists(sp) or not os.path.exists(mp): continue
        new_scalers[ch] = joblib.load(sp)
        m = PGNN(new_scalers[ch].mean_.shape[0])
        m.load_state_dict(torch.load(mp))
        m.eval(); new_models[ch] = m

    print(f"  Thermal channels: {len(thermal_models)} | New physics channels: {len(new_models)}")

    # ═══════════════════════════════════════════
    # STEP 4: Run OR-logic ensemble on ALL datasets
    # ═══════════════════════════════════════════
    print(f"\n[4/4] Running OR-logic ensemble on ALL {len(p2_files)} datasets..."); sys.stdout.flush()

    compute_fns = {"kci": (compute_kci, KCI_COLS, kci_baseline),
                   "rpfd": (compute_rpfd, RPFD_COLS, rpfd_baseline),
                   "hpg": (lambda df: compute_hpg(df), HPG_COLS, hpg_baseline)}
    results = []

    for fi, fpath in enumerate(sorted(p2_files)):
        eid = os.path.basename(fpath).split('_')[1]
        is_anom = eid in anomaly_events
        df = pd.read_pickle(fpath)
        n_total = len(df)

        # Thermal channels
        req = [c for c in list(THERMAL_TARGETS.values())+EXOGENOUS if c in df.columns]
        ds = df[req].ffill().bfill().dropna()
        if len(ds) < WINDOW: continue
        dsc = pd.DataFrame(global_scaler.transform(ds), columns=ds.columns)
        nw = len(range(0, len(ds)-WINDOW+1, STRIDE))

        # Score each window across ALL channels
        channel_scores = {nm: np.zeros(nw) for nm in list(thermal_models.keys()) + list(new_models.keys())}

        for wi, st in enumerate(range(0, len(ds)-WINDOW+1, STRIDE)):
            end = st + WINDOW
            wsc = dsc.iloc[st:end].values
            wraw = ds.iloc[st:end].values

            # Thermal PGNN channels
            for nm in thermal_models:
                tc = THERMAL_TARGETS[nm]
                if tc not in ds.columns: continue
                ti = list(ds.columns).index(tc)
                ei = [list(ds.columns).index(c) for c in EXOGENOUS if c in ds.columns]
                X, y = [], []
                for i in range(3, len(wsc)):
                    X.append(np.concatenate([wsc[i-3:i,ti], wsc[i-3:i,ei].flatten()]))
                    y.append(wsc[i,ti])
                X, y = np.array(X), np.array(y)
                if len(X) < 10: continue
                r = Ridge(alpha=1.0).fit(X, y)
                dt = r.coef_ - thermal_thetas[nm]
                res = y - X @ thermal_thetas[nm]
                ss = np.sum(res**2); st2 = np.sum((y-np.mean(y))**2)+1e-6
                r2 = max(0, 1-ss/st2)
                vec = np.concatenate([dt, [np.mean(res),np.std(res),np.max(np.abs(res)),
                    np.sqrt(np.mean(res**2)),skew(res),kurtosis(res),r2],
                    [np.mean(wraw[:,ei[0]]),np.mean(wraw[:,ei[1]]),np.max(wraw[:,ei[1]]),
                     np.mean(wraw[:,ei[2]]),np.mean(wraw[:,ei[2]])/(np.mean(wraw[:,ei[0]])+273.15)]])
                vec = np.nan_to_num(vec)
                xs = thermal_scalers[nm].transform([vec])
                with torch.no_grad():
                    p, _ = thermal_models[nm](torch.tensor(xs, dtype=torch.float32))
                    channel_scores[nm][wi] = max(channel_scores[nm][wi], p.item())

            # New physics channels
            df_clean = df.ffill().bfill()
            for ch in new_models:
                fn, cols, bl = compute_fns[ch]
                if not all(c in df.columns for c in cols): continue
                signal = fn(df_clean)
                signal_norm = (signal - bl["mean"]) / (bl["std"] + 1e-6)
                feats = extract_window_features(signal_norm, st, end)
                if feats is None: continue
                if "power_2_avg" in df.columns and "sensor_144_avg" in df.columns:
                    feats = np.append(feats, [df_clean.iloc[st:end]["power_2_avg"].mean(),
                                              df_clean.iloc[st:end]["sensor_144_avg"].mean()])
                else:
                    feats = np.append(feats, [0, 0])
                feats = np.nan_to_num(feats)
                xs = new_scalers[ch].transform([feats])
                with torch.no_grad():
                    p, _ = new_models[ch](torch.tensor(xs, dtype=torch.float32))
                    channel_scores[ch][wi] = p.item()

        # OR-logic ensemble: ANY channel with confirmed alarm = detection
        all_alarms = []
        for ch_name, scores in channel_scores.items():
            supp = 0
            for wi in range(nw):
                if supp > 0: supp -= 1; continue
                # Confirmation: 2 windows above 0.50 in rolling 4
                block = scores[max(0,wi-3):wi+1]
                if np.sum(block > 0.50) >= 2:
                    lead = nw - wi
                    lead_days = lead * STRIDE * 10 / (60*24)
                    if 2.0 <= lead_days <= 60.0:
                        all_alarms.append({"w": wi, "ch": ch_name, "prob": scores[wi], "lead_days": lead_days})
                        supp = RECOVERY

        # Deduplicate: keep earliest alarm per event
        if all_alarms and is_anom:
            earliest = min(all_alarms, key=lambda a: a["w"])
            channels_involved = list(set(a["ch"] for a in all_alarms))
            results.append({"event_id": eid, "is_anomaly": True, "type": "TP",
                "lead_days": round(earliest["lead_days"],1),
                "first_channel": earliest["ch"], "all_channels": ",".join(channels_involved),
                "prob": round(earliest["prob"],3), "n_channels": len(channels_involved)})
        elif all_alarms and not is_anom:
            for a in all_alarms:
                results.append({"event_id": eid, "is_anomaly": False, "type": "FP",
                    "lead_days": round(a["lead_days"],1), "first_channel": a["ch"],
                    "all_channels": a["ch"], "prob": round(a["prob"],3), "n_channels": 1})
        elif is_anom and not all_alarms:
            results.append({"event_id": eid, "is_anomaly": True, "type": "FN",
                "lead_days": 0, "first_channel": "none", "all_channels": "none",
                "prob": 0, "n_channels": 0})

        if (fi+1) % 10 == 0: print(f"  Processed {fi+1}/{len(p2_files)}..."); sys.stdout.flush()

    # ═══════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(RESULTS_DIR, "phase8_results.csv"), index=False)

    print("\n" + "="*70)
    print("PHASE 8 FINAL RESULTS — 8-CHANNEL OR-LOGIC ENSEMBLE")
    print("="*70)
    total = len(event_info[event_info["event_label"]=="anomaly"])
    tps = df_res[df_res["type"]=="TP"]
    fns = df_res[df_res["type"]=="FN"]
    fps = df_res[df_res["type"]=="FP"]
    print(f"Total Anomalies:  {total}")
    print(f"True Positives:   {len(tps)} ({len(tps)/total*100:.1f}% recall)")
    print(f"False Negatives:  {len(fns)}")
    print(f"False Positives:  {len(fps)}")
    if not tps.empty:
        print(f"\nLead Time: Mean={tps['lead_days'].mean():.1f}d, Min={tps['lead_days'].min():.1f}d, Max={tps['lead_days'].max():.1f}d")
        print(f"\nDetection Channel Breakdown:")
        for ch in tps["first_channel"].value_counts().items():
            print(f"  {ch[0]}: first to detect {ch[1]} events")
        print(f"  Avg channels confirming per event: {tps['n_channels'].mean():.1f}")
    if not fns.empty:
        print(f"\nMissed Events: {list(fns['event_id'])}")
    print("\n[SUCCESS] Phase 8 Complete.")

if __name__ == "__main__":
    main()
