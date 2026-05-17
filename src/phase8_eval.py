"""Phase 8 Evaluation Only (Optimized) — Uses saved models from phase8 training"""
import os, sys, glob, json, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from sklearn.linear_model import Ridge
from scipy.stats import skew, kurtosis
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
ARMAX_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\armax"
PGNN_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\pgnn"
RESULTS_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
P8_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\phase8"

with open(os.path.join(ARMAX_DIR, "armax_config.json")) as f: cfg = json.load(f)
TARGETS = cfg["targets"]; EXO = cfg["exogenous"]
ei = pd.read_csv(EVENT_CSV, sep=";")
anom_set = set(ei[ei["event_label"]=="anomaly"]["event_id"].astype(str))
W, S, REC = 432, 72, 20

class PGNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1))
        self.class_head = nn.Linear(64,1); self.phys_head = nn.Linear(64,1)
    def forward(self, x):
        h=self.shared(x); return torch.sigmoid(self.class_head(h)), self.phys_head(h)

KCI = ["sensor_130_avg","sensor_131_avg","sensor_132_avg"]
with open(os.path.join(P8_DIR, "baselines.json")) as f: BL = json.load(f)

def main():
    print("="*70); print("PHASE 8 EVAL: 8-CHANNEL OR-LOGIC"); print("="*70); sys.stdout.flush()
    gscaler = joblib.load(os.path.join(ARMAX_DIR, "global_scaler.pkl"))

    # Load only gearbox + generator thermal (2 strongest) instead of all 5
    th_models, th_scalers, th_thetas = {}, {}, {}
    for nm in ["gearbox", "generator"]:
        fp = os.path.join(PGNN_DIR, f"{nm}_feat_scaler.pkl")
        th_scalers[nm] = joblib.load(fp)
        th_thetas[nm] = np.load(os.path.join(ARMAX_DIR, f"{nm}_theta.npy"))
        m = PGNN(th_scalers[nm].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(PGNN_DIR, f"{nm}_pgnn.pt"))); m.eval()
        th_models[nm] = m

    # Load new physics models
    new_m, new_s = {}, {}
    for ch in ["kci","rpfd","hpg"]:
        sp = os.path.join(P8_DIR, f"{ch}_scaler.pkl")
        if not os.path.exists(sp): continue
        new_s[ch] = joblib.load(sp)
        m = PGNN(new_s[ch].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(P8_DIR, f"{ch}_pgnn.pt"))); m.eval()
        new_m[ch] = m

    print(f"Loaded {len(th_models)} thermal + {len(new_m)} physics channels"); sys.stdout.flush()
    all_f = sorted(glob.glob(os.path.join(DATA_DIR, "*_part2.pkl")))
    results = []

    for fi, fpath in enumerate(all_f):
        eid = os.path.basename(fpath).split('_')[1]
        is_a = eid in anom_set
        df = pd.read_pickle(fpath)
        req = [c for c in list(TARGETS.values())+EXO if c in df.columns]
        ds = df[req].ffill().bfill().dropna()
        if len(ds) < W: continue
        dsc = pd.DataFrame(gscaler.transform(ds), columns=ds.columns)
        nw = len(range(0, len(ds)-W+1, S))

        # Channel scores
        scores = {}
        for nm in th_models: scores[nm] = np.zeros(nw)
        for ch in new_m: scores[ch] = np.zeros(nw)

        df_c = df.ffill().bfill()
        # Precompute physics signals once per dataset
        kci_sig = None
        if all(c in df.columns for c in KCI):
            v = df_c[KCI].values
            kci_sig = (np.max(v,1)-np.min(v,1))/(np.mean(v,1)+1e-6)
            kci_sig = (kci_sig - BL["kci"]["mean"]) / (BL["kci"]["std"]+1e-6)
        rpfd_sig = None
        if "reactive_power_119_avg" in df.columns and "power_2_avg" in df.columns:
            rpfd_sig = df_c["reactive_power_119_avg"].values / (np.abs(df_c["power_2_avg"].values)+1)
            rpfd_sig = (rpfd_sig - BL["rpfd"]["mean"]) / (BL["rpfd"]["std"]+1e-6)
        hpg_sig = None
        if "sensor_48_avg" in df.columns:
            p = df_c["sensor_48_avg"].values; hpg_raw = np.zeros(len(p))
            for i in range(144, len(p)):
                seg = p[i-144:i]; hpg_raw[i] = np.std(seg)/(np.abs(np.mean(seg))+1e-6)
            hpg_sig = (hpg_raw - BL["hpg"]["mean"]) / (BL["hpg"]["std"]+1e-6)

        for wi, st in enumerate(range(0, len(ds)-W+1, S)):
            end = st + W
            wsc = dsc.iloc[st:end].values; wraw = ds.iloc[st:end].values

            # Thermal PGNNs
            for nm in th_models:
                tc = TARGETS[nm]
                if tc not in ds.columns: continue
                ti = list(ds.columns).index(tc)
                ei_idx = [list(ds.columns).index(c) for c in EXO if c in ds.columns]
                X, y = [], []
                for i in range(3, len(wsc)):
                    X.append(np.concatenate([wsc[i-3:i,ti], wsc[i-3:i,ei_idx].flatten()]))
                    y.append(wsc[i,ti])
                X, y = np.array(X), np.array(y)
                if len(X) < 10: continue
                r = Ridge(alpha=1.0).fit(X, y)
                dt = r.coef_ - th_thetas[nm]
                res = y - X @ th_thetas[nm]
                ss_r = np.sum(res**2); ss_t = np.sum((y-np.mean(y))**2)+1e-6
                r2 = max(0, 1-ss_r/ss_t)
                vec = np.concatenate([dt,[np.mean(res),np.std(res),np.max(np.abs(res)),
                    np.sqrt(np.mean(res**2)),skew(res),kurtosis(res),r2],
                    [np.mean(wraw[:,ei_idx[0]]),np.mean(wraw[:,ei_idx[1]]),np.max(wraw[:,ei_idx[1]]),
                     np.mean(wraw[:,ei_idx[2]]),np.mean(wraw[:,ei_idx[2]])/(np.mean(wraw[:,ei_idx[0]])+273.15)]])
                vec = np.nan_to_num(vec)
                xs = th_scalers[nm].transform([vec])
                with torch.no_grad():
                    p, _ = th_models[nm](torch.tensor(xs, dtype=torch.float32))
                    scores[nm][wi] = p.item()

            # New physics channels — vectorized
            for ch, sig in [("kci", kci_sig), ("rpfd", rpfd_sig), ("hpg", hpg_sig)]:
                if sig is None or ch not in new_m: continue
                seg = sig[st:end]
                if len(seg) < 10: continue
                feats = np.array([np.mean(seg),np.std(seg),np.max(seg),np.min(seg),
                                  skew(seg),kurtosis(seg),np.sqrt(np.mean(seg**2))])
                if "power_2_avg" in df.columns and "sensor_144_avg" in df.columns:
                    feats = np.append(feats, [df_c.iloc[st:end]["power_2_avg"].mean(),
                                              df_c.iloc[st:end]["sensor_144_avg"].mean()])
                else: feats = np.append(feats, [0,0])
                feats = np.nan_to_num(feats)
                xs = new_s[ch].transform([feats])
                with torch.no_grad():
                    p, _ = new_m[ch](torch.tensor(xs, dtype=torch.float32))
                    scores[ch][wi] = p.item()

        # OR-logic: ANY channel confirmed alarm = detection
        all_alarms = []
        for ch, sc in scores.items():
            supp = 0
            for wi in range(nw):
                if supp > 0: supp -= 1; continue
                block = sc[max(0,wi-3):wi+1]
                if np.sum(block > 0.50) >= 2:
                    ld = (nw - wi) * S * 10 / (60*24)
                    if 2.0 <= ld <= 60.0:
                        all_alarms.append({"w":wi,"ch":ch,"p":sc[wi],"ld":ld})
                        supp = REC

        if all_alarms and is_a:
            ea = min(all_alarms, key=lambda a: a["w"])
            chs = list(set(a["ch"] for a in all_alarms))
            results.append({"event_id":eid,"type":"TP","lead_days":round(ea["ld"],1),
                "first_ch":ea["ch"],"channels":",".join(chs),"n_ch":len(chs),"prob":round(ea["p"],3)})
        elif all_alarms and not is_a:
            results.append({"event_id":eid,"type":"FP","lead_days":round(all_alarms[0]["ld"],1),
                "first_ch":all_alarms[0]["ch"],"channels":all_alarms[0]["ch"],"n_ch":1,"prob":round(all_alarms[0]["p"],3)})
        elif is_a:
            results.append({"event_id":eid,"type":"FN","lead_days":0,"first_ch":"none","channels":"none","n_ch":0,"prob":0})

        if (fi+1)%5==0: print(f"  {fi+1}/{len(all_f)}..."); sys.stdout.flush()

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(RESULTS_DIR,"phase8_results.csv"),index=False)
    print("\n"+"="*70); print("PHASE 8 RESULTS"); print("="*70)
    total = 27
    tps = df_r[df_r["type"]=="TP"]
    fns = df_r[df_r["type"]=="FN"]
    fps = df_r[df_r["type"]=="FP"]
    print(f"TP: {len(tps)}/{total} ({len(tps)/total*100:.1f}%)")
    print(f"FN: {len(fns)}")
    print(f"FP: {len(fps)}")
    if not tps.empty:
        print(f"Lead: mean={tps['lead_days'].mean():.1f}d, min={tps['lead_days'].min():.1f}d, max={tps['lead_days'].max():.1f}d")
        print(f"First detection channel breakdown:")
        for ch, cnt in tps["first_ch"].value_counts().items(): print(f"  {ch}: {cnt} events")
    if not fns.empty: print(f"Missed: {list(fns['event_id'])}")
    print("\n[DONE]")

if __name__=="__main__": main()
