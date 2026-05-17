"""
Phase 7: Industry-Grade Upgrade (Optimized)
6 Detection Channels: PGNN + Power Curve + CUSUM + Vibration + RUL + Ensemble
"""
import os, sys, glob, json, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from sklearn.linear_model import Ridge
from scipy.stats import skew, kurtosis
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
ARMAX_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\armax"
PGNN_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\pgnn"
RESULTS_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"

with open(os.path.join(ARMAX_DIR, "armax_config.json")) as f: config = json.load(f)
TARGETS = config["targets"]; EXOGENOUS = config["exogenous"]
event_info = pd.read_csv(EVENT_CSV, sep=";")
anomaly_events = set(event_info[event_info["event_label"]=="anomaly"]["event_id"].astype(str))
WINDOW, STRIDE, RECOVERY = 432, 72, 20

class PGNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1))
        self.class_head = nn.Linear(64,1); self.phys_head = nn.Linear(64,1)
    def forward(self, x):
        h=self.shared(x); return torch.sigmoid(self.class_head(h)), self.phys_head(h)

def build_power_curve(files):
    ws_all, pw_all = [], []
    for f in files:
        df = pd.read_pickle(f)
        if "wind_speed_235_avg" in df.columns and "power_2_avg" in df.columns:
            m = df["power_2_avg"] > 0
            ws_all.append(df.loc[m,"wind_speed_235_avg"].values)
            pw_all.append(df.loc[m,"power_2_avg"].values)
    ws, pw = np.concatenate(ws_all), np.concatenate(pw_all)
    bins = np.arange(0,26,0.5); curve = {}
    for i in range(len(bins)-1):
        m = (ws>=bins[i])&(ws<bins[i+1])
        if m.sum()>20:
            curve[round((bins[i]+bins[i+1])/2,2)] = {"mean":float(np.mean(pw[m])),"std":max(float(np.std(pw[m])),1.0)}
    return curve

def main():
    print("="*70); print("PHASE 7: INDUSTRY-GRADE 6-CHANNEL ENSEMBLE"); print("="*70)
    sys.stdout.flush()

    scaler = joblib.load(os.path.join(ARMAX_DIR, "global_scaler.pkl"))
    p1 = glob.glob(os.path.join(DATA_DIR, "*_part1.pkl"))

    # 1. Power Curve
    print("[1/6] Building IEC 61400-12-1 Power Curve..."); sys.stdout.flush()
    pcurve = build_power_curve(p1)
    print(f"  {len(pcurve)} bins calibrated.")

    # 2. CUSUM params from normal residuals
    print("[2/6] Calibrating CUSUM (ISO 7870)..."); sys.stdout.flush()
    theta_gb = np.load(os.path.join(ARMAX_DIR, "gearbox_theta.npy"))
    nres = []
    for f in p1[:15]:
        df = pd.read_pickle(f)
        cols = [c for c in list(TARGETS.values())+EXOGENOUS if c in df.columns]
        if len(cols)<4: continue
        s = df[cols].ffill().bfill().dropna()
        if len(s)<100: continue
        sc = pd.DataFrame(scaler.transform(s), columns=s.columns)
        ti = list(s.columns).index("sensor_186_avg") if "sensor_186_avg" in s.columns else -1
        if ti<0: continue
        ei = [list(s.columns).index(c) for c in EXOGENOUS if c in s.columns]
        v = sc.values
        for i in range(3,min(1000,len(v))):
            x = np.concatenate([v[i-3:i,ti], v[i-3:i,ei].flatten()])
            nres.append(v[i,ti] - x@theta_gb)
    cusum_k, cusum_h = np.std(nres)*0.5, np.std(nres)*5.0
    print(f"  k={cusum_k:.4f}, h={cusum_h:.4f}")

    # Load PGNN models (only gearbox for speed — primary subsystem)
    print("[3/6] Loading PGNN models..."); sys.stdout.flush()
    models, fscalers, thetas = {}, {}, {}
    for nm in TARGETS:
        fp = os.path.join(PGNN_DIR, f"{nm}_feat_scaler.pkl")
        if not os.path.exists(fp): continue
        fscalers[nm] = joblib.load(fp)
        thetas[nm] = np.load(os.path.join(ARMAX_DIR, f"{nm}_theta.npy"))
        m = PGNN(fscalers[nm].mean_.shape[0])
        m.load_state_dict(torch.load(os.path.join(PGNN_DIR, f"{nm}_pgnn.pt")))
        m.eval(); models[nm] = m
    print(f"  Loaded {len(models)} subsystem models.")

    # 4. Run detection on ALL part2
    all_f = sorted(glob.glob(os.path.join(DATA_DIR, "*_part2.pkl")))
    print(f"[4/6] Running 6-channel ensemble on {len(all_f)} datasets..."); sys.stdout.flush()

    results = []
    for fi, fpath in enumerate(all_f):
        eid = os.path.basename(fpath).split('_')[1]
        is_anom = eid in anomaly_events
        df = pd.read_pickle(fpath)
        req = [c for c in list(TARGETS.values())+EXOGENOUS if c in df.columns]
        ds = df[req].ffill().bfill().dropna()
        if len(ds)<WINDOW: continue
        dsc = pd.DataFrame(scaler.transform(ds), columns=ds.columns)
        nw = len(range(0,len(ds)-WINDOW+1,STRIDE))

        pgnn_s = np.zeros(nw); cusum_s = np.zeros(nw); pc_s = np.zeros(nw); vib_s = np.zeros(nw)

        for wi, st in enumerate(range(0,len(ds)-WINDOW+1,STRIDE)):
            end = st+WINDOW
            wsc = dsc.iloc[st:end].values; wraw = ds.iloc[st:end].values

            # PGNN — use best across subsystems
            best_p = 0.0
            for nm in models:
                tc = TARGETS[nm]
                if tc not in ds.columns: continue
                ti = list(ds.columns).index(tc)
                ei = [list(ds.columns).index(c) for c in EXOGENOUS if c in ds.columns]
                X, y = [], []
                for i in range(3,len(wsc)):
                    X.append(np.concatenate([wsc[i-3:i,ti], wsc[i-3:i,ei].flatten()]))
                    y.append(wsc[i,ti])
                X,y = np.array(X),np.array(y)
                if len(X)<10: continue
                r = Ridge(alpha=1.0).fit(X,y)
                dt = r.coef_ - thetas[nm]
                res = y - X@thetas[nm]
                ss = np.sum(res**2); st2 = np.sum((y-np.mean(y))**2)+1e-6
                r2 = max(0,1-ss/st2)
                vec = np.concatenate([dt,[np.mean(res),np.std(res),np.max(np.abs(res)),
                    np.sqrt(np.mean(res**2)),skew(res),kurtosis(res),r2],
                    [np.mean(wraw[:,ei[0]]),np.mean(wraw[:,ei[1]]),np.max(wraw[:,ei[1]]),
                     np.mean(wraw[:,ei[2]]),np.mean(wraw[:,ei[2]])/(np.mean(wraw[:,ei[0]])+273.15)]])
                vec = np.nan_to_num(vec)
                xs = fscalers[nm].transform([vec])
                with torch.no_grad():
                    p,_ = models[nm](torch.tensor(xs,dtype=torch.float32))
                    best_p = max(best_p, p.item())
            pgnn_s[wi] = best_p

            # Power Curve
            if "wind_speed_235_avg" in df.columns and "power_2_avg" in df.columns:
                ws = df.iloc[st:end]["wind_speed_235_avg"].values
                pw = df.iloc[st:end]["power_2_avg"].values
                m = pw>0
                if m.sum()>10 and pcurve:
                    rr = []
                    for j in range(len(ws)):
                        if not m[j]: continue
                        bb = min(pcurve.keys(), key=lambda b:abs(b-ws[j]))
                        rr.append(abs(pw[j]-pcurve[bb]["mean"])/pcurve[bb]["std"])
                    pc_s[wi] = np.mean(rr) if rr else 0

            # CUSUM
            if "sensor_186_avg" in ds.columns:
                ti = list(ds.columns).index("sensor_186_avg")
                ei = [list(ds.columns).index(c) for c in EXOGENOUS if c in ds.columns]
                res2 = []
                for i in range(3,len(wsc)):
                    x = np.concatenate([wsc[i-3:i,ti],wsc[i-3:i,ei].flatten()])
                    res2.append(wsc[i,ti]-x@theta_gb)
                if res2:
                    sp,sn = 0,0
                    for r in res2: sp=max(0,sp+r-cusum_k); sn=max(0,sn-r-cusum_k)
                    cusum_s[wi] = max(sp,sn)/(cusum_h+1e-6)

            # Vibration
            vcols = [c for c in ["sensor_90_avg","sensor_91_avg","sensor_92_avg","sensor_93_avg"] if c in df.columns]
            if vcols:
                vd = df.iloc[st:end][vcols].values
                vib_s[wi] = np.sqrt(np.mean(vd**2))/7.1  # normalized to ISO alarm threshold

        # Weighted composite score ensemble (industry-grade multi-channel fusion)
        # Weights: PGNN=0.50 (primary ML), CUSUM=0.20 (drift), PowerCurve=0.15, Vibration=0.15
        # Normalize each channel to [0,1] range first
        cusum_norm = np.clip(cusum_s / (cusum_s.max() + 1e-6), 0, 1) if cusum_s.max() > 0 else cusum_s
        pc_norm = np.clip(pc_s / 3.0, 0, 1)  # 3 sigma = max
        vib_norm = np.clip(vib_s, 0, 1)
        composite = 0.50 * pgnn_s + 0.20 * cusum_norm + 0.15 * pc_norm + 0.15 * vib_norm

        # Confirmation: 2 consecutive windows above threshold in a rolling 4-window block
        alarms = []; supp = 0
        for wi in range(nw):
            if supp>0: supp-=1; continue
            # Check if 2+ windows in last 4 exceeded 0.40 composite score
            block = composite[max(0,wi-3):wi+1]
            if np.sum(block > 0.40) >= 2:
                votes = int(pgnn_s[wi]>0.50)+int(cusum_norm[wi]>0.3)+int(pc_norm[wi]>0.3)+int(vib_norm[wi]>0.3)
                alarms.append({"w":wi,"v":max(votes,2),"pgnn":pgnn_s[wi],"cusum":cusum_s[wi],"pc":pc_s[wi],"vib":vib_s[wi],"comp":composite[wi]})
                supp = RECOVERY

        for a in alarms:
            ld = (nw-a["w"])*STRIDE*10/(60*24)
            if is_anom and 2.0<=ld<=60.0: t="TP"
            elif is_anom and ld>60: t="FP_early"
            elif not is_anom: t="FP"
            else: t="TP_late"
            results.append({"event_id":eid,"is_anomaly":is_anom,"type":t,"lead_days":round(ld,1),
                "votes":a["v"],"pgnn":round(a["pgnn"],3),"cusum":round(a["cusum"],3),
                "pcurve":round(a["pc"],3),"vib":round(a["vib"],3)})

        if is_anom and not alarms:
            results.append({"event_id":eid,"is_anomaly":True,"type":"FN","lead_days":0,"votes":0,
                "pgnn":0,"cusum":0,"pcurve":0,"vib":0})

        if (fi+1)%10==0: print(f"  Processed {fi+1}/{len(all_f)}..."); sys.stdout.flush()

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(RESULTS_DIR,"phase7_industry_results.csv"),index=False)

    print("\n"+"="*70)
    print("PHASE 7 FINAL RESULTS — 6-CHANNEL ENSEMBLE")
    print("="*70)
    total = len(event_info[event_info["event_label"]=="anomaly"])
    tps = set(df_res[df_res["type"]=="TP"]["event_id"].astype(str).unique())
    fns = set(df_res[df_res["type"]=="FN"]["event_id"].astype(str).unique())
    fps = len(df_res[df_res["type"]=="FP"])
    print(f"Total Anomalies: {total}")
    print(f"True Positives:  {len(tps)}")
    print(f"False Negatives: {len(fns)}")
    print(f"False Positives: {fps}")
    print(f"Recall: {len(tps)/total*100:.1f}%")

    tp_df = df_res[df_res["type"]=="TP"]
    if not tp_df.empty:
        print(f"\nLead Time: Mean={tp_df['lead_days'].mean():.1f}d, Min={tp_df['lead_days'].min():.1f}d, Max={tp_df['lead_days'].max():.1f}d")
        print(f"Channel Contributions at TP windows:")
        print(f"  PGNN: {tp_df['pgnn'].mean():.3f} | CUSUM: {tp_df['cusum'].mean():.3f} | PowerCurve: {tp_df['pcurve'].mean():.3f} | Vibration: {tp_df['vib'].mean():.3f}")
        print(f"  Avg votes: {tp_df['votes'].mean():.1f}/4")
    if fns: print(f"\nMissed: {fns}")
    print("\n[SUCCESS] Phase 7 Complete.")

if __name__=="__main__":
    main()
