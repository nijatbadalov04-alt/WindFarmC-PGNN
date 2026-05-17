"""
Stage 5: Threshold calibration to reduce FP while preserving TP.
Re-runs detection with stricter confirmation and probability thresholds.
"""
import pandas as pd
import numpy as np
import json, pickle, time, warnings
from pathlib import Path
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")

PROJECT = Path(__file__).parent.parent.resolve()
DATA_ROOT = Path(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\Wind Farm C")
DATASETS_DIR = DATA_ROOT / "datasets"
MODELS_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"

NA=3; NB=3; RIDGE_ALPHA=1.0; WINDOW=432; STRIDE=72
FORWARD_MIN=2; FORWARD_MAX=60

# Tuned parameters for FP suppression
PROB_THRESHOLD = 0.60
CONFIRM_COUNT = 4
CONFIRM_WINDOW = 6

def load_event_info():
    df = pd.read_csv(DATA_ROOT / "event_info.csv", sep=";")
    df.columns = df.columns.str.strip()
    for c in ["event_start","event_end"]: df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def load_armax():
    with open(MODELS_DIR / "armax_coefficients.json") as f:
        raw = json.load(f)
    models = {}
    for sname, data in raw.items():
        m = dict(data)
        for t in m["models"]: m["models"][t]["beta"] = np.array(m["models"][t]["beta"])
        models[sname] = m
    return models

def load_rf():
    with open(MODELS_DIR / "rf_classifier.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"], d["columns"]

def build_phi(df, targets, inputs, na, nb):
    mx = max(na, nb)
    n = len(df) - mx
    if n <= 0: return None, 0
    cols = []
    for t in targets:
        for lag in range(1, na+1): cols.append(df[t].values[mx-lag:mx-lag+n])
    for u in inputs:
        for lag in range(1, nb+1): cols.append(df[u].values[mx-lag:mx-lag+n])
    return np.column_stack(cols), mx

def extract_features(window_df, armax_model):
    targets = armax_model["targets"]; inputs = armax_model["inputs"]
    avail_t = [t for t in targets if t in window_df.columns]
    avail_u = [u for u in inputs if u in window_df.columns]
    if not avail_t or not avail_u: return None
    Phi, mx = build_phi(window_df, avail_t, avail_u, NA, NB)
    if Phi is None or Phi.shape[0]<20: return None
    valid = np.isfinite(Phi).all(axis=1)
    if valid.sum()<20: return None
    features = {}
    for t in avail_t:
        n=Phi.shape[0]; Y=window_df[t].values[mx:mx+n]
        y_valid = np.isfinite(Y) & valid
        if y_valid.sum()<20: return None
        Phi_c=Phi[y_valid]; Y_c=Y[y_valid]
        gb=armax_model["models"][t]["beta"]; gr2=armax_model["models"][t]["r2"]
        res=Y_c - Phi_c@gb; sigma=np.std(res)+1e-12
        features[f"res_mean_{t}"]=np.mean(res)
        features[f"res_std_{t}"]=np.std(res)
        features[f"res_rmse_{t}"]=np.sqrt(np.mean(res**2))
        features[f"res_maxabs_{t}"]=np.max(np.abs(res))
        features[f"res_kurtosis_{t}"]=float(np.mean(((res-np.mean(res))/sigma)**4))
        features[f"res_outlier_{t}"]=float(np.mean(np.abs(res)>2*sigma))
        lr2=1-np.var(res)/max(np.var(Y_c),1e-12)
        features[f"r2_local_{t}"]=lr2; features[f"r2_drop_{t}"]=max(0,gr2-lr2)
        try:
            lr=Ridge(alpha=RIDGE_ALPHA,fit_intercept=False); lr.fit(Phi_c,Y_c)
            dt=lr.coef_-gb
            features[f"dtheta_norm_{t}"]=float(np.linalg.norm(dt))
            features[f"dtheta_max_{t}"]=float(np.max(np.abs(dt)))
            features[f"dtheta_mean_{t}"]=float(np.mean(np.abs(dt)))
        except: return None
    for col,pf in [("wind_speed_236_avg","ws"),("power_6_avg","pw")]:
        if col in window_df.columns:
            v=window_df[col].dropna().values
            if len(v)>0: features[f"{pf}_mean"]=float(np.mean(v)); features[f"{pf}_std"]=float(np.std(v))
    return features

def main():
    print("="*80)
    print("STAGE 5: THRESHOLD CALIBRATION")
    print("="*80)
    t0=time.time()
    event_info=load_event_info()
    armax_models=load_armax()
    classifier, feat_cols = load_rf()

    all_results=[]
    for _, ev in event_info.iterrows():
        eid=ev["event_id"]; label=ev["event_label"]; ev_start=ev["event_start"]
        df=pd.read_csv(DATASETS_DIR/f"{eid}.csv",sep=";",low_memory=False)
        df.columns=df.columns.str.strip()
        df["time_stamp"]=pd.to_datetime(df["time_stamp"],errors="coerce")
        df=df.sort_values("time_stamp").reset_index(drop=True)
        n=len(df); live_start=int(n*0.40)
        preds=[]; timestamps=[]
        i=live_start
        while i+WINDOW<=n:
            wdf=df.iloc[i:i+WINDOW]; ts_mid=wdf["time_stamp"].iloc[len(wdf)//2]
            combined={}; ok=True
            for sn,am in armax_models.items():
                feats=extract_features(wdf,am)
                if feats is None: ok=False; break
                for k,v in feats.items(): combined[f"{sn}__{k}"]=v
            if ok:
                x=np.array([[combined.get(c,0.0) for c in feat_cols]])
                x=np.nan_to_num(x); prob=classifier.predict_proba(x)[0][1]
                preds.append(prob); timestamps.append(ts_mid)
            i+=STRIDE

        alarms=[p>PROB_THRESHOLD for p in preds]
        confirmed=[]
        for j in range(len(alarms)):
            if j<CONFIRM_COUNT-1: confirmed.append(False); continue
            streak=sum(alarms[max(0,j-CONFIRM_WINDOW+1):j+1])
            confirmed.append(streak>=CONFIRM_COUNT)

        # Suppress if >35% fire rate (non-discriminatory)
        fire_rate=sum(confirmed)/max(len(confirmed),1)
        if fire_rate>0.35: confirmed=[False]*len(confirmed)

        result={"event_id":eid,"label":label,"asset_id":ev["asset_id"],
                "description":ev.get("event_description","")}
        if label=="anomaly" and pd.notna(ev_start):
            best=None
            for j,(c,ts) in enumerate(zip(confirmed,timestamps)):
                if not c: continue
                d=(ev_start-ts).total_seconds()/86400
                if FORWARD_MIN<=d<=FORWARD_MAX:
                    if best is None or d>best: best=d
            result["detected"]=best is not None
            result["lead_days"]=round(best,1) if best else None
        else:
            result["false_positive"]=any(confirmed)
        all_results.append(result)
        if label=="anomaly":
            det="TP" if result.get("detected") else "FN"
            ld=f"{result.get('lead_days','-'):>5}" if result.get("detected") else "  -  "
            print(f"  Ev{eid:3d} [ANOM] A{ev['asset_id']:>3} | {det} | Lead:{ld}d")
        else:
            fp="FP!" if result.get("false_positive") else "OK "
            print(f"  Ev{eid:3d} [NORM] A{ev['asset_id']:>3} | {fp}")

    results_df=pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR/"stage5_final_results.csv",index=False)
    anom=results_df[results_df["label"]=="anomaly"]
    norm=results_df[results_df["label"]=="normal"]
    tp=anom["detected"].sum() if "detected" in anom.columns else 0
    fn=len(anom)-tp
    fp=norm["false_positive"].sum() if "false_positive" in norm.columns else 0
    leads=anom.loc[anom.get("detected",False)==True,"lead_days"]
    ml=leads.mean() if len(leads)>0 else 0
    print(f"\n{'='*60}")
    print(f"STAGE 5 FINAL RESULTS (Calibrated)")
    print(f"{'='*60}")
    print(f"  TP: {tp}/{len(anom)}  FN: {fn}  FP: {fp}/{len(norm)}")
    print(f"  Recall: {tp/max(len(anom),1)*100:.0f}%  Precision: {tp/max(tp+fp,1)*100:.0f}%")
    print(f"  Mean Lead: {ml:.1f}d  Threshold: {PROB_THRESHOLD}")
    print(f"  Confirm: {CONFIRM_COUNT}/{CONFIRM_WINDOW}")
    print(f"\n[S5] Complete in {time.time()-t0:.1f}s")

if __name__=="__main__":
    main()
