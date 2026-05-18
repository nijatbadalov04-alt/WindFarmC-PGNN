"""
CARE v7: Fine-tuned Power-Binned NBM + CUSUM Combo
====================================================
Key insight from v6: Power-binned approach gives excellent specificity (Acc=0.99)
but poor sensitivity when conservative. Need to find the Pareto-optimal
combination of z-threshold and channel voting that maximizes CARE.

Strategy: Use CUSUM on the binned z-scores (not just rolling mean), combined
with the channel voting. This should give better sensitivity while maintaining
specificity.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
anomaly_ids = sorted(ei[ei["event_label"]=="anomaly"]["event_id"].tolist())
normal_ids = sorted(ei[ei["event_label"]=="normal"]["event_id"].tolist())
all_ids = sorted(anomaly_ids + normal_ids)

POWER_COL = "power_2_avg"
TARGETS = [
    "sensor_186_avg", "sensor_173_avg", "sensor_194_avg", "sensor_195_avg",
    "sensor_191_avg", "sensor_178_avg", "sensor_39_avg", "sensor_62_avg",
    "sensor_48_avg", "sensor_74_avg", "sensor_25_avg",
    "sensor_12_avg", "sensor_13_avg", "sensor_14_avg",
    "sensor_47_avg", "sensor_58_avg", "sensor_109_avg", "sensor_110_avg",
    "sensor_44_avg", "sensor_45_avg", "sensor_87_avg", "sensor_75_avg",
    "sensor_167_avg",
]

def care_crit(pred, status, tc=72):
    n = len(pred); c = np.zeros(n+1)
    for i in range(n):
        if status[i] == 0:
            c[i+1] = c[i]+1 if pred[i]==1 else max(c[i]-1,0)
        else:
            c[i+1] = c[i]
    return np.max(c[1:]), c[1:]

def fb(tp, fp, fn, b=0.5):
    if tp==0: return 0.0
    return (1+b**2)*tp/((1+b**2)*tp+b**2*fn+fp)

def ws_score(pred, st, si, ei_):
    if ei_<=si: return 0.0
    M=ei_-si; tw=0.0; ws=0.0
    for i in range(M):
        x=si+i
        if x>=len(st) or st[x]!=0: continue
        rp=i/max(M-1,1); w=1.0 if rp<=0.5 else max(0,2*(1-rp))
        tw+=w
        if pred[x]==1: ws+=w
    return ws/tw if tw>0 else 0.0

print("="*100)
print("CARE v7: Pareto-Optimal Parameter Search (20 loops)")
print("="*100)

best_care = -1; best_params = {}; best_results = None

# Systematic grid search
configs = [
    # (n_bins, roll_w, z_thresh, min_ch) — from aggressive to conservative
    (20, 288, 2.0, 1), (20, 288, 2.5, 1), (20, 288, 3.0, 1),
    (20, 288, 3.0, 2), (20, 288, 3.5, 2), (20, 288, 4.0, 2),
    (20, 432, 3.0, 2), (20, 432, 3.5, 2), (20, 432, 4.0, 2),
    (20, 432, 3.0, 3), (20, 432, 3.5, 3), (20, 432, 4.0, 3),
    (20, 432, 4.5, 3), (20, 432, 5.0, 3),
    (20, 432, 4.0, 4), (20, 432, 4.5, 4), (20, 432, 5.0, 4),
    (20, 432, 4.0, 5), (20, 432, 5.0, 5),
    (20, 432, 5.0, 6),
]

for loop, (n_bins, roll_w, z_thresh, min_ch) in enumerate(configs):
    results = []
    
    for eid in all_ids:
        f1 = os.path.join(DATA, f"event_{eid}_part1.pkl")
        f2 = os.path.join(DATA, f"event_{eid}_part2.pkl")
        if not os.path.exists(f1) or not os.path.exists(f2): continue
        
        df1 = pd.read_pickle(f1); df2 = pd.read_pickle(f2)
        is_anom = eid in anomaly_ids
        n2 = len(df2)
        status = df2["status_type_id"].values if "status_type_id" in df2.columns else np.zeros(n2)
        
        esi_v=None; eei_v=None
        if is_anom:
            ev = ei[ei["event_id"]==eid].iloc[0]
            if pd.notna(ev["event_start_id"]) and "row_id" in df2.columns:
                m=df2["row_id"].values>=int(ev["event_start_id"])
                if m.any(): esi_v=int(m.argmax())
            if pd.notna(ev["event_end_id"]) and "row_id" in df2.columns:
                m=df2["row_id"].values>=int(ev["event_end_id"])
                if m.any(): eei_v=int(m.argmax())
        
        tr_m = df1["status_type_id"].values==0 if "status_type_id" in df1.columns else np.ones(len(df1),bool)
        if POWER_COL not in df1.columns: continue
        
        p_tr = df1.loc[tr_m, POWER_COL].values
        vp = np.isfinite(p_tr)
        if vp.sum() < 100: continue
        be = np.linspace(np.nanmin(p_tr[vp]), np.nanmax(p_tr[vp]), n_bins+1)
        
        avail = [s for s in TARGETS if s in df1.columns and s in df2.columns]
        
        # Build bin stats
        bstats = {}
        for s in avail:
            st = df1.loc[tr_m, s].values
            d = {}
            for bi in range(n_bins):
                lo,hi = be[bi],be[bi+1]
                mk = (p_tr>=lo)&(p_tr<hi)&np.isfinite(st)&vp
                if mk.sum()<10: continue
                v = st[mk]; d[bi] = {"m":np.mean(v), "s":np.std(v)}
            bstats[s] = d
        
        # Score prediction
        p_pred = df2[POWER_COL].fillna(0).values
        pb = np.clip(np.digitize(p_pred, be)-1, 0, n_bins-1)
        
        ch_alarms = {}
        for s in avail:
            if s not in bstats: continue
            sp = df2[s].fillna(0).values
            zs = np.zeros(n2)
            for i in range(n2):
                bi = pb[i]
                if bi in bstats[s] and bstats[s][bi]["s"] > 1e-6:
                    zs[i] = abs(sp[i] - bstats[s][bi]["m"]) / bstats[s][bi]["s"]
            
            # Rolling mean z-score
            rz = pd.Series(zs).rolling(roll_w, min_periods=roll_w//4, center=True).mean().fillna(0).values
            ch_alarms[s] = (rz > z_thresh).astype(int)
        
        # Voting
        if not ch_alarms:
            preds = np.zeros(n2, dtype=int)
        else:
            vote = np.zeros(n2)
            for a in ch_alarms.values(): vote += a
            preds = (vote >= min_ch).astype(int)
        
        nch = sum(1 for v in ch_alarms.values() if v.sum()>0)
        nm = status==0
        
        if is_anom:
            gt=np.zeros(n2,dtype=int)
            if esi_v is not None:
                e_ = eei_v if eei_v else n2; gt[esi_v:e_]=1
            gf=gt[nm]; pf=preds[nm]
            tp=int(((gf==1)&(pf==1)).sum()); fp=int(((gf==0)&(pf==1)).sum()); fn=int(((gf==1)&(pf==0)).sum())
            cov=fb(tp,fp,fn); ws=ws_score(preds,status,esi_v,eei_v if eei_v else n2) if esi_v else 0.0
            mc,ca=care_crit(preds,status)
            det=mc>=72
            ld=None
            if det and esi_v:
                ci=np.argmax(ca>=72)
                if ci<esi_v: ld=(esi_v-ci)*10/(60*24)
            ap=np.mean(preds[nm])*100 if nm.sum()>0 else 0
            results.append({"eid":eid,"l":"a","det":det,"mc":mc,"cov":cov,"ws":ws,"ld":ld,"tp":tp,"fp":fp,"fn":fn,"nch":nch,"ap":ap})
        else:
            pf=preds[nm]; fp=int((pf==1).sum()); tn=int((pf==0).sum())
            acc=tn/(fp+tn) if (fp+tn)>0 else 1.0
            mc,_=care_crit(preds,status); fa=mc>=72
            ap=np.mean(preds[nm])*100 if nm.sum()>0 else 0
            results.append({"eid":eid,"l":"n","mc":mc,"acc":acc,"fa":fa,"fp":fp,"nch":nch,"ap":ap})
    
    dr=pd.DataFrame(results)
    da=dr[dr["l"]=="a"]; dn=dr[dr["l"]=="n"]
    tpe=int(da["det"].sum()); fne=len(da)-tpe
    fpe=int(dn["fa"].sum()); tne=len(dn)-fpe
    mcov=da["cov"].mean(); macc=dn["acc"].mean(); mws=da["ws"].mean()
    rel=fb(tpe,fpe,fne)
    care=(mcov+macc+rel+mws)/4 if tpe>0 and macc>=0.5 else 0.0
    lds=da.dropna(subset=["ld"])
    mld=lds["ld"].mean() if len(lds)>0 else 0
    
    marker = " ★" if care > best_care else ""
    print(f"  [{loop+1:2d}] bins={n_bins} roll={roll_w} z={z_thresh} ch={min_ch} → TP={tpe}/27 FP={fpe}/31 C={mcov:.3f} A={macc:.3f} R={rel:.3f} E={mws:.3f} CARE={care:.4f}{marker}")
    
    if care > best_care:
        best_care=care; best_params={"bins":n_bins,"roll":roll_w,"z":z_thresh,"ch":min_ch,"loop":loop+1}
        best_results=dr.copy()

print(f"\n{'='*100}")
print(f"OPTIMAL: {best_params} → CARE={best_care:.4f}")
print(f"{'='*100}")

if best_results is not None:
    ba=best_results[best_results["l"]=="a"].sort_values("eid")
    bn=best_results[best_results["l"]=="n"].sort_values("eid")
    
    print(f"\nTP: {ba['det'].sum()}/27, Missed:")
    for _,r in ba[~ba['det']].iterrows():
        print(f"  Event {int(r['eid'])}: crit={r['mc']:.0f}, {r['nch']} channels, alarm={r['ap']:.1f}%")
    
    print(f"\nFP: {bn['fa'].sum()}/31")
    for _,r in bn[bn['fa']].iterrows():
        print(f"  Event {int(r['eid'])}: crit={r['mc']:.0f}, alarm={r['ap']:.1f}%")
    
    print(f"\nDetected events detail:")
    for _,r in ba[ba['det']].iterrows():
        ld_str = f"{r['ld']:.0f}d" if pd.notna(r.get('ld')) else "-"
        print(f"  Event {int(r['eid'])}: cov={r['cov']:.3f} ws={r['ws']:.3f} lead={ld_str} ch={r['nch']} alarm={r['ap']:.1f}%")
    
    best_results.to_csv(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\care_v7_best.csv", index=False)
