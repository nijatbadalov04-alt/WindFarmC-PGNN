import pandas as pd, numpy as np, os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
CONTEXT = ["wind_speed_236_avg","wind_speed_235_avg","power_2_avg","power_5_avg","power_6_avg"]

for eid in [1, 4]:
    df1 = pd.read_pickle(os.path.join(DATA, f"event_{eid}_part1.pkl"))
    df2 = pd.read_pickle(os.path.join(DATA, f"event_{eid}_part2.pkl"))
    mask = df1["status_type_id"] == 0
    
    ctx = [c for c in CONTEXT if c in df1.columns]
    sensor = "sensor_186_avg"
    
    X_tr = df1.loc[mask, ctx].values; y_tr = df1.loc[mask, sensor].values
    v = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    X_tr = X_tr[v]; y_tr = y_tr[v]
    
    sc = StandardScaler().fit(X_tr)
    model = Ridge(alpha=10).fit(sc.transform(X_tr), y_tr)
    r2 = model.score(sc.transform(X_tr), y_tr)
    
    res_tr = y_tr - model.predict(sc.transform(X_tr))
    
    X_te = df2[ctx].fillna(0).values
    X_te = np.where(np.isfinite(X_te), X_te, 0)
    y_te = df2[sensor].fillna(0).values
    y_pred = model.predict(sc.transform(X_te))
    res_te = y_te - y_pred
    norm_res = (res_te - res_tr.mean()) / res_tr.std()
    
    lbl = "normal" if eid == 1 else "anomaly"
    print(f"Event {eid} ({lbl}):")
    print(f"  R2={r2:.4f}")
    print(f"  Train res: mean={res_tr.mean():.2f} std={res_tr.std():.2f}")
    print(f"  Test norm_res: mean={norm_res.mean():.2f} std={norm_res.std():.2f} max={np.max(np.abs(norm_res)):.1f}")
    above3 = np.sum(np.abs(norm_res)>3)
    above5 = np.sum(np.abs(norm_res)>5)
    above10 = np.sum(np.abs(norm_res)>10)
    print(f"  >3s: {above3}/{len(norm_res)} ({above3/len(norm_res)*100:.1f}%)")
    print(f"  >5s: {above5}/{len(norm_res)} ({above5/len(norm_res)*100:.1f}%)")
    print(f"  >10s: {above10}/{len(norm_res)} ({above10/len(norm_res)*100:.1f}%)")
    
    # Also try: bin by power level, compute mean temp per bin
    # This is essentially what the "bin model" approach does
    print(f"  Context features: {ctx}")
    print()
