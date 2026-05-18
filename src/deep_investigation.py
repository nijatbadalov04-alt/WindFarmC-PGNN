"""Deep forensic investigation: 5 remaining FNs + lead time analysis for all TPs"""
import pandas as pd, numpy as np, os, glob

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
FEAT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\feature_description.csv"
RESULTS = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\phase9_results.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
res = pd.read_csv(RESULTS)
feat = pd.read_csv(FEAT_CSV, sep=";")

# ═══════════════════════════════════════
# PART 1: ROOT CAUSE ANALYSIS — 5 MISSED EVENTS
# ═══════════════════════════════════════
print("="*80)
print("PART 1: ROOT CAUSE ANALYSIS — 5 FALSE NEGATIVES")
print("="*80)

missed = res[res["type"]=="FN"]["event_id"].astype(str).tolist()
for eid in missed:
    f = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if not os.path.exists(f): continue
    df = pd.read_pickle(f)
    n = len(df)
    desc = ei[ei["event_id"].astype(str)==eid]
    edesc = str(desc["event_description"].iloc[0]) if not desc.empty else "?"
    
    print(f"\n{'-'*60}")
    print(f"EVENT {eid}: {edesc}")
    print(f"{'-'*60}")
    print(f"  Total samples: {n} ({n*10/60/24:.1f} days)")
    
    # Status analysis
    if "status_type_id" in df.columns:
        st = df["status_type_id"].value_counts().to_dict()
        last30 = df.iloc[-4320:]
        st_last = last30["status_type_id"].value_counts().to_dict()
        last7 = df.iloc[-1008:]
        st_7d = last7["status_type_id"].value_counts().to_dict()
        print(f"  Overall status: {st}")
        print(f"  Last 30d status: {st_last}")
        print(f"  Last 7d status: {st_7d}")
    
    # Power analysis
    if "power_2_avg" in df.columns:
        pw1 = df.iloc[:n//4]["power_2_avg"]
        pw2 = df.iloc[n//4:n//2]["power_2_avg"]
        pw3 = df.iloc[n//2:3*n//4]["power_2_avg"]
        pw4 = df.iloc[3*n//4:]["power_2_avg"]
        print(f"  Power quartiles: Q1={pw1.mean():.3f} Q2={pw2.mean():.3f} Q3={pw3.mean():.3f} Q4={pw4.mean():.3f}")
        # Zero power during high wind
        if "wind_speed_235_avg" in df.columns:
            last30_pw = df.iloc[-4320:]
            susp = ((last30_pw["power_2_avg"]<0.01) & (last30_pw["wind_speed_235_avg"]>0.05) & (last30_pw["status_type_id"]==0)).sum()
            print(f"  Suspicious zero-power in last 30d (wind>cut-in, status=0): {susp}")
    
    # Find ALL sensors with drift > 2 sigma in last 30 days vs first half
    print(f"\n  TOP 10 DRIFTING SENSORS (last 30d vs first half):")
    drifts = []
    first_half = df.iloc[:n//2]
    last_30d = df.iloc[-4320:]
    for c in df.columns:
        if "_avg" not in c: continue
        m1 = first_half[c].mean()
        s1 = first_half[c].std()
        m2 = last_30d[c].mean()
        if s1 > 0.001:
            d = abs(m2 - m1) / s1
            if d > 1.5:
                parts = c.replace("_avg","").replace("_max","").replace("_min","").replace("_std","")
                desc_row = feat[feat["sensor_name"]==parts]
                sensor_desc = str(desc_row["description"].iloc[0])[:50] if not desc_row.empty else "?"
                drifts.append((c, d, m1, m2, sensor_desc))
    drifts.sort(key=lambda x: -x[1])
    for c, d, m1, m2, desc in drifts[:10]:
        print(f"    {c}: {m1:.3f} -> {m2:.3f} ({d:.1f} sigma) - {desc}")
    
    # Check what our channels actually scored for this event
    # Specifically check the last 14 days (2016 samples) window scores
    print(f"\n  CHANNEL BEHAVIOR (last 14 days):")
    last14 = df.iloc[-2016:]
    # KCI
    if all(c in df.columns for c in ["sensor_130_avg","sensor_131_avg","sensor_132_avg"]):
        v = last14[["sensor_130_avg","sensor_131_avg","sensor_132_avg"]].values
        kci = (np.max(v,1)-np.min(v,1))/(np.mean(v,1)+1e-6)
        print(f"    KCI (current imbalance): mean={np.mean(kci):.4f}, max={np.max(kci):.4f}")
    if "sensor_48_avg" in df.columns:
        p = last14["sensor_48_avg"].values
        cov = np.std(p)/(np.abs(np.mean(p))+1e-6)
        print(f"    HPG (hydraulic CoV): {cov:.4f}")
    if "sensor_194_avg" in df.columns:
        print(f"    Rotor bearing: mean={last14['sensor_194_avg'].mean():.1f}°C, max={last14['sensor_194_avg'].max():.1f}°C")
    if "sensor_12_avg" in df.columns:
        print(f"    Battery current: mean={last14['sensor_12_avg'].mean():.3f}A")
    if "sensor_39_avg" in df.columns:
        print(f"    Cabinet temp: mean={last14['sensor_39_avg'].mean():.1f}°C")

# ═══════════════════════════════════════
# PART 2: LEAD TIME ANALYSIS — ALL TPs
# ═══════════════════════════════════════
print("\n\n" + "="*80)
print("PART 2: LEAD TIME ANALYSIS — ALL 22 TRUE POSITIVES")
print("="*80)

tps = res[res["type"]=="TP"].sort_values("lead_days")
print(f"\n{'Event':<8} {'Lead(d)':<10} {'First Ch':<18} {'#Ch':<5} {'Channels'}")
print("-"*80)
for _, r in tps.iterrows():
    print(f"{r['event_id']:<8} {r['lead_days']:<10} {r['first_ch']:<18} {r['n_ch']:<5} {r['channels']}")

print(f"\nLead Time Distribution:")
print(f"  < 7 days:  {len(tps[tps['lead_days']<7])} events (late detection)")
print(f"  7-14 days: {len(tps[(tps['lead_days']>=7)&(tps['lead_days']<14)])} events")
print(f"  14-30 days: {len(tps[(tps['lead_days']>=14)&(tps['lead_days']<30)])} events (good)")
print(f"  30-60 days: {len(tps[tps['lead_days']>=30])} events (excellent)")

# Check which channels detect EARLIEST vs LATEST
print(f"\nChannel Earliness Ranking (mean lead time when first):")
for ch in tps["first_ch"].unique():
    subset = tps[tps["first_ch"]==ch]
    print(f"  {ch}: {subset['lead_days'].mean():.1f}d (n={len(subset)})")

# ═══════════════════════════════════════
# PART 3: WHAT WOULD CATCH THE FNs?
# ═══════════════════════════════════════
print("\n\n" + "="*80)
print("PART 3: BRAINSTORMING — WHAT WOULD CATCH THE 5 FNs?")
print("="*80)

# For each FN, check if a simpler threshold on ANY sensor would have caught it
for eid in missed:
    f = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if not os.path.exists(f): continue
    df = pd.read_pickle(f)
    n = len(df)
    first_half = df.iloc[:n//2]
    
    # Check every avg sensor: does it cross 3-sigma threshold in last 30 days?
    catches = []
    for c in df.columns:
        if "_avg" not in c: continue
        m1 = first_half[c].mean()
        s1 = first_half[c].std()
        if s1 < 0.001: continue
        
        # Scan backwards from end to find FIRST window where 3-sigma is exceeded
        for day_offset in range(60, 1, -1):
            idx = n - int(day_offset * 144)
            if idx < 0: continue
            win = df.iloc[idx:idx+432]
            win_mean = win[c].mean()
            if abs(win_mean - m1) / s1 > 3.0:
                catches.append((c, day_offset, abs(win_mean-m1)/s1))
                break
    
    catches.sort(key=lambda x: -x[1])  # Sort by earliest detection
    if catches:
        print(f"\nEvent {eid}: Could be caught by simple 3 sigma threshold on:")
        for c, day, sigma in catches[:5]:
            desc_row = feat[feat["sensor_name"]==c.replace("_avg","")]
            sd = str(desc_row["description"].iloc[0])[:50] if not desc_row.empty else "?"
            print(f"  {c} at {day}d lead ({sigma:.1f} sigma) - {sd}")
    else:
        print(f"\nEvent {eid}: NO sensor exceeds 3σ in any 3-day window. Genuinely undetectable.")
