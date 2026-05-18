import pandas as pd, numpy as np, os

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
res = pd.read_csv(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\phase8_results.csv")

print("=== FP DEEP DIVE ===")
fps = res[res["type"]=="FP"]
for _, row in fps.iterrows():
    eid = str(row["event_id"])
    ld = row["lead_days"]
    ch = row["first_ch"]
    f = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if not os.path.exists(f): continue
    df = pd.read_pickle(f)
    n = len(df)
    alarm_idx = int(n - (ld * 24 * 6))
    win = df.iloc[max(0,alarm_idx-432):min(n,alarm_idx+432)]
    st = win["status_type_id"].value_counts().to_dict() if "status_type_id" in win.columns else {}
    pw = win["power_2_avg"].mean() if "power_2_avg" in win.columns else 0
    zero_pw = (win["power_2_avg"] < 0.01).sum() if "power_2_avg" in win.columns else 0
    ws = win["wind_speed_235_avg"].mean() if "wind_speed_235_avg" in win.columns else 0
    # Suspicious: wind > 3m/s but no power
    if "power_2_avg" in win.columns and "wind_speed_235_avg" in win.columns:
        susp = ((win["power_2_avg"]<0.01) & (win["wind_speed_235_avg"]>0.05)).sum()
    else:
        susp = 0
    print(f"Event {eid} (lead={ld}d, ch={ch}): status={st}")
    print(f"  power={pw:.3f}, zero_pw={zero_pw}/{len(win)}, wind={ws:.3f}, suspicious={susp}")

print("\n=== FN DEEP DIVE ===")
fns = res[res["type"]=="FN"]
for _, row in fns.iterrows():
    eid = str(row["event_id"])
    f = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if not os.path.exists(f): continue
    df = pd.read_pickle(f)
    n = len(df)
    last30 = df.iloc[-4320:]
    st = last30["status_type_id"].value_counts().to_dict() if "status_type_id" in last30.columns else {}
    pw1 = df.iloc[:n//2]["power_2_avg"].mean()
    pw2 = last30["power_2_avg"].mean()
    ws1 = df.iloc[:n//2]["wind_speed_235_avg"].mean() if "wind_speed_235_avg" in df.columns else 0
    ws2 = last30["wind_speed_235_avg"].mean() if "wind_speed_235_avg" in df.columns else 0

    # Check ALL sensor std changes for FN events (find ANY drifting sensor)
    print(f"Event {eid}: last30d_status={st}")
    print(f"  power: first_half={pw1:.3f}, last30d={pw2:.3f}")
    print(f"  wind: first_half={ws1:.3f}, last30d={ws2:.3f}")

    # Find top drifting sensors
    drifts = []
    for c in df.columns:
        if "_avg" not in c: continue
        m1 = df.iloc[:n//2][c].mean()
        s1 = df.iloc[:n//2][c].std()
        m2 = last30[c].mean()
        if s1 > 0.001:
            d = abs(m2 - m1) / s1
            if d > 1.0:
                drifts.append((c, d, m1, m2))
    drifts.sort(key=lambda x: -x[1])
    if drifts:
        print(f"  Top drifting sensors (>1 sigma):")
        for c, d, m1, m2 in drifts[:5]:
            print(f"    {c}: {m1:.3f} -> {m2:.3f} ({d:.2f} sigma)")
    else:
        print("  No sensors drifting > 1 sigma!")
    print()
