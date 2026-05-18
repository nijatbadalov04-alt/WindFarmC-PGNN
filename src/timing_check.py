import pandas as pd, numpy as np, os

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
ei = pd.read_csv(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv", sep=";")

# CARE paper says: the PREDICTION time frame includes padding before and after.
# The EVENT is a sub-interval within Part 2. Check timing.

for eid in [11, 12, 35, 70, 9, 31]:
    f2 = os.path.join(DATA, f"event_{eid}_part2.pkl")
    df2 = pd.read_pickle(f2)
    ev = ei[ei["event_id"]==eid].iloc[0]
    esi = int(ev["event_start_id"]) if pd.notna(ev["event_start_id"]) else None
    eei = int(ev["event_end_id"]) if pd.notna(ev["event_end_id"]) else None
    
    rows = df2["id"].values
    total = len(df2)
    
    si = None; ei_ = None
    if esi:
        m = rows >= esi
        if m.any(): si = int(m.argmax())
    if eei:
        m = rows >= eei
        if m.any(): ei_ = int(m.argmax())
    
    pct_before = si/total*100 if si else 0
    evt_len = (ei_-si) if (si and ei_) else 0
    days_before = si * 10 / (60*24) if si else 0
    evt_days = evt_len * 10 / (60*24)
    days_after = (total - (ei_ if ei_ else total)) * 10 / (60*24)
    
    desc = ev["event_description"][:80] if pd.notna(ev["event_description"]) else "N/A"
    
    print(f"Event {eid}: total={total} ({total*10/60/24:.0f}d)")
    print(f"  Event start idx={si} ({days_before:.0f}d from start)")
    print(f"  Event end idx={ei_} (length={evt_days:.0f}d)")
    print(f"  Padding after: {days_after:.0f}d")
    print(f"  Desc: {desc}")
    print()
