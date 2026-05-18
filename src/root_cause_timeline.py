"""
DEEP ROOT CAUSE TIMELINE ANALYSIS
For EVERY TP event: trace the sensor timeline, show when each anomaly starts,
verify if detection channel is physically connected to the actual fault.
"""
import pandas as pd, numpy as np, os

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
FEAT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\feature_description.csv"
RESULTS = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\phase10_results.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
feat = pd.read_csv(FEAT_CSV, sep=";")
res = pd.read_csv(RESULTS)
tps = res[res["type"]=="TP"].sort_values("event_id")

def get_desc(sensor_name):
    sn = sensor_name.replace("_avg","").replace("_max","").replace("_min","")
    r = feat[feat["sensor_name"]==sn]
    return str(r["description"].iloc[0])[:50] if not r.empty else "?"

def timeline_analysis(df, sensors, n_total):
    """For each sensor, find: when drift starts (>2sig), when it peaks, direction, magnitude."""
    n = len(df)
    fh = df.iloc[:n//2]
    results = []
    W = 432; S = 72
    nw = len(range(0, n-W+1, S))
    
    for s in sensors:
        if s not in df.columns: continue
        bl_m = fh[s].mean()
        bl_s = fh[s].std()
        if bl_s < 0.001: continue
        
        # Window-by-window analysis
        first_2sig_wi = None; first_3sig_wi = None
        max_sigma = 0; max_wi = 0
        timeline = []
        for wi in range(nw):
            st = wi * S; end = st + W
            if end > n: break
            wm = df.iloc[st:end][s].mean()
            sigma = (wm - bl_m) / bl_s  # signed
            abs_sigma = abs(sigma)
            timeline.append(sigma)
            if abs_sigma > max_sigma:
                max_sigma = abs_sigma; max_wi = wi
            if abs_sigma > 2.0 and first_2sig_wi is None:
                first_2sig_wi = wi
            if abs_sigma > 3.0 and first_3sig_wi is None:
                first_3sig_wi = wi
        
        if max_sigma < 1.0: continue  # skip totally normal sensors
        
        lead_2sig = (nw - first_2sig_wi) * S * 10 / (60*24) if first_2sig_wi else None
        lead_3sig = (nw - first_3sig_wi) * S * 10 / (60*24) if first_3sig_wi else None
        direction = "UP" if timeline[-1] > 0 else "DOWN" if timeline[-1] < 0 else "FLAT"
        
        # Check if trend is sustained or intermittent
        if first_3sig_wi:
            above_3 = sum(1 for t in timeline[first_3sig_wi:] if abs(t) > 3.0)
            pct_above = above_3 / max(1, nw - first_3sig_wi) * 100
        else:
            pct_above = 0
        
        results.append({
            "sensor": s,
            "desc": get_desc(s),
            "max_sigma": max_sigma,
            "direction": direction,
            "lead_2sig": lead_2sig,
            "lead_3sig": lead_3sig,
            "sustained_pct": pct_above,
            "final_sigma": abs(timeline[-1]) if timeline else 0,
        })
    
    results.sort(key=lambda x: -(x["lead_2sig"] or 0))
    return results

# Key sensor groups for physical systems
SYSTEM_SENSORS = {
    "Pitch System": ["sensor_62_avg","sensor_12_avg","sensor_13_avg","sensor_14_avg"],
    "Gearbox": ["sensor_186_avg","sensor_87_avg","sensor_44_avg","sensor_45_avg"],
    "Generator": ["sensor_173_avg","sensor_130_avg","sensor_131_avg","sensor_132_avg"],
    "Hydraulic": ["sensor_48_avg","sensor_74_avg","sensor_178_avg"],
    "Transformer": ["sensor_191_avg"],
    "Electrical Bus": ["sensor_25_avg","sensor_47_avg","sensor_58_avg","sensor_59_avg","sensor_60_avg"],
    "Converter/Power": ["sensor_75_avg","sensor_109_avg","sensor_110_avg"],
    "Bearing": ["sensor_194_avg","sensor_195_avg"],
    "Cabinet/Cooling": ["sensor_39_avg","sensor_167_avg"],
    "Battery": ["sensor_12_avg","sensor_13_avg","sensor_14_avg"],
}

ALL_KEY_SENSORS = list(set(s for sensors in SYSTEM_SENSORS.values() for s in sensors))

print("="*110)
print("COMPREHENSIVE ROOT CAUSE TIMELINE ANALYSIS - ALL 27 TP EVENTS")
print("="*110)

for _, r in tps.iterrows():
    eid = str(int(r["event_id"]))
    f = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if not os.path.exists(f): continue
    df = pd.read_pickle(f)
    n = len(df)
    days = n * 10 / (60*24)
    
    desc_row = ei[ei["event_id"].astype(str)==eid]
    actual = str(desc_row["event_description"].iloc[0]) if not desc_row.empty else "?"
    first_ch = r["first_ch"]
    all_chs = str(r["channels"]).split(",")
    lead = r["lead_days"]
    n_ch = int(r["n_ch"])
    
    print(f"\n{'#'*110}")
    print(f"EVENT {eid}: {actual[:100]}")
    print(f"Detection: {first_ch} at {lead:.0f}d | {n_ch} channels | Data: {days:.0f} days")
    print(f"All channels: {', '.join(all_chs)}")
    print(f"{'#'*110}")
    
    # Run timeline on ALL key sensors
    tl = timeline_analysis(df, ALL_KEY_SENSORS, n)
    
    if not tl:
        print("  No significant sensor drift detected")
        continue
    
    # Group by system
    print(f"\n  TIMELINE: When each sensor starts drifting (sorted by earliest)")
    print(f"  {'Sensor':<25} {'Description':<45} {'Dir':<5} {'Max σ':>7} {'First>2σ':>10} {'First>3σ':>10} {'Sustained':>10} {'Now':>6}")
    print(f"  {'-'*120}")
    
    for t in tl[:25]:  # top 25 most anomalous
        l2 = f"{t['lead_2sig']:.0f}d" if t['lead_2sig'] else "-"
        l3 = f"{t['lead_3sig']:.0f}d" if t['lead_3sig'] else "-"
        sus = f"{t['sustained_pct']:.0f}%" if t['lead_3sig'] else "-"
        print(f"  {t['sensor']:<25} {t['desc']:<45} {t['direction']:<5} {t['max_sigma']:>6.1f} {l2:>10} {l3:>10} {sus:>10} {t['final_sigma']:>5.1f}")
    
    # Identify which SYSTEM is degrading first
    print(f"\n  SYSTEM-LEVEL FAULT PROPAGATION:")
    system_first = {}
    for sname, sensors in SYSTEM_SENSORS.items():
        earliest = None
        for t in tl:
            if t["sensor"] in sensors and t["lead_2sig"]:
                if earliest is None or t["lead_2sig"] > earliest:
                    earliest = t["lead_2sig"]
        if earliest:
            system_first[sname] = earliest
    
    for sname, lead_d in sorted(system_first.items(), key=lambda x: -x[1]):
        marker = " <-- FIRST" if lead_d == max(system_first.values()) else ""
        print(f"    {sname:<25} anomaly begins at {lead_d:.0f}d before failure{marker}")
    
    # ROOT CAUSE VERDICT
    print(f"\n  ROOT CAUSE ASSESSMENT:")
    if system_first:
        first_system = max(system_first, key=system_first.get)
        first_lead = system_first[first_system]
        
        # Check if first_system matches the fault description
        actual_lower = actual.lower()
        match_keywords = {
            "Pitch System": ["pitch","blade","axis"],
            "Gearbox": ["gear","coupling"],
            "Generator": ["generator","stator"],
            "Hydraulic": ["hydraulic","pump","brake","valve","oil","accumulator"],
            "Transformer": ["transformer"],
            "Electrical Bus": ["communication","24v","voltage","current","slip ring","beckhoff","wiring"],
            "Converter/Power": ["converter","reactive","filter","fuse"],
            "Bearing": ["bearing"],
            "Cabinet/Cooling": ["cabinet","cooling","temperature"],
            "Battery": ["battery","dc-link"],
        }
        
        desc_matches = []
        for sys_name, keywords in match_keywords.items():
            if any(k in actual_lower for k in keywords):
                desc_matches.append(sys_name)
        
        if first_system in desc_matches:
            print(f"    ✅ CORRECT: First degrading system ({first_system} at {first_lead:.0f}d) MATCHES fault '{actual[:50]}'")
        elif desc_matches and any(d in system_first for d in desc_matches):
            matching_sys = [d for d in desc_matches if d in system_first]
            for ms in matching_sys:
                print(f"    ⚡ CASCADE: Fault system ({ms}) starts at {system_first[ms]:.0f}d, but {first_system} degrades first at {first_lead:.0f}d")
                print(f"       -> {first_system} stressed {first_lead - system_first[ms]:.0f}d BEFORE {ms} → physical cascade confirmed")
        elif desc_matches:
            print(f"    ❌ MISMATCH: Expected {desc_matches} but first degrading is {first_system}")
            print(f"       Expected sensors show NO anomaly → detection may be unrelated")
        else:
            print(f"    ❓ UNVERIFIABLE: Fault description doesn't map to known sensor systems")
            print(f"       First degrading system: {first_system} at {first_lead:.0f}d")

print(f"\n\n{'='*110}")
print("DONE - All 27 events analyzed")
print("="*110)
