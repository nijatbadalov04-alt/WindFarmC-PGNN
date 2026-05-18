"""
Deep reliability investigation:
For each 'WRONG' event, verify if the cascade is real by checking:
1. Do the EXPECTED sensors show ANY anomaly (even below threshold)?
2. Does the timeline of channel firing make physical cascade sense?
3. Is the first-firing channel's sensor actually related to the fault?
"""
import pandas as pd, numpy as np, os

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
FEAT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\feature_description.csv"
RESULTS = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\phase10_results.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
feat = pd.read_csv(FEAT_CSV, sep=";")
res = pd.read_csv(RESULTS)

# Fault -> expected sensors (direct physical connection)
FAULT_EXPECTED_SENSORS = {
    "hydraulic": ["sensor_48_avg", "sensor_74_avg", "sensor_178_avg"],  # pressure, oil level, temp
    "pitch": ["sensor_62_avg", "sensor_12_avg", "sensor_13_avg", "sensor_14_avg"],  # pitch temp, battery currents
    "gearbox": ["sensor_186_avg", "sensor_87_avg"],  # gearbox temp, oil pump current
    "generator": ["sensor_173_avg", "sensor_130_avg", "sensor_131_avg", "sensor_132_avg"],  # gen temp, currents
    "converter": ["sensor_75_avg", "sensor_47_avg", "sensor_58_avg"],  # reactive pwr, mains freq, ABB voltage
    "bearing": ["sensor_194_avg", "sensor_195_avg"],  # rotor bearing temps
    "cooling": ["sensor_39_avg", "sensor_109_avg", "sensor_110_avg"],  # cabinet temp, aeration filters
    "battery": ["sensor_12_avg", "sensor_13_avg", "sensor_14_avg"],  # battery discharge currents
    "communication": ["sensor_25_avg"],  # 24V current (powers comm systems)
    "oil": ["sensor_48_avg", "sensor_74_avg", "sensor_87_avg", "sensor_44_avg", "sensor_45_avg"],
    "valve": ["sensor_48_avg", "sensor_74_avg"],
    "brake": ["sensor_48_avg", "sensor_178_avg"],
    "pump": ["sensor_87_avg", "sensor_48_avg", "sensor_74_avg"],
}

# Events where no matching channel fires (the 16 "WRONG" ones)
INVESTIGATE = ['4','11','18','28','31','33','35','39','44','47','55','78','79','81','90','91']

print("="*100)
print("DEEP RELIABILITY CHECK: Are the cascade explanations real?")
print("="*100)

for eid in INVESTIGATE:
    f = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if not os.path.exists(f): continue
    df = pd.read_pickle(f)
    n = len(df)
    fh = df.iloc[:n//2]  # baseline
    
    desc_row = ei[ei["event_id"].astype(str)==eid]
    actual = str(desc_row["event_description"].iloc[0]) if not desc_row.empty else "?"
    actual_lower = actual.lower()
    
    # Get the result for this event
    rr = res[(res["event_id"].astype(str)==eid) & (res["type"]=="TP")]
    if rr.empty: continue
    first_ch = rr.iloc[0]["first_ch"]
    all_chs = str(rr.iloc[0]["channels"]).split(",")
    lead = rr.iloc[0]["lead_days"]
    
    print(f"\n{'='*80}")
    print(f"EVENT {eid}: {actual[:75]}")
    print(f"First channel: {first_ch} at {lead:.0f}d lead | All: {', '.join(all_chs)}")
    print(f"{'='*80}")
    
    # Find expected sensors based on fault description keywords
    expected_sensors = set()
    found_keywords = []
    for keyword, sensors in FAULT_EXPECTED_SENSORS.items():
        if keyword in actual_lower:
            expected_sensors.update(sensors)
            found_keywords.append(keyword)
    
    if not expected_sensors:
        print(f"  No recognized fault keywords -> cannot verify cascade")
        print(f"  Description keywords: {[w for w in actual_lower.split() if len(w)>3][:10]}")
        continue
    
    print(f"  Keywords found: {found_keywords}")
    print(f"  Expected sensors: {list(expected_sensors)[:8]}")
    
    # CHECK 1: Do expected sensors show ANY anomaly?
    print(f"\n  [CHECK 1] Do expected sensors show anomaly? (even below 3-sigma threshold)")
    any_anomaly = False
    for s in sorted(expected_sensors):
        if s not in df.columns: 
            print(f"    {s}: NOT IN DATA")
            continue
        bl_m = fh[s].mean()
        bl_s = fh[s].std()
        if bl_s < 0.001: 
            print(f"    {s}: Zero variance (constant)")
            continue
        
        # Check last 30 days
        last30 = df.iloc[-4320:]
        l30_m = last30[s].mean()
        sigma = abs(l30_m - bl_m) / bl_s
        
        # Check window-by-window: when does it first exceed 2-sigma?
        W = 432; S = 72
        first_2sig = None; first_3sig = None; max_sigma = 0
        for wi in range(len(range(0, n-W+1, S))):
            st = wi * S; end = st + W
            if end > n: break
            wm = df.iloc[st:end][s].mean()
            ws = abs(wm - bl_m) / bl_s
            if ws > max_sigma: max_sigma = ws
            if ws > 2.0 and first_2sig is None:
                nw_total = len(range(0, n-W+1, S))
                first_2sig = (nw_total - wi) * S * 10 / (60*24)
            if ws > 3.0 and first_3sig is None:
                nw_total = len(range(0, n-W+1, S))
                first_3sig = (nw_total - wi) * S * 10 / (60*24)
        
        status = "OK"
        if max_sigma > 3.0: 
            status = f"ANOMALY 3sig at {first_3sig:.0f}d"
            any_anomaly = True
        elif max_sigma > 2.0: 
            status = f"WEAK at {first_2sig:.0f}d"
            any_anomaly = True
        elif max_sigma > 1.5: 
            status = "MARGINAL"
        else: 
            status = "NORMAL"
        
        # Sensor description
        sname = s.replace("_avg","")
        dr = feat[feat["sensor_name"]==sname]
        sdesc = str(dr["description"].iloc[0])[:40] if not dr.empty else "?"
        
        print(f"    {s} ({sdesc}): last30={sigma:.1f}sig, max={max_sigma:.1f}sig -> {status}")
    
    # CHECK 2: What does the FIRST channel's sensor actually show?
    print(f"\n  [CHECK 2] What did first channel '{first_ch}' actually see?")
    # Map channel to its primary sensor
    CH_TO_SENSOR = {
        "drift_batt": "sensor_13_avg", "drift_24v": "sensor_25_avg",
        "drift_freq": "sensor_47_avg", "drift_abb": "sensor_58_avg",
        "drift_hvrp": "sensor_75_avg", "drift_filter": "sensor_109_avg",
        "drift_pump": "sensor_87_avg", "mains_frequency": "sensor_47_avg",
        "abb_voltage_l1": "sensor_58_avg", "ensemble": None,
        "generator": "sensor_173_avg", "gearbox": "sensor_186_avg",
        "pitch": "sensor_62_avg", "hydraulic": "sensor_178_avg",
        "transformer": "sensor_191_avg", "kci": "sensor_130_avg",
        "hpg": "sensor_48_avg", "rotor_bearing": "sensor_194_avg",
        "battery_current": "sensor_12_avg", "cabinet_temp": "sensor_39_avg",
    }
    first_sensor = CH_TO_SENSOR.get(first_ch)
    if first_sensor and first_sensor in df.columns:
        bl_m = fh[first_sensor].mean()
        bl_s = fh[first_sensor].std()
        if bl_s > 0.001:
            last30_m = df.iloc[-4320:][first_sensor].mean()
            sigma = abs(last30_m - bl_m) / bl_s
            sname = first_sensor.replace("_avg","")
            dr = feat[feat["sensor_name"]==sname]
            sdesc = str(dr["description"].iloc[0])[:40] if not dr.empty else "?"
            print(f"    Sensor: {first_sensor} ({sdesc})")
            print(f"    Drift: {sigma:.1f} sigma from baseline")
    elif first_ch == "ensemble":
        print(f"    Ensemble: multiple channels voting together (no single sensor)")
    
    # CHECK 3: Is the cascade physically plausible?
    print(f"\n  [CHECK 3] Cascade plausibility:")
    if not any_anomaly:
        print(f"    RESULT: Expected sensors show NO anomaly at all")
        print(f"    -> The '{first_ch}' detection may be COINCIDENTAL, not cascaded")
        print(f"    -> RELIABILITY: LOW - this detection may not be physically connected to the fault")
    else:
        print(f"    RESULT: Expected sensors DO show anomaly (at least weak signal)")
        print(f"    -> Physical cascade is PLAUSIBLE")
        print(f"    -> RELIABILITY: MODERATE - detection is real but via indirect pathway")

print(f"\n\n{'='*100}")
print("OVERALL RELIABILITY ASSESSMENT")
print("="*100)
