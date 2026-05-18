"""
Investigation: For each TP, does the CORRECT physics channel fire?
Two-stage approach: (1) Early warning = any channel, (2) Fault ID = matching channel
"""
import pandas as pd, numpy as np, os, re

RESULTS = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\phase10_results.csv"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
res = pd.read_csv(RESULTS)
tps = res[res["type"]=="TP"]

# Fault keyword -> expected channels mapping
FAULT_TO_CHANNELS = {
    "gearbox": ["gearbox", "drift_pump", "gear_oil_pump", "hpg"],
    "gear": ["gearbox", "drift_pump", "gear_oil_pump", "hpg"],
    "oil": ["gearbox", "drift_pump", "gear_oil_pump", "hpg", "hydraulic"],
    "hydraulic": ["hydraulic", "hpg", "drift_pump"],
    "generator": ["generator", "kci"],
    "stator": ["generator", "kci"],
    "transformer": ["transformer"],
    "pitch": ["pitch", "battery_current", "drift_batt", "battery_axis2"],
    "battery": ["battery_current", "drift_batt", "battery_axis2"],
    "dc-link": ["battery_current", "drift_batt", "drift_freq", "drift_abb", "mains_frequency"],
    "converter": ["drift_hvrp", "hv_reactive", "kci", "rpfd", "drift_freq"],
    "slip ring": ["nacelle_24v_current", "drift_24v", "battery_current", "drift_batt"],
    "communication": ["nacelle_24v_current", "drift_24v"],
    "beckhoff": ["nacelle_24v_current", "drift_24v", "pitch"],
    "bearing": ["rotor_bearing"],
    "cabinet": ["cabinet_temp"],
    "filter": ["drift_filter", "aeration_filter"],
    "voltage": ["drift_abb", "abb_voltage_l1", "drift_freq", "mains_frequency"],
    "current": ["kci", "nacelle_24v_current", "drift_24v", "battery_current"],
    "cooling": ["drift_filter", "cabinet_temp"],
    "brake": ["hydraulic", "hpg"],
    "yaw": ["hydraulic"],
    "grease": ["hydraulic", "hpg"],
    "wiring": ["nacelle_24v_current", "drift_24v", "kci"],
    "fuse": ["drift_filter", "aeration_filter", "cabinet_temp"],
    "pump": ["drift_pump", "gear_oil_pump", "hpg", "hydraulic"],
    "valve": ["hydraulic", "hpg"],
    "accumulator": ["hydraulic", "hpg"],
}

print("="*100)
print("FAULT CLASSIFICATION: Does the CORRECT channel fire for each event?")
print("="*100)
print(f"\n{'Event':<7} {'Actual Fault':<55} {'Early Warning':<20} {'Matching Channels Found'}")
print("-"*130)

correct_also = 0
correct_only = 0
no_match = 0

for _, r in tps.sort_values("event_id").iterrows():
    eid = str(int(r["event_id"]))
    desc_row = ei[ei["event_id"].astype(str)==eid]
    actual = str(desc_row["event_description"].iloc[0]) if not desc_row.empty else "Unknown"
    actual_lower = actual.lower()
    
    first_ch = r["first_ch"]
    all_chs = str(r["channels"]).split(",")
    
    # Find which channels MATCH the actual fault
    matching = []
    for keyword, expected_chs in FAULT_TO_CHANNELS.items():
        if keyword in actual_lower:
            for ch in all_chs:
                if ch in expected_chs and ch not in matching:
                    matching.append(ch)
    
    # Check if first channel is in matching list
    first_matches = first_ch in matching
    
    if matching:
        if first_matches:
            correct_only += 1
            status = "FIRST=CORRECT"
        else:
            correct_also += 1
            status = "CORRECT FIRES LATER"
    else:
        no_match += 1
        status = "NO MATCH IN CHANNELS"
    
    match_str = ", ".join(matching) if matching else "NONE"
    print(f"{eid:<7} {actual[:54]:<55} {first_ch:<20} {match_str} [{status}]")

print(f"\n{'='*100}")
print(f"RESULTS:")
print(f"  First channel IS the correct one: {correct_only}/27 ({correct_only/27*100:.1f}%)")
print(f"  Correct channel fires but later:  {correct_also}/27 ({correct_also/27*100:.1f}%)")
print(f"  No matching channel fires at all: {no_match}/27 ({no_match/27*100:.1f}%)")
print(f"  TOTAL with correct channel:       {correct_only+correct_also}/27 ({(correct_only+correct_also)/27*100:.1f}%)")

print(f"\n{'='*100}")
print(f"TWO-STAGE DETECTION CAPABILITY:")
print(f"  Stage 1 (EARLY WARNING): 'Something is wrong' -> 27/27 detected (100%)")
print(f"  Stage 2 (FAULT ID):      'Here is what is wrong' -> {correct_only+correct_also}/27 correctly identified")

# Now show: for the "no match" events, what are they and why?
print(f"\n{'='*100}")
print(f"EVENTS WHERE NO CORRECT CHANNEL FIRES:")
print(f"{'='*100}")
for _, r in tps.sort_values("event_id").iterrows():
    eid = str(int(r["event_id"]))
    desc_row = ei[ei["event_id"].astype(str)==eid]
    actual = str(desc_row["event_description"].iloc[0]) if not desc_row.empty else "?"
    actual_lower = actual.lower()
    
    all_chs = str(r["channels"]).split(",")
    matching = []
    for keyword, expected_chs in FAULT_TO_CHANNELS.items():
        if keyword in actual_lower:
            for ch in all_chs:
                if ch in expected_chs and ch not in matching:
                    matching.append(ch)
    
    if not matching:
        print(f"\n  Event {eid}: {actual[:80]}")
        print(f"    Channels that fired: {', '.join(all_chs)}")
        # Why does nothing match? Check what keywords are in the description
        found_kw = [k for k in FAULT_TO_CHANNELS if k in actual_lower]
        if found_kw:
            expected = set()
            for k in found_kw:
                expected.update(FAULT_TO_CHANNELS[k])
            print(f"    Expected channels (from keywords '{', '.join(found_kw)}'): {', '.join(expected)}")
            print(f"    Missing: {', '.join(expected - set(all_chs))}")
        else:
            print(f"    No recognized fault keywords in description")
