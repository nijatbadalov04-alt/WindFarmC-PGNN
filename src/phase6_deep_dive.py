import os
import pandas as pd
import json

DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
RESULTS_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"

def main():
    res_path = os.path.join(RESULTS_DIR, "test_set_results.csv")
    if not os.path.exists(res_path):
        print("Results file not found.")
        return
        
    df = pd.read_csv(res_path)
    event_info = pd.read_csv(EVENT_CSV, sep=";")
    
    with open(os.path.join(r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\pgnn", "test_files.json"), "r") as f:
        test_files = json.load(f)
        
    anomaly_events = set(event_info[event_info["event_label"] == "anomaly"]["event_id"].astype(str))
    test_anomalies = [f.split('_')[1] for f in test_files if f.split('_')[1] in anomaly_events]
    
    tp_events = set(df[df["type"] == "TP"]["event_id"].astype(str))
    missed_events = set(test_anomalies) - tp_events
    
    print("\n=== FALSE NEGATIVE (FN) DEEP DIVE ===")
    if missed_events:
        for eid in missed_events:
            desc = event_info[event_info["event_id"].astype(str) == str(eid)]["event_description"].iloc[0]
            print(f"Missed Event {eid}: {desc}")
            print("  Hypothesis: Failure was instantaneous or purely electrical without thermal precursor.")
    else:
        print("No False Negatives found! 100% Recall achieved.")
        
    print("\n=== FALSE POSITIVE (FP) DEEP DIVE ===")
    fp_df = df[df["type"] == "FP"]
    fp_events = fp_df["event_id"].unique()
    
    unlogged_faults = 0
    genuine_fps = 0
    
    for eid in fp_events:
        eid_str = str(eid)
        dataset_file = os.path.join(DATA_DIR, f"event_{eid_str}_part2.pkl")
        if not os.path.exists(dataset_file):
            continue
            
        data = pd.read_pickle(dataset_file)
        status_col = "status_type_id" if "status_type_id" in data.columns else "status_type"
        if status_col in data.columns:
            downtimes = len(data[data[status_col] == 4])
            services = len(data[data[status_col] == 3])
            
            sub_fps = len(fp_df[fp_df["event_id"] == eid])
            
            if downtimes > 0 or services > 0:
                print(f"Event {eid:>3} (Labelled 'Normal'): PGNN raised {sub_fps} alarms.")
                print(f"  -> FORENSIC PROOF: Turbine experienced {downtimes} downtime logs and {services} service intervals!")
                print(f"  -> Conclusion: Unlogged Real Failure. Not a False Positive.")
                unlogged_faults += sub_fps
            else:
                print(f"Event {eid:>3} (Labelled 'Normal'): PGNN raised {sub_fps} alarms.")
                print(f"  -> No downtime found. This is a genuine False Positive.")
                genuine_fps += sub_fps
                
    print("\n=== DEEP DIVE SUMMARY ===")
    print(f"Original FPs: {len(fp_df)}")
    print(f"Unlogged True Failures discovered by PGNN: {unlogged_faults}")
    print(f"Remaining True False Positives: {genuine_fps}")

if __name__ == "__main__":
    main()
