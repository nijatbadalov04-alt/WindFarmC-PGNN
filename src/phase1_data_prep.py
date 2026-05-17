import os
import pandas as pd
import glob

def main():
    DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\Data Sets"
    OUTPUT_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(csv_files)} datasets. Starting Phase 1 processing...")

    for file in csv_files:
        filename = os.path.basename(file)
        event_id = filename.split(".")[0]
        
        # Read the data
        try:
            df = pd.read_csv(file, sep=";")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        if "time_stamp" not in df.columns:
            print(f"Skipping {filename}: no time_stamp column")
            continue
            
        # Format and sort
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        df = df.sort_values(by="time_stamp").reset_index(drop=True)
        
        # Drop train_test if exists
        if "train_test" in df.columns:
            df = df.drop(columns=["train_test"])
            
        # Split 50/50
        midpoint = len(df) // 2
        part1 = df.iloc[:midpoint].copy()
        part2 = df.iloc[midpoint:].copy()
        
        # Filter Part 1 for normal statuses (0 and 2)
        # Handle different potential names for the status column
        status_col = None
        if "status_type_id" in df.columns:
            status_col = "status_type_id"
        elif "status_type" in df.columns:
            status_col = "status_type"
            
        if status_col:
            part1 = part1[part1[status_col].isin([0, 2])]
        else:
            print(f"Warning: Status column not found in {filename}")
            
        # Save the splits using pickle for very fast I/O in the next phases
        out_file1 = os.path.join(OUTPUT_DIR, f"event_{event_id}_part1.pkl")
        out_file2 = os.path.join(OUTPUT_DIR, f"event_{event_id}_part2.pkl")
        
        part1.to_pickle(out_file1)
        part2.to_pickle(out_file2)
        
        print(f"Processed Event {event_id:>3}: Part1={len(part1):>5} rows (filtered), Part2={len(part2):>5} rows")

    print("\n[SUCCESS] Phase 1 Data Preparation Complete.")

if __name__ == "__main__":
    main()
