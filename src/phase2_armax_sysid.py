import os
import glob
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def create_lagged_features(df, target, exo, n_lags=3):
    X, y = [], []
    t_vals = df[target].values
    e_vals = df[exo].values
    for i in range(n_lags, len(df)):
        t_lags = t_vals[i-n_lags:i]
        e_lags = e_vals[i-n_lags:i].flatten()
        X.append(np.concatenate([t_lags, e_lags]))
        y.append(t_vals[i])
    return np.array(X), np.array(y)

def main():
    DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
    MODEL_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\armax"
    os.makedirs(MODEL_DIR, exist_ok=True)

    TARGETS = {
        "gearbox": "sensor_186_avg",     # Gearbox oil temperature 1
        "transformer": "sensor_191_avg", # Oil temperature 1 main transformer
        "hydraulic": "sensor_178_avg",   # Hydraulic oil tank temperature 1
        "generator": "sensor_173_avg",   # Generator temperature 1 cooling air inlet
        "pitch": "sensor_62_avg"         # Temperature Axis 1
    }
    EXOGENOUS = ["sensor_7_avg", "sensor_144_avg", "power_2_avg"]

    part1_files = glob.glob(os.path.join(DATA_DIR, "*_part1.pkl"))
    print(f"Loading {len(part1_files)} training datasets...")

    dfs = []
    for f in part1_files:
        df = pd.read_pickle(f)
        req_cols = list(TARGETS.values()) + EXOGENOUS
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            print(f"Warning: {os.path.basename(f)} missing {missing}. Skipping.")
            continue
            
        # forward/backward fill in case of minor sensor drops, then drop if still nan
        df_sub = df[req_cols].ffill().bfill().dropna()
        if len(df_sub) > 0:
            dfs.append(df_sub)

    if not dfs:
        print("No valid data found to train ARMAX.")
        return

    master_df = pd.concat(dfs, ignore_index=True)
    print(f"Global training pool size: {len(master_df)} samples")

    scaler = StandardScaler()
    scaler.fit(master_df)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "global_scaler.pkl"))
    print("Saved global scaler.")

    config = {"n_lags": 3, "targets": TARGETS, "exogenous": EXOGENOUS}
    with open(os.path.join(MODEL_DIR, "armax_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Train per subsystem
    for name, target_col in TARGETS.items():
        print(f"\nTraining ARMAX for {name} ({target_col})...")
        X_all, y_all = [], []
        
        for df in dfs:
            if len(df) <= 3: continue
            sc = scaler.transform(df)
            df_sc = pd.DataFrame(sc, columns=df.columns)
            X, y = create_lagged_features(df_sc, target_col, EXOGENOUS, n_lags=3)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
            
        if not X_all:
            print(f"No valid lagged features for {name}.")
            continue
            
        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_all)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        np.save(os.path.join(MODEL_DIR, f"{name}_theta.npy"), model.coef_)
        print(f"  Saved {name}_theta.npy (shape {model.coef_.shape}). R^2 = {model.score(X_train, y_train):.4f}")

    print("\n[SUCCESS] Phase 2 ARMAX System Identification Complete.")

if __name__ == "__main__":
    main()
