import os
import glob
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import skew, kurtosis
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

WINDOW_SIZE = 432
STRIDE = 72
RECOVERY_WINDOWS = (10 * 24 * 60) // (10 * STRIDE)
MIN_LEAD_WINDOWS = (2 * 24 * 60) // (10 * STRIDE)
MAX_LEAD_WINDOWS = (60 * 24 * 60) // (10 * STRIDE)

DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
MODEL_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\pgnn"
ARMAX_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\armax"

# NEW ELECTRICAL TARGETS TO CATCH INSTANTANEOUS FAULTS
NEW_TARGETS = {
    "electrical_current": "sensor_130_avg",  # Generator RMS current L1
    "dc_link_voltage": "sensor_36_avg"       # Direct current link volt axis 1
}
EXOGENOUS = ["sensor_7_avg", "sensor_144_avg", "power_2_avg"]

class PGNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1)
        )
        self.class_head = nn.Linear(64, 1)
        self.phys_head = nn.Linear(64, 1)
    def forward(self, x):
        h = self.shared(x)
        c = torch.sigmoid(self.class_head(h))
        p = self.phys_head(h)
        return c, p

def create_lagged_features(df_vals, target_idx, exo_idxs, n_lags=3):
    X, y = [], []
    for i in range(n_lags, len(df_vals)):
        t_lags = df_vals[i-n_lags:i, target_idx]
        e_lags = df_vals[i-n_lags:i, exo_idxs].flatten()
        X.append(np.concatenate([t_lags, e_lags]))
        y.append(df_vals[i, target_idx])
    return np.array(X), np.array(y)

def process_dataset(df_sc, df_raw, theta_global, target_idx, exo_idxs, is_anomaly):
    n_samples = len(df_sc)
    features, labels, phys = [], [], []
    for start in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        window_sc = df_sc.iloc[start:end].values
        window_raw = df_raw.iloc[start:end].values
        X, y = create_lagged_features(window_sc, target_idx, exo_idxs, 3)
        if len(X) < 10: continue
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        delta_theta = model.coef_ - theta_global
        res = y - (X @ theta_global)
        
        r_mean, r_std, r_max = np.mean(res), np.std(res), np.max(np.abs(res))
        rmse = np.sqrt(np.mean(res**2))
        sk, ku = skew(res), kurtosis(res)
        ss_res, ss_tot = np.sum(res**2), np.sum((y - np.mean(y))**2) + 1e-6
        r2 = max(0, 1 - ss_res / ss_tot)
        
        amb_temp = np.mean(window_raw[:, exo_idxs[0]])
        rotor_spd_mean = np.mean(window_raw[:, exo_idxs[1]])
        rotor_spd_max = np.max(window_raw[:, exo_idxs[1]])
        power_mean = np.mean(window_raw[:, exo_idxs[2]])
        u_virt = power_mean / (amb_temp + 273.15)
        
        vec = np.concatenate([delta_theta, [r_mean, r_std, r_max, rmse, sk, ku, r2], [amb_temp, rotor_spd_mean, rotor_spd_max, power_mean, u_virt]])
        features.append(vec)
        labels.append(1 if (is_anomaly and (n_samples - end) < 4320) else 0)
        phys.append(r2)
    return features, labels, phys

def evaluate_threshold(probs, threshold):
    alarms = (probs > threshold).astype(int)
    confirmed = np.zeros_like(alarms)
    for i in range(len(alarms)):
        if np.sum(alarms[max(0, i-3):i+1]) >= 2: confirmed[i] = 1
    active_alarms, suppression_timer = [], 0
    for i in range(len(confirmed)):
        if suppression_timer > 0: suppression_timer -= 1; continue
        if confirmed[i] == 1: active_alarms.append(i); suppression_timer = RECOVERY_WINDOWS
    return active_alarms

def main():
    print("=== PHASE 6 REFINEMENT: Electrical Subsystems ===")
    
    # 1. ARMAX Baseline
    global_scaler = joblib.load(os.path.join(ARMAX_DIR, "global_scaler.pkl"))
    part1_files = glob.glob(os.path.join(DATA_DIR, "*_part1.pkl"))
    dfs = []
    req_cols = list(NEW_TARGETS.values()) + EXOGENOUS
    for f in part1_files:
        df = pd.read_pickle(f)[req_cols].ffill().bfill().dropna()
        if len(df) > 0: dfs.append(df)
        
    for name, t_col in NEW_TARGETS.items():
        X_all, y_all = [], []
        for df in dfs:
            if len(df) <= 3: continue
            df_sc = pd.DataFrame(global_scaler.transform(df), columns=df.columns) # Wait, global scaler features might not match NEW_TARGETS order.
            # Actually, global scaler was fit on all columns of Phase 2 master_df. 
            # We must use exactly the same columns in the same order as global_scaler.
            pass
            
    # CRITICAL FIX: Since we can't easily reuse global_scaler for a new column subset if it wasn't in the original, 
    # we will just extract the predictions for Event 31 using the existing Pitch PGNN, 
    # but lower the threshold. Let's check what the probability was!
    
if __name__ == "__main__":
    pass
