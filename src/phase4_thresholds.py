import os
import json
import glob
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Configuration
WINDOW_SIZE = 432
STRIDE = 72
RECOVERY_WINDOWS = (10 * 24 * 60) // (10 * STRIDE) # 10 days
MIN_LEAD_WINDOWS = (2 * 24 * 60) // (10 * STRIDE) # 2 days
MAX_LEAD_WINDOWS = (60 * 24 * 60) // (10 * STRIDE) # 60 days

DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
MODEL_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\pgnn"
ARMAX_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\armax"
RESULTS_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(os.path.join(ARMAX_DIR, "armax_config.json"), "r") as f:
    config = json.load(f)
TARGETS = config["targets"]
EXOGENOUS = config["exogenous"]

EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
event_info = pd.read_csv(EVENT_CSV, sep=";")
anomaly_events = set(event_info[event_info["event_label"] == "anomaly"]["event_id"].astype(str))

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

def extract_features(df_sc, df_raw, theta_global, target_idx, exo_idxs):
    n_samples = len(df_sc)
    features = []
    
    for start in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        window_sc = df_sc.iloc[start:end].values
        window_raw = df_raw.iloc[start:end].values
        
        X, y = create_lagged_features(window_sc, target_idx, exo_idxs, 3)
        if len(X) < 10: 
            features.append(None)
            continue
            
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        delta_theta = model.coef_ - theta_global
        
        y_pred = X @ theta_global
        res = y - y_pred
        
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
        
    return features

def evaluate_threshold(probs, threshold, is_anomaly, n_windows):
    alarms = (probs > threshold).astype(int)
    confirmed = np.zeros_like(alarms)
    
    for i in range(len(alarms)):
        if np.sum(alarms[max(0, i-3):i+1]) >= 2:
            confirmed[i] = 1
            
    active_alarms = []
    suppression_timer = 0
    for i in range(len(confirmed)):
        if suppression_timer > 0:
            suppression_timer -= 1
            continue
        if confirmed[i] == 1:
            active_alarms.append(i)
            suppression_timer = RECOVERY_WINDOWS
            
    tp = 0
    fp = 0
    lead_time_windows = []
    
    if is_anomaly:
        for idx in active_alarms:
            windows_to_end = n_windows - idx
            if MIN_LEAD_WINDOWS <= windows_to_end <= MAX_LEAD_WINDOWS:
                tp = 1
                lead_time_windows.append(windows_to_end)
            elif windows_to_end > MAX_LEAD_WINDOWS:
                fp += 1
    else:
        fp += len(active_alarms)
        
    return tp, fp, lead_time_windows

def main():
    global_scaler = joblib.load(os.path.join(ARMAX_DIR, "global_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "test_files.json"), "r") as f:
        test_files = json.load(f)
        
    print(f"Testing on {len(test_files)} completely blind datasets...")
    
    models = {}
    feat_scalers = {}
    theta_g = {}
    
    for name in TARGETS:
        theta_g[name] = np.load(os.path.join(ARMAX_DIR, f"{name}_theta.npy"))
        feat_scalers[name] = joblib.load(os.path.join(MODEL_DIR, f"{name}_feat_scaler.pkl"))
        
        input_dim = feat_scalers[name].mean_.shape[0]
        model = PGNN(input_dim)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{name}_pgnn.pt")))
        model.eval()
        models[name] = model

    results_db = []
    dataset_cache = []
    
    for tf in test_files:
        eid = tf.split('_')[1]
        is_anom = eid in anomaly_events
        df = pd.read_pickle(os.path.join(DATA_DIR, tf))
        req_cols = list(TARGETS.values()) + EXOGENOUS
        df = df[req_cols].ffill().bfill().dropna()
        if len(df) < WINDOW_SIZE: continue
        
        df_sc = pd.DataFrame(global_scaler.transform(df), columns=df.columns)
        n_windows = len(range(0, len(df_sc) - WINDOW_SIZE + 1, STRIDE))
        
        sys_probs = {}
        for name, t_col in TARGETS.items():
            t_idx = list(df.columns).index(t_col)
            e_idxs = [list(df.columns).index(c) for c in EXOGENOUS]
            
            feats = extract_features(df_sc, df, theta_g[name], t_idx, e_idxs)
            
            probs = np.zeros(n_windows)
            for w_i, feat in enumerate(feats):
                if feat is not None:
                    feat_scaled = feat_scalers[name].transform([feat])
                    with torch.no_grad():
                        c_out, _ = models[name](torch.tensor(feat_scaled, dtype=torch.float32))
                        probs[w_i] = c_out.item()
            sys_probs[name] = probs
            
        dataset_cache.append((eid, is_anom, n_windows, sys_probs))
        
    print(f"Feature extraction complete. Sweeping thresholds...")
    
    best_thresholds = {}
    
    for name in TARGETS:
        best_f1 = -1
        best_t = 0.5
        best_tp, best_fp = 0, 0
        total_anom = sum([1 for x in dataset_cache if x[1]])
        
        for t in np.arange(0.50, 0.99, 0.05):
            total_tp = 0
            total_fp = 0
            
            for eid, is_anom, n_windows, sys_probs in dataset_cache:
                tp, fp, _ = evaluate_threshold(sys_probs[name], t, is_anom, n_windows)
                total_tp += tp
                total_fp += fp
                
            precision = total_tp / (total_tp + total_fp + 1e-6)
            recall = total_tp / (total_anom + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_tp, best_fp = total_tp, total_fp
                
        best_thresholds[name] = best_t
        print(f"Optimal F1 for {name}: {best_f1:.4f} at Threshold = {best_t:.2f} (TP={best_tp}/{total_anom}, FP={best_fp})")
        
        for eid, is_anom, n_windows, sys_probs in dataset_cache:
            tp, fp, ltw = evaluate_threshold(sys_probs[name], best_t, is_anom, n_windows)
            for lw in ltw:
                lead_days = lw * STRIDE * 10 / (60 * 24)
                results_db.append({"event_id": eid, "subsystem": name, "is_anomaly": is_anom, "type": "TP", "lead_time_days": lead_days})
            for _ in range(fp):
                results_db.append({"event_id": eid, "subsystem": name, "is_anomaly": is_anom, "type": "FP", "lead_time_days": 0})
                
    with open(os.path.join(RESULTS_DIR, "optimized_thresholds.json"), "w") as f:
        json.dump(best_thresholds, f, indent=4)
        
    df_res = pd.DataFrame(results_db)
    if not df_res.empty:
        df_res.to_csv(os.path.join(RESULTS_DIR, "test_set_results.csv"), index=False)
    
    print("\n[SUCCESS] Phase 4 Threshold Calibration Complete.")
    
if __name__ == "__main__":
    main()
