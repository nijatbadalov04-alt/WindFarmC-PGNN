import os
import glob
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from scipy.stats import skew, kurtosis
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
WINDOW_SIZE = 432  # 3 days at 10-min
STRIDE = 72        # 12 hours
DATA_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
MODEL_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\pgnn"
ARMAX_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\models\armax"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load metadata
with open(os.path.join(ARMAX_DIR, "armax_config.json"), "r") as f:
    config = json.load(f)
TARGETS = config["targets"]
EXOGENOUS = config["exogenous"]

# Load event info for labeling
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
event_info = pd.read_csv(EVENT_CSV, sep=";")
anomaly_events = set(event_info[event_info["event_label"] == "anomaly"]["event_id"].astype(str))

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
    features, labels, phys_targets = [], [], []
    
    for start in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        window_sc = df_sc.iloc[start:end].values
        window_raw = df_raw.iloc[start:end].values
        
        X, y = create_lagged_features(window_sc, target_idx, exo_idxs, 3)
        if len(X) < 10: continue
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        theta_local = model.coef_
        delta_theta = theta_local - theta_global
        
        y_pred = X @ theta_global
        res = y - y_pred
        
        r_mean = np.mean(res)
        r_std = np.std(res)
        r_max = np.max(np.abs(res))
        rmse = np.sqrt(np.mean(res**2))
        sk = skew(res)
        ku = kurtosis(res)
        
        ss_res = np.sum(res**2)
        ss_tot = np.sum((y - np.mean(y))**2) + 1e-6
        r2 = max(0, 1 - ss_res / ss_tot)
        
        amb_temp = np.mean(window_raw[:, exo_idxs[0]])
        rotor_spd_mean = np.mean(window_raw[:, exo_idxs[1]])
        rotor_spd_max = np.max(window_raw[:, exo_idxs[1]])
        power_mean = np.mean(window_raw[:, exo_idxs[2]])
        u_virt = power_mean / (amb_temp + 273.15)
        
        vec = np.concatenate([
            delta_theta, 
            [r_mean, r_std, r_max, rmse, sk, ku, r2],
            [amb_temp, rotor_spd_mean, rotor_spd_max, power_mean, u_virt]
        ])
        
        samples_to_end = n_samples - end
        label = 1 if (is_anomaly and samples_to_end < 4320) else 0
        
        features.append(vec)
        labels.append(label)
        phys_targets.append(r2)
        
    return features, labels, phys_targets

class PGNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.class_head = nn.Linear(64, 1)
        self.phys_head = nn.Linear(64, 1)
        
    def forward(self, x):
        h = self.shared(x)
        c = torch.sigmoid(self.class_head(h))
        p = self.phys_head(h)
        return c, p

def main():
    scaler = joblib.load(os.path.join(ARMAX_DIR, "global_scaler.pkl"))
    part2_files = glob.glob(os.path.join(DATA_DIR, "*_part2.pkl"))
    
    np.random.seed(42)
    np.random.shuffle(part2_files)
    split_idx = int(len(part2_files) * 0.7)
    train_files = part2_files[:split_idx]
    
    with open(os.path.join(MODEL_DIR, "test_files.json"), "w") as f:
        json.dump([os.path.basename(pf) for pf in part2_files[split_idx:]], f)
        
    print(f"Extracting features from {len(train_files)} Train/Val datasets...")
    
    all_features = {k: [] for k in TARGETS}
    all_labels = {k: [] for k in TARGETS}
    all_phys = {k: [] for k in TARGETS}
    
    for f in train_files:
        eid = os.path.basename(f).split('_')[1]
        is_anom = eid in anomaly_events
        df = pd.read_pickle(f)
        req_cols = list(TARGETS.values()) + EXOGENOUS
        df = df[req_cols].ffill().bfill().dropna()
        if len(df) < WINDOW_SIZE: continue
        
        df_sc = pd.DataFrame(scaler.transform(df), columns=df.columns)
        
        for name, t_col in TARGETS.items():
            t_idx = list(df.columns).index(t_col)
            e_idxs = [list(df.columns).index(c) for c in EXOGENOUS]
            
            theta_g = np.load(os.path.join(ARMAX_DIR, f"{name}_theta.npy"))
            feats, labs, phys = process_dataset(df_sc, df, theta_g, t_idx, e_idxs, is_anom)
            
            if feats:
                all_features[name].extend(feats)
                all_labels[name].extend(labs)
                all_phys[name].extend(phys)
                
    for name in TARGETS:
        print(f"\nTraining PGNN for {name}...")
        X = np.nan_to_num(np.array(all_features[name]))
        y = np.array(all_labels[name], dtype=np.float32).reshape(-1, 1)
        p = np.array(all_phys[name], dtype=np.float32).reshape(-1, 1)
        
        if len(X) == 0: continue
        
        feat_scaler = StandardScaler().fit(X)
        joblib.dump(feat_scaler, os.path.join(MODEL_DIR, f"{name}_feat_scaler.pkl"))
        X = feat_scaler.transform(X)
        
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                torch.tensor(y), torch.tensor(p))
        
        # Calculate class weights for unbalanced BCE
        pos_weight = torch.tensor([(len(y) - y.sum()) / (y.sum() + 1)])
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Using BCEWithLogits requires removing sigmoid from model, but since we have it, we'll use regular BCE
        # Actually, simpler to just use standard BCE and rely on data quantity.
        bce = nn.BCELoss()
        
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        model = PGNN(input_dim=X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        mse = nn.MSELoss()
        
        for epoch in range(15):
            model.train()
            total_loss = 0
            for bx, by, bp in loader:
                optimizer.zero_grad()
                pred_c, pred_p = model(bx)
                loss = bce(pred_c, by) + 0.1 * mse(pred_p, bp)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch == 14:
                print(f"  Epoch 15 Loss: {total_loss/len(loader):.4f}")
                
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{name}_pgnn.pt"))
        print(f"  Saved PGNN for {name}.")

    print("\n[SUCCESS] Phase 3 PGNN & Feature Engineering Complete.")

if __name__ == "__main__":
    main()
