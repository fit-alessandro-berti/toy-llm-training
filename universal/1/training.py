import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# Data paths (assuming generated data from previous script)
BASE_DATA_DIR = "../../data/time_series/1"
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAINING")
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")

# Model & Training Parameters
SEQ_LEN = 64  # Input sequence length
PRED_HORIZONS = [1, 3, 5]  # Predict 1, 3, 5 steps ahead
FAIL_HORIZONS = [3, 5, 10] # Detect failure within next 3, 5, 10 steps
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0] # For RCA target, consider failure in shortest horizon

MAX_SENSORS_CAP = 20 # Cap maximum number of sensors the model's fixed layers will be built for.
                     # Files with more sensors will be truncated, fewer will be padded.

# Architectural Params
SENSOR_EMBED_DIM = 32
TRANSFORMER_DIM = 64 # This is the d_model for the InterSensorTransformer
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2
MOE_HIDDEN_DIM = 128
NUM_EXPERTS = 4
MOE_OUTPUT_DIM = 64 # Dimension of the MoE's output feature vector

# Training Params
BATCH_SIZE = 32
EPOCHS = 10 # Adjust as needed
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "foundation_timeseries_model.pth"
PREPROCESSOR_SAVE_PATH = "preprocessor_config.npz"


# --- Data Handling ---
def get_max_sensors_from_files(file_paths):
    max_s = 0
    for fp in file_paths:
        try:
            # Quickly peek at header to get sensor count
            df_peek = pd.read_csv(fp, nrows=1)
            num_cols = df_peek.shape[1]
            # Assuming last column is CURRENT_FAILURE
            max_s = max(max_s, num_cols - 1)
        except Exception as e:
            print(f"Warning: Could not read {fp} for sensor count: {e}")
            continue
    return min(max_s, MAX_SENSORS_CAP) # Cap it


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizons, fail_horizons, rca_failure_lookahead, max_sensors_global):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons
        self.fail_horizons = fail_horizons
        self.rca_failure_lookahead = rca_failure_lookahead
        self.max_sensors_global = max_sensors_global # Max sensors model is built for

        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
        self.data_cache = [] # Cache loaded and preprocessed series
        self.window_indices = [] # Stores (file_idx, window_start_idx)

        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(f"Loading data from {self.data_dir}...")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping file {fp} due to loading error: {e}")
                continue

            sensor_cols = [col for col in df.columns if col.startswith("Sensor")]
            if not sensor_cols:
                print(f"Warning: No sensor columns found in {fp}. Skipping.")
                continue

            # Truncate if more sensors than cap, select first MAX_SENSORS_CAP
            if len(sensor_cols) > self.max_sensors_global:
                sensor_cols = sensor_cols[:self.max_sensors_global]
            
            # Features (sensor data) and failure flags
            features = df[sensor_cols].values.astype(np.float32)
            failure_flags = df["CURRENT_FAILURE"].values.astype(np.int64)
            
            num_actual_sensors = features.shape[1]

            # Normalize features (per-file, per-sensor for simplicity here)
            scalers = [StandardScaler() for _ in range(num_actual_sensors)]
            for i in range(num_actual_sensors):
                if features.shape[0] > 0: # Ensure there's data to fit
                    features[:, i] = scalers[i].fit_transform(features[:, i].reshape(-1, 1)).flatten()

            self.data_cache.append({
                "features": features, # [T, NumActualSensors]
                "failure_flags": failure_flags, # [T]
                "num_actual_sensors": num_actual_sensors
            })

            # Create window indices
            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons), self.rca_failure_lookahead)
            for i in range(len(df) - self.seq_len - max_lookahead + 1):
                self.window_indices.append((file_idx, i))
        print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")


    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        cached_item = self.data_cache[file_idx]
        
        features_full = cached_item["features"]
        failure_flags_full = cached_item["failure_flags"]
        num_actual_sensors = cached_item["num_actual_sensors"]

        input_seq_features = features_full[window_start_idx : window_start_idx + self.seq_len]
        
        padded_input_seq_features = np.zeros((self.seq_len, self.max_sensors_global), dtype=np.float32)
        padded_input_seq_features[:, :num_actual_sensors] = input_seq_features
        
        sensor_mask = np.zeros(self.max_sensors_global, dtype=np.float32)
        sensor_mask[:num_actual_sensors] = 1.0

        pred_targets = np.zeros((self.max_sensors_global, len(self.pred_horizons)), dtype=np.float32)
        for i, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < len(features_full):
                pred_targets[:num_actual_sensors, i] = features_full[target_idx, :num_actual_sensors]

        fail_targets = np.zeros(len(self.fail_horizons), dtype=np.float32)
        for i, n in enumerate(self.fail_horizons):
            fail_window_start = window_start_idx + self.seq_len
            fail_window_end = fail_window_start + n
            if fail_window_end <= len(failure_flags_full):
                if np.any(failure_flags_full[fail_window_start:fail_window_end]):
                    fail_targets[i] = 1.0
        
        rca_targets = np.zeros(self.max_sensors_global, dtype=np.float32)
        rca_fail_window_start = window_start_idx + self.seq_len
        rca_fail_window_end = rca_fail_window_start + self.rca_failure_lookahead
        
        is_imminent_failure = False
        if rca_fail_window_end <= len(failure_flags_full):
            if np.any(failure_flags_full[rca_fail_window_start : rca_fail_window_end]):
                is_imminent_failure = True

        if is_imminent_failure and num_actual_sensors > 0:
            failure_sub_window_features = features_full[rca_fail_window_start : rca_fail_window_end, :num_actual_sensors]
            if failure_sub_window_features.shape[0] > 0:
                # Use stats from the input_seq_features for defining anomaly threshold
                current_input_window_features = features_full[window_start_idx : window_start_idx + self.seq_len, :num_actual_sensors]
                if current_input_window_features.shape[0] > 0:
                    input_means = np.mean(current_input_window_features, axis=0)
                    input_stds = np.std(current_input_window_features, axis=0)
                    input_stds[input_stds < 1e-6] = 1e-6 # Avoid division by zero for constant sensors

                    for s_idx in range(num_actual_sensors):
                        sensor_fail_values = failure_sub_window_features[:, s_idx]
                        if np.any(np.abs(sensor_fail_values - input_means[s_idx]) > 3 * input_stds[s_idx]):
                            rca_targets[s_idx] = 1.0
        
        return {
            "input_features": torch.from_numpy(padded_input_seq_features),
            "sensor_mask": torch.from_numpy(sensor_mask),
            "pred_targets": torch.from_numpy(pred_targets),
            "fail_targets": torch.from_numpy(fail_targets),
            "rca_targets": torch.from_numpy(rca_targets)
        }

# --- Model Architecture ---
class SensorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return hn.squeeze(0)

class InterSensorTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, max_sensors): # embed_dim here is d_model
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_sensors, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dim_feedforward=embed_dim*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        # x shape: [Batch, MaxSensors, EmbedDim]
        # Ensure pos_encoder matches MaxSensors dimension of x dynamically
        x = x + self.pos_encoder[:, :x.size(1), :] 
        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, max_sensors, sensor_input_dim, sensor_embed_dim, 
                 transformer_dim, transformer_nhead, transformer_nlayers,
                 moe_hidden_dim, num_experts, moe_output_dim,
                 pred_horizons_len, fail_horizons_len):
        super().__init__()
        self.max_sensors = max_sensors
        self.sensor_encoder = SensorEncoder(sensor_input_dim, sensor_embed_dim)
        
        if sensor_embed_dim != transformer_dim:
            self.transformer_input_proj = nn.Linear(sensor_embed_dim, transformer_dim)
        else:
            self.transformer_input_proj = nn.Identity()

        self.inter_sensor_transformer = InterSensorTransformer(
            embed_dim=transformer_dim, # This is d_model for the transformer
            nhead=transformer_nhead, 
            num_layers=transformer_nlayers, 
            max_sensors=max_sensors
        )
        
        self.gating_network = GatingNetwork(transformer_dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(transformer_dim, moe_hidden_dim, moe_output_dim) for _ in range(num_experts)
        ])
        
        self.pred_head = nn.Linear(transformer_dim + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(transformer_dim + moe_output_dim, 1)


    def forward(self, x_features, sensor_mask):
        batch_size, seq_len, _ = x_features.shape
        
        x_reshaped = x_features.permute(0, 2, 1).unsqueeze(-1)
        x_reshaped = x_reshaped.reshape(batch_size * self.max_sensors, seq_len, 1)
        
        sensor_embeddings_flat = self.sensor_encoder(x_reshaped)
        sensor_embeddings = sensor_embeddings_flat.reshape(batch_size, self.max_sensors, -1)
        sensor_embeddings = sensor_embeddings * sensor_mask.unsqueeze(-1)

        projected_sensor_embeddings = self.transformer_input_proj(sensor_embeddings)
        
        transformer_padding_mask = (sensor_mask == 0)
        transformer_out = self.inter_sensor_transformer(projected_sensor_embeddings, transformer_padding_mask)
        # transformer_out is now [B, MaxSensors, TransformerDim]
        
        expanded_sensor_mask = sensor_mask.unsqueeze(-1)
        summed_features = (transformer_out * expanded_sensor_mask).sum(dim=1)
        active_sensors_count = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_features = summed_features / active_sensors_count # [B, TransformerDim]

        gating_scores = self.gating_network(global_features)
        expert_outputs_list = [expert(global_features) for expert in self.experts]
        
        stacked_expert_outputs = torch.stack(expert_outputs_list, dim=1)
        moe_global_output = (gating_scores.unsqueeze(-1) * stacked_expert_outputs).sum(dim=1)

        moe_global_output_expanded = moe_global_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        per_sensor_combined_features = torch.cat([transformer_out, moe_global_output_expanded], dim=-1)

        pred_output = self.pred_head(per_sensor_combined_features)
        fail_output = self.fail_head(moe_global_output)
        rca_output = self.rca_head(per_sensor_combined_features).squeeze(-1)
        
        return pred_output, fail_output, rca_output

# --- Training Loop ---
def train_model():
    print(f"Using device: {DEVICE}")

    all_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv")) + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_files:
        print("No CSV files found in TRAIN_DIR or VALID_DIR. Exiting.")
        return
        
    max_sensors_overall = get_max_sensors_from_files(all_files)
    if max_sensors_overall == 0:
        print("Could not determine max_sensors_overall or no sensors found. Exiting.")
        return
    print(f"Determined max_sensors_overall (capped at {MAX_SENSORS_CAP}): {max_sensors_overall}")
    
    np.savez(PREPROCESSOR_SAVE_PATH, max_sensors_overall=max_sensors_overall)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS, RCA_FAILURE_LOOKAHEAD, max_sensors_overall)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS, RCA_FAILURE_LOOKAHEAD, max_sensors_overall)

    if len(train_dataset) == 0: # valid_dataset can be empty if not enough files for split
        print("Not enough data to create training dataset. Exiting.")
        return
    if len(valid_dataset) == 0:
        print("Warning: Validation dataset is empty. Proceeding with training only.")


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    if len(valid_dataset) > 0:
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    else:
        valid_loader = None

    model = FoundationalTimeSeriesModel(
        max_sensors=max_sensors_overall,
        sensor_input_dim=1,
        sensor_embed_dim=SENSOR_EMBED_DIM,
        transformer_dim=TRANSFORMER_DIM,
        transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        moe_hidden_dim=MOE_HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        moe_output_dim=MOE_OUTPUT_DIM,
        pred_horizons_len=len(PRED_HORIZONS),
        fail_horizons_len=len(FAIL_HORIZONS)
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    mse_loss_fn = nn.MSELoss(reduction='none')
    bce_loss_fn = nn.BCEWithLogitsLoss() # Default reduction is mean
    bce_loss_elementwise_fn = nn.BCEWithLogitsLoss(reduction='none')


    w_pred, w_fail, w_rca = 1.0, 1.0, 0.5 

    for epoch in range(EPOCHS):
        model.train()
        total_loss_train, total_pred_loss_train, total_fail_loss_train, total_rca_loss_train = 0,0,0,0
        
        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(DEVICE)
            sensor_mask = batch["sensor_mask"].to(DEVICE)
            pred_targets = batch["pred_targets"].to(DEVICE)
            fail_targets = batch["fail_targets"].to(DEVICE)
            rca_targets = batch["rca_targets"].to(DEVICE)

            optimizer.zero_grad()
            
            pred_output, fail_output, rca_output = model(input_features, sensor_mask)
            
            active_sensor_elements = sensor_mask.sum().clamp(min=1) * len(PRED_HORIZONS) # Total elements for prediction
            loss_pred_unmasked = mse_loss_fn(pred_output, pred_targets)
            loss_pred = (loss_pred_unmasked * sensor_mask.unsqueeze(-1)).sum() / active_sensor_elements

            loss_fail = bce_loss_fn(fail_output, fail_targets)
            
            loss_rca_elementwise = bce_loss_elementwise_fn(rca_output, rca_targets)
            loss_rca = (loss_rca_elementwise * sensor_mask).sum() / sensor_mask.sum().clamp(min=1)
            
            combined_loss = w_pred * loss_pred + w_fail * loss_fail + w_rca * loss_rca
            
            # Check for NaN loss
            if torch.isnan(combined_loss):
                print(f"NaN loss detected at Epoch {epoch+1}, Batch {batch_idx+1}. Skipping batch.")
                print(f"Individual losses - P:{loss_pred.item()}, F:{loss_fail.item()}, R:{loss_rca.item()}")
                # Optionally save problematic batch for inspection
                # torch.save(batch, f"nan_batch_epoch{epoch+1}_batch{batch_idx+1}.pt")
                continue # Skip optimizer step and loss accumulation for this batch

            combined_loss.backward()
            optimizer.step()

            total_loss_train += combined_loss.item()
            total_pred_loss_train += loss_pred.item()
            total_fail_loss_train += loss_fail.item()
            total_rca_loss_train += loss_rca.item()

            if batch_idx > 0 and batch_idx % 50 == 0 :
                 print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Train Loss: {combined_loss.item():.4f} "
                      f"(P:{loss_pred.item():.4f} F:{loss_fail.item():.4f} R:{loss_rca.item():.4f})")
        
        num_batches_processed_train = len(train_loader) if len(train_loader) > 0 else 1 # Avoid div by zero
        avg_train_loss = total_loss_train / num_batches_processed_train
        avg_pred_loss_train = total_pred_loss_train / num_batches_processed_train
        avg_fail_loss_train = total_fail_loss_train / num_batches_processed_train
        avg_rca_loss_train = total_rca_loss_train / num_batches_processed_train
        print(f"Epoch {epoch+1} Avg Train Loss: {avg_train_loss:.4f} "
              f"(P:{avg_pred_loss_train:.4f} F:{avg_fail_loss_train:.4f} R:{avg_rca_loss_train:.4f})")

        if valid_loader:
            model.eval()
            total_loss_val, total_pred_loss_val, total_fail_loss_val, total_rca_loss_val = 0,0,0,0
            with torch.no_grad():
                for batch in valid_loader:
                    input_features = batch["input_features"].to(DEVICE)
                    sensor_mask = batch["sensor_mask"].to(DEVICE)
                    pred_targets = batch["pred_targets"].to(DEVICE)
                    fail_targets = batch["fail_targets"].to(DEVICE)
                    rca_targets = batch["rca_targets"].to(DEVICE)

                    pred_output, fail_output, rca_output = model(input_features, sensor_mask)
                    
                    active_sensor_elements_val = sensor_mask.sum().clamp(min=1) * len(PRED_HORIZONS)
                    loss_pred_unmasked = mse_loss_fn(pred_output, pred_targets)
                    loss_pred = (loss_pred_unmasked * sensor_mask.unsqueeze(-1)).sum() / active_sensor_elements_val
                    loss_fail = bce_loss_fn(fail_output, fail_targets)
                    loss_rca_elementwise = bce_loss_elementwise_fn(rca_output, rca_targets)
                    loss_rca = (loss_rca_elementwise * sensor_mask).sum() / sensor_mask.sum().clamp(min=1)
                    
                    combined_loss = w_pred * loss_pred + w_fail * loss_fail + w_rca * loss_rca
                    total_loss_val += combined_loss.item()
                    total_pred_loss_val += loss_pred.item()
                    total_fail_loss_val += loss_fail.item()
                    total_rca_loss_val += loss_rca.item()
            
            num_batches_processed_val = len(valid_loader) if len(valid_loader) > 0 else 1
            avg_val_loss = total_loss_val / num_batches_processed_val
            avg_pred_loss_val = total_pred_loss_val / num_batches_processed_val
            avg_fail_loss_val = total_fail_loss_val / num_batches_processed_val
            avg_rca_loss_val = total_rca_loss_val / num_batches_processed_val
            print(f"Epoch {epoch+1} Avg Validation Loss: {avg_val_loss:.4f} "
                  f"(P:{avg_pred_loss_val:.4f} F:{avg_fail_loss_val:.4f} R:{avg_rca_loss_val:.4f})")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Preprocessor config (max_sensors_overall) saved to {PREPROCESSOR_SAVE_PATH}")

if __name__ == '__main__':
    train_model()