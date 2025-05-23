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
BASE_DATA_DIR = "generated_time_series_data"
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
TRANSFORMER_DIM = 64
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
            # A more robust approach might be rolling window normalization or global stats from training
            scalers = [StandardScaler() for _ in range(num_actual_sensors)]
            for i in range(num_actual_sensors):
                features[:, i] = scalers[i].fit_transform(features[:, i].reshape(-1, 1)).flatten()

            self.data_cache.append({
                "features": features, # [T, NumActualSensors]
                "failure_flags": failure_flags, # [T]
                "num_actual_sensors": num_actual_sensors
            })

            # Create window indices
            # Ensure enough data for sequence + max prediction/failure horizon
            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons))
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

        # Input sequence
        input_seq_features = features_full[window_start_idx : window_start_idx + self.seq_len]
        # input_seq_failures = failure_flags_full[window_start_idx : window_start_idx + self.seq_len] # Not directly used as model input but could be

        # Padding for features if num_actual_sensors < self.max_sensors_global
        padded_input_seq_features = np.zeros((self.seq_len, self.max_sensors_global), dtype=np.float32)
        padded_input_seq_features[:, :num_actual_sensors] = input_seq_features
        
        sensor_mask = np.zeros(self.max_sensors_global, dtype=np.float32)
        sensor_mask[:num_actual_sensors] = 1.0

        # --- Prediction Targets ---
        # Shape: [MaxSensors, NumPredHorizons]
        pred_targets = np.zeros((self.max_sensors_global, len(self.pred_horizons)), dtype=np.float32)
        for i, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < len(features_full):
                pred_targets[:num_actual_sensors, i] = features_full[target_idx, :num_actual_sensors]
            # else: values remain 0 (or use a special padding value)

        # --- Failure Detection Targets ---
        # Shape: [NumFailHorizons]
        fail_targets = np.zeros(len(self.fail_horizons), dtype=np.float32)
        for i, n in enumerate(self.fail_horizons):
            fail_window_start = window_start_idx + self.seq_len
            fail_window_end = fail_window_start + n
            if fail_window_end <= len(failure_flags_full):
                if np.any(failure_flags_full[fail_window_start:fail_window_end]):
                    fail_targets[i] = 1.0
        
        # --- RCA Targets ---
        # Shape: [MaxSensors] - Binary: is sensor 'responsible' for an imminent failure?
        # Proxy: If failure in self.rca_failure_lookahead, which sensors are anomalous in that failure window?
        rca_targets = np.zeros(self.max_sensors_global, dtype=np.float32)
        rca_fail_window_start = window_start_idx + self.seq_len
        rca_fail_window_end = rca_fail_window_start + self.rca_failure_lookahead
        
        is_imminent_failure = False
        if rca_fail_window_end <= len(failure_flags_full):
            if np.any(failure_flags_full[rca_fail_window_start : rca_fail_window_end]):
                is_imminent_failure = True

        if is_imminent_failure:
            # Identify anomalous sensors in the imminent failure window
            # Anomaly: value deviates > 3 std from its mean in the input_seq_features
            failure_sub_window = features_full[rca_fail_window_start : rca_fail_window_end, :num_actual_sensors]
            if failure_sub_window.shape[0] > 0: # If failure window is not empty
                input_means = np.mean(input_seq_features[:, :num_actual_sensors], axis=0)
                input_stds = np.std(input_seq_features[:, :num_actual_sensors], axis=0)
                input_stds[input_stds == 0] = 1e-6 # Avoid division by zero

                # Check if any point in failure_sub_window for a sensor is anomalous
                for s_idx in range(num_actual_sensors):
                    sensor_fail_values = failure_sub_window[:, s_idx]
                    # Check for significant deviation from input window's stats for that sensor
                    if np.any(np.abs(sensor_fail_values - input_means[s_idx]) > 3 * input_stds[s_idx]):
                        rca_targets[s_idx] = 1.0
        
        return {
            "input_features": torch.from_numpy(padded_input_seq_features), # [SeqLen, MaxSensors]
            "sensor_mask": torch.from_numpy(sensor_mask), # [MaxSensors]
            "pred_targets": torch.from_numpy(pred_targets), # [MaxSensors, NumPredHorizons]
            "fail_targets": torch.from_numpy(fail_targets), # [NumFailHorizons]
            "rca_targets": torch.from_numpy(rca_targets) # [MaxSensors]
        }

# --- Model Architecture ---
class SensorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, x): # x: [Batch * MaxSensors, SeqLen, InputDim=1]
        _, (hn, _) = self.lstm(x)
        return hn.squeeze(0) # [Batch * MaxSensors, HiddenDim]

class InterSensorTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, max_sensors):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_sensors, embed_dim)) # Learned pos encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = embed_dim

    def forward(self, x, src_key_padding_mask): # x: [Batch, MaxSensors, EmbedDim]
                                                # src_key_padding_mask: [Batch, MaxSensors] (True for masked)
        x = x + self.pos_encoder[:, :x.size(1), :] # Add pos encoding up to current num sensors
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
        
        self.inter_sensor_transformer = InterSensorTransformer(
            sensor_embed_dim, transformer_nhead, transformer_nlayers, max_sensors
        )

        # CLS token for global features
        self.cls_token = nn.Parameter(torch.zeros(1, 1, sensor_embed_dim))
        # Adjust transformer input dim if CLS is used this way, or adjust how CLS token interacts
        # For simplicity, we'll pool transformer output instead of CLS token for now.

        # MoE
        self.gating_network = GatingNetwork(transformer_dim, num_experts) # Input is pooled transformer output
        self.experts = nn.ModuleList([
            Expert(transformer_dim, moe_hidden_dim, moe_output_dim) for _ in range(num_experts)
        ])
        
        # Task Heads
        # Prediction: operates on per-sensor transformer outputs
        self.pred_head = nn.Linear(transformer_dim + moe_output_dim, pred_horizons_len) # Predict all horizons at once per sensor
        
        # Failure Detection: operates on MoE global output
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        
        # RCA: operates on per-sensor transformer outputs
        self.rca_head = nn.Linear(transformer_dim + moe_output_dim, 1) # Score per sensor


    def forward(self, x_features, sensor_mask):
        # x_features: [Batch, SeqLen, MaxSensors]
        # sensor_mask: [Batch, MaxSensors] (1 for active, 0 for padding)
        
        batch_size, seq_len, _ = x_features.shape
        
        # Sensor Encoding
        # Reshape for sensor_encoder: [B, S, MS] -> [B, MS, S, 1] -> [B*MS, S, 1]
        x_reshaped = x_features.permute(0, 2, 1).unsqueeze(-1) # [B, MaxSensors, SeqLen, 1]
        x_reshaped = x_reshaped.reshape(batch_size * self.max_sensors, seq_len, 1)
        
        sensor_embeddings = self.sensor_encoder(x_reshaped) # [B*MaxSensors, SensorEmbedDim]
        sensor_embeddings = sensor_embeddings.reshape(batch_size, self.max_sensors, -1) # [B, MaxSensors, SensorEmbedDim]
        
        # Apply mask after sensor encoding (zero out embeddings of padded sensors)
        sensor_embeddings = sensor_embeddings * sensor_mask.unsqueeze(-1)

        # Inter-Sensor Transformer
        # Transformer expects padding mask where True means masked
        transformer_padding_mask = (sensor_mask == 0)
        transformer_out = self.inter_sensor_transformer(sensor_embeddings, transformer_padding_mask)
        # transformer_out: [B, MaxSensors, TransformerDim]

        # Global features by masked average pooling
        # Expand mask for broadcasting: [B, MaxSensors, 1]
        expanded_sensor_mask = sensor_mask.unsqueeze(-1)
        # Sum active sensor features: [B, TransformerDim]
        summed_features = (transformer_out * expanded_sensor_mask).sum(dim=1)
        # Count active sensors per batch item, avoid div by zero: [B, 1]
        active_sensors_count = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_features = summed_features / active_sensors_count # [B, TransformerDim]

        # MoE Layer
        gating_scores = self.gating_network(global_features) # [B, NumExperts]
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(global_features)) # Each is [B, MoeOutputDim]
        
        # Weighted sum of expert outputs
        # gating_scores: [B, NumExperts], stack(expert_outputs): [NumExperts, B, MoeOutputDim] -> permute
        stacked_expert_outputs = torch.stack(expert_outputs, dim=1) # [B, NumExperts, MoeOutputDim]
        moe_global_output = (gating_scores.unsqueeze(-1) * stacked_expert_outputs).sum(dim=1) # [B, MoeOutputDim]

        # Prepare features for per-sensor heads by concatenating MoE output
        # moe_global_output_expanded: [B, 1, MoeOutputDim] -> broadcast to [B, MaxSensors, MoeOutputDim]
        moe_global_output_expanded = moe_global_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        per_sensor_combined_features = torch.cat([transformer_out, moe_global_output_expanded], dim=-1)
        # per_sensor_combined_features: [B, MaxSensors, TransformerDim + MoeOutputDim]

        # Task-specific predictions
        pred_output = self.pred_head(per_sensor_combined_features) # [B, MaxSensors, NumPredHorizons]
        fail_output = self.fail_head(moe_global_output)            # [B, NumFailHorizons]
        rca_output = self.rca_head(per_sensor_combined_features).squeeze(-1) # [B, MaxSensors]
        
        return pred_output, fail_output, rca_output

# --- Training Loop ---
def train_model():
    print(f"Using device: {DEVICE}")

    # Scan all data to determine global max sensors for consistent model architecture
    all_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv")) + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_files:
        print("No CSV files found in TRAIN_DIR or VALID_DIR. Exiting.")
        return
        
    max_sensors_overall = get_max_sensors_from_files(all_files)
    if max_sensors_overall == 0:
        print("Could not determine max_sensors_overall or no sensors found. Exiting.")
        return
    print(f"Determined max_sensors_overall (capped at {MAX_SENSORS_CAP}): {max_sensors_overall}")
    
    # Save this for inference later
    np.savez(PREPROCESSOR_SAVE_PATH, max_sensors_overall=max_sensors_overall)


    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS, RCA_FAILURE_LOOKAHEAD, max_sensors_overall)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS, RCA_FAILURE_LOOKAHEAD, max_sensors_overall)

    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        print("Not enough data to create datasets. Exiting.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # Set num_workers > 0 for parallel loading if issues arise
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FoundationalTimeSeriesModel(
        max_sensors=max_sensors_overall,
        sensor_input_dim=1, # Raw value per sensor
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
    
    # Loss functions
    mse_loss = nn.MSELoss(reduction='none') # For masked loss
    bce_loss = nn.BCEWithLogitsLoss()

    # Loss weights (tune these)
    w_pred, w_fail, w_rca = 1.0, 1.0, 0.5 

    for epoch in range(EPOCHS):
        model.train()
        total_loss_train, total_pred_loss_train, total_fail_loss_train, total_rca_loss_train = 0,0,0,0
        
        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(DEVICE)
            sensor_mask = batch["sensor_mask"].to(DEVICE) # [B, MaxSensors]
            pred_targets = batch["pred_targets"].to(DEVICE)
            fail_targets = batch["fail_targets"].to(DEVICE)
            rca_targets = batch["rca_targets"].to(DEVICE)

            optimizer.zero_grad()
            
            pred_output, fail_output, rca_output = model(input_features, sensor_mask)
            
            # Prediction Loss (Masked)
            # pred_output: [B, MaxSensors, NumPredHorizons], pred_targets: [B, MaxSensors, NumPredHorizons]
            loss_pred_unmasked = mse_loss(pred_output, pred_targets)
            # sensor_mask for pred: [B, MaxSensors, 1] to broadcast with NumPredHorizons
            loss_pred = (loss_pred_unmasked * sensor_mask.unsqueeze(-1)).sum() / sensor_mask.sum().clamp(min=1)

            # Failure Detection Loss
            loss_fail = bce_loss(fail_output, fail_targets)
            
            # RCA Loss (Masked) - rca_output/targets are [B, MaxSensors]
            loss_rca_unmasked = bce_loss(rca_output, rca_targets) # bce_loss handles reduction internally, need elementwise for masking
            # For element-wise BCE for masking:
            loss_rca_elementwise = nn.BCEWithLogitsLoss(reduction='none')(rca_output, rca_targets)
            loss_rca = (loss_rca_elementwise * sensor_mask).sum() / sensor_mask.sum().clamp(min=1)
            
            combined_loss = w_pred * loss_pred + w_fail * loss_fail + w_rca * loss_rca
            combined_loss.backward()
            optimizer.step()

            total_loss_train += combined_loss.item()
            total_pred_loss_train += loss_pred.item()
            total_fail_loss_train += loss_fail.item()
            total_rca_loss_train += loss_rca.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Train Loss: {combined_loss.item():.4f} "
                      f"(P:{loss_pred.item():.4f} F:{loss_fail.item():.4f} R:{loss_rca.item():.4f})")

        avg_train_loss = total_loss_train / len(train_loader)
        avg_pred_loss_train = total_pred_loss_train / len(train_loader)
        avg_fail_loss_train = total_fail_loss_train / len(train_loader)
        avg_rca_loss_train = total_rca_loss_train / len(train_loader)
        print(f"Epoch {epoch+1} Avg Train Loss: {avg_train_loss:.4f} "
              f"(P:{avg_pred_loss_train:.4f} F:{avg_fail_loss_train:.4f} R:{avg_rca_loss_train:.4f})")

        # Validation
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
                
                loss_pred_unmasked = mse_loss(pred_output, pred_targets)
                loss_pred = (loss_pred_unmasked * sensor_mask.unsqueeze(-1)).sum() / sensor_mask.sum().clamp(min=1)
                loss_fail = bce_loss(fail_output, fail_targets)
                loss_rca_elementwise = nn.BCEWithLogitsLoss(reduction='none')(rca_output, rca_targets)
                loss_rca = (loss_rca_elementwise * sensor_mask).sum() / sensor_mask.sum().clamp(min=1)
                
                combined_loss = w_pred * loss_pred + w_fail * loss_fail + w_rca * loss_rca
                total_loss_val += combined_loss.item()
                total_pred_loss_val += loss_pred.item()
                total_fail_loss_val += loss_fail.item()
                total_rca_loss_val += loss_rca.item()

        avg_val_loss = total_loss_val / len(valid_loader)
        avg_pred_loss_val = total_pred_loss_val / len(valid_loader)
        avg_fail_loss_val = total_fail_loss_val / len(valid_loader)
        avg_rca_loss_val = total_rca_loss_val / len(valid_loader)
        print(f"Epoch {epoch+1} Avg Validation Loss: {avg_val_loss:.4f} "
              f"(P:{avg_pred_loss_val:.4f} F:{avg_fail_loss_val:.4f} R:{avg_rca_loss_val:.4f})")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Preprocessor config (max_sensors_overall) saved to {PREPROCESSOR_SAVE_PATH}")

if __name__ == '__main__':
    # Set seeds for reproducibility (optional)
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)
        
    train_model()