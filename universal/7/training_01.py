import os
import glob
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# IMPORTANT: USER MUST SET THESE PATHS
BASE_DATA_DIR = "../../data/time_series/1"  # Example base path
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAINING")  # <<< SET THIS
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # <<< SET THIS

# Model & Data Parameters
SEQ_LEN = 64
PRED_HORIZON_TARGET = 5
MAX_SENSORS_CAP = 20

# Architectural Params for Per-Sensor Encoder
SENSOR_INPUT_DIM = 1
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1  # Dropout for TCN; active during model.train()

# Training Parameters
EPOCHS = 10  # Adjust as needed
BATCH_SIZE = 32  # Adjust based on your GPU memory
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2
HUBER_DELTA = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "simplified_timeseries_model_h5.pth"
PREPROCESSOR_SAVE_PATH = "simplified_preprocessor_config_h5.npz"


# --- RevIN Layer ---
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x_norm = (x - self.mean) / self.stdev
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm
        elif mode == 'denorm':
            if not hasattr(self, 'mean') or not hasattr(self, 'stdev'): return x
            x_denorm = x
            if self.affine:
                safe_affine_weight = self.affine_weight + self.eps if torch.abs(
                    self.affine_weight).min() < self.eps else self.affine_weight
                x_denorm = (x_denorm - self.affine_bias) / safe_affine_weight
            x_denorm = x_denorm * self.stdev + self.mean
            return x_denorm
        else:
            raise NotImplementedError(f"RevIN mode '{mode}' not implemented.")

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=-1, keepdim=True)
        self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps)


# --- Helper: Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# --- TCN Residual Block ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        try:
            self.conv1 = nn.utils.parametrizations.weight_norm(self.conv1)
        except AttributeError:
            self.conv1 = nn.utils.weight_norm(self.conv1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.padding,
                               dilation=dilation)
        try:
            self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)
        except AttributeError:
            self.conv2 = nn.utils.weight_norm(self.conv2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


# --- Per-Sensor TCN Encoder ---
class PerSensorEncoderTCN(nn.Module):
    def __init__(self, input_dim, proj_dim, tcn_out_dim, seq_len, num_levels, kernel_size, dropout):
        super(PerSensorEncoderTCN, self).__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=max(seq_len, 5000))  # Ensure max_len is sufficient
        tcn_blocks = []
        current_channels = proj_dim
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels_block = tcn_out_dim
            tcn_blocks.append(
                TemporalBlock(current_channels, out_channels_block, kernel_size, stride=1, dilation=dilation_size,
                              dropout=dropout))
            current_channels = out_channels_block
        self.tcn_network = nn.Sequential(*tcn_blocks)
        self.final_norm = nn.LayerNorm(tcn_out_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)
        x_tcn_out = self.tcn_network(x)
        x_permuted_back = x_tcn_out.permute(0, 2, 1)
        x_normed = self.final_norm(x_permuted_back)
        return x_normed


# --- Data Handling ---
def get_max_sensors_from_files(file_paths, cap=MAX_SENSORS_CAP):
    max_s = 0
    if not file_paths:
        print("Warning: No data files found for determining max sensors.")
        return 0
    for fp in file_paths:
        try:
            df_peek = pd.read_csv(fp, nrows=1)
            sensor_cols = [c for c in df_peek.columns if c.startswith("Sensor")]
            if not sensor_cols:  # Attempt to infer if "Sensor" prefix not used
                potential_cols = [c for c in df_peek.columns if df_peek[c].dtype in [np.float64, np.int64]]
                # Basic exclusion, might need refinement
                potential_cols = [c for c in potential_cols if
                                  not any(kw in c.lower() for kw in ['time', 'date', 'label', 'failure', 'id'])]
                sensor_cols = potential_cols
            max_s = max(max_s, len(sensor_cols))
        except Exception as e:
            print(f"Warning: Could not read {fp} to determine max sensors: {e}")
            continue
    return min(max_s, cap) if cap > 0 and max_s > cap else max_s


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizon, max_sensors_global):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.max_sensors_global = max_sensors_global
        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
        self.data_cache = []
        self.window_indices = []
        if not self.file_paths:
            print(f"ERROR: No CSV files found in directory: {data_dir}. Please check the path.")
            # Consider raising an error or exiting if no data is critical
            # raise FileNotFoundError(f"No CSV files found in directory: {data_dir}")
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(f"Loading data from {self.data_dir}...")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping file {fp} due to loading error: {e}"); continue
            sensor_cols = [c for c in df.columns if c.startswith("Sensor")]
            if not sensor_cols:
                # Try to infer sensor columns if "Sensor" prefix is not strictly used
                potential_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64, np.float32]]
                sensor_cols = [c for c in potential_cols if not any(kw in c.lower() for kw in
                                                                    ['time', 'date', 'label', 'failure', 'id',
                                                                     'current_failure'])]  # common non-sensor keywords
                if not sensor_cols:
                    print(f"Warning: No 'Sensor*' or inferable sensor columns found in {fp}. Skipping this file.")
                    continue
            if len(sensor_cols) > self.max_sensors_global: sensor_cols = sensor_cols[:self.max_sensors_global]
            features = df[sensor_cols].values.astype(np.float32)
            num_actual_sensors = features.shape[1]
            if num_actual_sensors == 0: print(
                f"Warning: No sensor data after column selection in {fp}. Skipping."); continue
            scalers = [StandardScaler() for _ in range(num_actual_sensors)]
            for i in range(num_actual_sensors):
                if features.shape[0] > 1:
                    features[:, i] = scalers[i].fit_transform(features[:, i].reshape(-1, 1)).flatten()
                elif features.shape[0] == 1:
                    features[:, i] = 0  # Or some other placeholder for single point series
            self.data_cache.append({"features": features, "num_actual_sensors": num_actual_sensors})
            max_lookahead = self.pred_horizon
            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        if not self.data_cache:
            print(f"CRITICAL WARNING: No data successfully loaded from {self.data_dir}. Dataset is empty.")
        else:
            print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item_data = self.data_cache[file_idx];
        features_full, n_actual = item_data["features"], item_data["num_actual_sensors"]
        input_orig = features_full[window_start_idx: window_start_idx + self.seq_len]
        last_known_val_at_input_end = np.zeros(self.max_sensors_global, dtype=np.float32)
        if input_orig.shape[0] > 0: last_known_val_at_input_end[:n_actual] = input_orig[-1, :n_actual]
        padded_input = np.zeros((self.seq_len, self.max_sensors_global), dtype=np.float32);
        padded_input[:, :n_actual] = input_orig
        sensor_mask = np.zeros(self.max_sensors_global, dtype=np.float32);
        sensor_mask[:n_actual] = 1.0
        delta_target_h5 = np.zeros(self.max_sensors_global, dtype=np.float32)
        target_idx_h5 = window_start_idx + self.seq_len + self.pred_horizon - 1
        if target_idx_h5 < len(features_full):
            target_value_h5 = features_full[target_idx_h5, :n_actual]
            delta_target_h5[:n_actual] = target_value_h5 - last_known_val_at_input_end[:n_actual]
        return {"input_features": torch.from_numpy(padded_input), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_input": torch.from_numpy(last_known_val_at_input_end),
                "pred_delta_target_h5": torch.from_numpy(delta_target_h5)}


# --- Simplified Prediction Model ---
class SimplifiedPredictionModel(nn.Module):
    def __init__(self, max_sensors, seq_len, sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout, revin_affine=True):
        super().__init__()
        self.max_sensors = max_sensors;
        self.seq_len = seq_len;
        self.sensor_tcn_out_dim = sensor_tcn_out_dim
        self.revin_layer = RevIN(num_features=max_sensors, affine=revin_affine)
        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)
        self.pred_head_h5 = nn.Linear(sensor_tcn_out_dim, 1)

    def forward(self, x_features_orig_scale, sensor_mask):
        batch_size, seq_len, _ = x_features_orig_scale.shape
        x_revin_input = x_features_orig_scale.permute(0, 2, 1) * sensor_mask.unsqueeze(-1)
        x_norm_revin = self.revin_layer(x_revin_input, mode='norm')
        last_val_norm_for_delta_recon = x_norm_revin[:, :, -1]
        x_for_encoder = x_norm_revin.permute(0, 2, 1)
        x_reshaped_for_encoder = x_for_encoder.reshape(batch_size * self.max_sensors, seq_len, SENSOR_INPUT_DIM)
        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, self.max_sensors, seq_len,
                                                                         self.sensor_tcn_out_dim)
        sensor_temporal_features = sensor_temporal_features * sensor_mask.view(batch_size, self.max_sensors, 1, 1)
        features_for_pred_head = sensor_temporal_features[:, :, -1, :]
        pred_delta_norm_h5 = self.pred_head_h5(features_for_pred_head)
        pred_abs_norm_h5 = last_val_norm_for_delta_recon.unsqueeze(-1) + pred_delta_norm_h5
        pred_abs_denorm_h5 = self.revin_layer(pred_abs_norm_h5, mode='denorm') * sensor_mask.unsqueeze(-1)
        return pred_abs_denorm_h5


# --- Training Function ---
def train_and_save_model():
    print(f"Using device: {DEVICE}")
    print(f"TRAIN_DIR: {TRAIN_DIR}")
    print(f"VALID_DIR: {VALID_DIR}")

    if not (os.path.exists(TRAIN_DIR) and os.path.isdir(TRAIN_DIR)):
        print(f"ERROR: TRAIN_DIR '{TRAIN_DIR}' does not exist or is not a directory. Please create it and add data.")
        return
    if not (os.path.exists(VALID_DIR) and os.path.isdir(VALID_DIR)):
        print(f"ERROR: VALID_DIR '{VALID_DIR}' does not exist or is not a directory. Please create it and add data.")
        return

    all_file_paths = glob.glob(os.path.join(TRAIN_DIR, "*.csv")) + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_file_paths:
        print("ERROR: No CSV files found in TRAIN_DIR or VALID_DIR. Cannot determine max_sensors_overall.")
        return
    max_s_overall = get_max_sensors_from_files(all_file_paths, MAX_SENSORS_CAP)
    if max_s_overall == 0:
        print("ERROR: Determined max_sensors_overall to be 0. Check data files or 'get_max_sensors_from_files' logic.")
        return
    print(f"Determined max_sensors_overall: {max_s_overall}")
    np.savez(PREPROCESSOR_SAVE_PATH, max_s_overall=max_s_overall, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON_TARGET)
    print(f"Preprocessor config saved to {PREPROCESSOR_SAVE_PATH}")

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZON_TARGET, max_s_overall)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZON_TARGET, max_s_overall)

    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty. Please check data and logs.")
        return
    if len(valid_dataset) == 0:
        print("Warning: Validation dataset is empty. Proceeding without validation.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=True if DEVICE.type == 'cuda' else False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=True if DEVICE.type == 'cuda' else False) if len(valid_dataset) > 0 else None

    model = SimplifiedPredictionModel(
        max_sensors=max_s_overall, seq_len=SEQ_LEN, sensor_input_dim=SENSOR_INPUT_DIM,
        sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM, sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')  # Use 'none' for manual masking

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(DEVICE)
            sensor_mask = batch["sensor_mask"].to(DEVICE)
            last_known = batch["last_known_values_input"].to(DEVICE)
            delta_target = batch["pred_delta_target_h5"].to(DEVICE)

            optimizer.zero_grad()
            pred_abs_denorm_h5 = model(input_features, sensor_mask)  # [B, MaxSensors, 1]

            # Construct absolute target for loss calculation
            # delta_target is [B, MaxSensors], last_known is [B, MaxSensors]
            abs_target_h5 = last_known.unsqueeze(-1) + delta_target.unsqueeze(-1)  # [B, MaxSensors, 1]

            loss_elements = loss_fn(pred_abs_denorm_h5, abs_target_h5)  # [B, MaxSensors, 1]

            # Apply mask and calculate mean loss for active sensor-timesteps
            masked_loss = loss_elements * sensor_mask.unsqueeze(-1)
            num_active_elements = sensor_mask.sum().clamp(min=1e-9)  # Total active sensors in batch
            batch_loss = masked_loss.sum() / num_active_elements

            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                print(
                    f"Warning: NaN or Inf loss detected at Epoch {epoch + 1}, Batch {batch_idx + 1}. Skipping update.")
                continue

            batch_loss.backward()
            optimizer.step()
            total_train_loss += batch_loss.item()

            if batch_idx % (len(train_loader) // 5 if len(train_loader) >= 5 else 1) == 0:  # Print ~5 times per epoch
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_idx + 1}/{len(train_loader)} | Train Loss: {batch_loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Average Training Loss: {avg_train_loss:.4f}")

        if valid_loader:
            model.eval()
            total_valid_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    input_features = batch["input_features"].to(DEVICE)
                    sensor_mask = batch["sensor_mask"].to(DEVICE)
                    last_known = batch["last_known_values_input"].to(DEVICE)
                    delta_target = batch["pred_delta_target_h5"].to(DEVICE)
                    pred_abs_denorm_h5 = model(input_features, sensor_mask)
                    abs_target_h5 = last_known.unsqueeze(-1) + delta_target.unsqueeze(-1)
                    loss_elements = loss_fn(pred_abs_denorm_h5, abs_target_h5)
                    masked_loss = loss_elements * sensor_mask.unsqueeze(-1)
                    num_active_elements = sensor_mask.sum().clamp(min=1e-9)
                    batch_loss = masked_loss.sum() / num_active_elements
                    if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                        total_valid_loss += batch_loss.item()
            avg_valid_loss = total_valid_loss / len(valid_loader) if len(valid_loader) > 0 else float('inf')
            print(f"Epoch {epoch + 1}/{EPOCHS} | Average Validation Loss: {avg_valid_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    print("---------------------------------------------------------------------")
    print("IMPORTANT: Ensure TRAIN_DIR and VALID_DIR are correctly set in the script.")
    print(f"Current TRAIN_DIR: {TRAIN_DIR}")
    print(f"Current VALID_DIR: {VALID_DIR}")
    print("If these are not your data directories, please edit the script before running.")
    print("---------------------------------------------------------------------")

    # Check if default paths exist, if not, prompt user strongly.
    if BASE_DATA_DIR == "../../data/time_series/1":  # Default example path
        print("\nWARNING: Using default example data paths. These might not exist on your system.")
        print("Please verify and update TRAIN_DIR and VALID_DIR if necessary.\n")

    train_and_save_model()