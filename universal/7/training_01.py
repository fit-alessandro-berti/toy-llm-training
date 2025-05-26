import os
import glob
import random # Retained for potential dataset shuffling if DataLoader is used, though training is removed
import math  # For positional encoding
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F # May not be needed if MoE and other parts are fully removed
from torch.utils.data import Dataset, DataLoader # DataLoader might be optional for the final script
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1" # Example path, adjust as needed
DATA_DIR_EXAMPLE = os.path.join(BASE_DATA_DIR, "TRAINING") # Using TRAINING dir as an example for data loading

# Model & Data Parameters
SEQ_LEN = 64
PRED_HORIZON_TARGET = 5 # Focusing on a single prediction horizon
MAX_SENSORS_CAP = 20 # Max sensors to consider from CSV

# Architectural Params for Per-Sensor Encoder
SENSOR_INPUT_DIM = 1 # Each sensor value at a time step

# I. Per-sensor Temporal Encoder (TCN based)
SENSOR_TCN_PROJ_DIM = 32  # Dimension after initial projection, input to first TCN layer
SENSOR_TCN_OUT_DIM = 32  # Output channels of each TCN block and final TCN encoder output
TCN_LEVELS = 4  # Number of TCN residual blocks
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1 # Dropout for TCN, can be set to 0 if not in training/eval mode distinction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward(self, x, mode):  # x: [Batch, NumSensors, SeqLen]
        if mode == 'norm':
            self._get_statistics(x)
            x_norm = (x - self.mean) / self.stdev
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm
        elif mode == 'denorm':
            if not hasattr(self, 'mean') or not hasattr(self, 'stdev'): return x # Safety for inference if stats not computed
            x_denorm = x
            if self.affine:
                # Ensure affine_weight is not zero for division
                safe_affine_weight = self.affine_weight + self.eps if torch.abs(self.affine_weight).min() < self.eps else self.affine_weight
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
        self.register_buffer('pe', pe.unsqueeze(0)) # shape [1, max_len, d_model]

    def forward(self, x): # x shape: [Batch, SeqLen, d_model]
        return x + self.pe[:, :x.size(1), :]


# --- TCN Residual Block ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        # Using weight_norm, try-except for compatibility with different PyTorch versions
        try:
            self.conv1 = nn.utils.parametrizations.weight_norm(self.conv1)
        except AttributeError:
            self.conv1 = nn.utils.weight_norm(self.conv1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        try:
            self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)
        except AttributeError:
            self.conv2 = nn.utils.weight_norm(self.conv2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()

    def forward(self, x):  # x: [Batch, Channels_in, SeqLen]
        out = self.conv1(x)
        # Correct padding removal for Conv1d with causal-like effect
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
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=seq_len) # Ensure max_len >= seq_len

        tcn_blocks = []
        current_channels = proj_dim
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels_block = tcn_out_dim # Each block outputs tcn_out_dim
            tcn_blocks.append(TemporalBlock(current_channels, out_channels_block, kernel_size, stride=1,
                                            dilation=dilation_size, dropout=dropout))
            current_channels = out_channels_block # Input to next block is output of current

        self.tcn_network = nn.Sequential(*tcn_blocks)
        self.final_norm = nn.LayerNorm(tcn_out_dim) # Normalize features from TCN

    def forward(self, x):  # Expected x: [Batch*MaxSensors, SeqLen, InputDim]
        x = self.input_proj(x) # [B*MS, SL, ProjDim]
        x = self.pos_encoder(x) # [B*MS, SL, ProjDim]
        x = x.permute(0, 2, 1)  # To [B*MS, ProjDim, SL] for Conv1d
        x_tcn_out = self.tcn_network(x) # [B*MS, TCN_OUT_DIM, SL]
        x_permuted_back = x_tcn_out.permute(0, 2, 1) # To [B*MS, SL, TCN_OUT_DIM]
        x_normed = self.final_norm(x_permuted_back) # Apply LayerNorm on the feature dimension
        return x_normed


# --- Data Handling ---
def get_max_sensors_from_files(file_paths, cap=MAX_SENSORS_CAP):
    max_s = 0
    if not file_paths: # Handle case with no files
        print("Warning: No data files found for determining max sensors.")
        return 0
    for fp in file_paths:
        try:
            df_peek = pd.read_csv(fp, nrows=1)
            # Assuming sensor columns are all columns except a potential 'timestamp' or 'CURRENT_FAILURE'
            # For simplicity, let's assume columns starting with "Sensor" or count all if no failure flag
            sensor_cols = [c for c in df_peek.columns if c.startswith("Sensor")]
            if not sensor_cols and "CURRENT_FAILURE" in df_peek.columns : # if SensorN not present, assume all others are sensors except failure
                 sensor_cols = [c for c in df_peek.columns if c != "CURRENT_FAILURE" and c!= "TIMESTAMP_COL_NAME_IF_ANY"] # Add your timestamp col name
            elif not sensor_cols:
                 sensor_cols = df_peek.columns
            max_s = max(max_s, len(sensor_cols))
        except Exception as e:
            print(f"Warning: Could not read {fp} to determine max sensors: {e}")
            continue
    return min(max_s, cap) if cap > 0 else max_s


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizon, max_sensors_global):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon # Single horizon value
        self.max_sensors_global = max_sensors_global
        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
        self.data_cache = []
        self.window_indices = [] # To store (file_idx, window_start_idx)

        if not self.file_paths:
            print(f"Warning: No CSV files found in directory: {data_dir}")
            # raise ValueError(f"No CSV files found in directory: {data_dir}")


        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(f"Loading data from {self.data_dir}...")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping file {fp} due to loading error: {e}")
                continue

            sensor_cols = [c for c in df.columns if c.startswith("Sensor")]
            if not sensor_cols: # If "Sensor" convention isn't used, try to infer or skip
                print(f"Warning: No 'Sensor*' columns found in {fp}. Skipping this file.")
                continue

            if len(sensor_cols) > self.max_sensors_global:
                sensor_cols = sensor_cols[:self.max_sensors_global]

            features = df[sensor_cols].values.astype(np.float32)
            num_actual_sensors = features.shape[1]

            if num_actual_sensors == 0:
                print(f"Warning: No sensor data extracted from {fp} with selected columns. Skipping.")
                continue

            # Normalize each sensor's series
            scalers = [StandardScaler() for _ in range(num_actual_sensors)]
            for i in range(num_actual_sensors):
                if features.shape[0] > 0: # Ensure there's data to fit
                    features[:, i] = scalers[i].fit_transform(features[:, i].reshape(-1, 1)).flatten()
                else: # Should not happen if file is not empty and cols exist
                    print(f"Warning: Empty features for sensor {i} in {fp}")


            self.data_cache.append({
                "features": features,
                "num_actual_sensors": num_actual_sensors
            })

            # Create window indices
            # Max lookahead is just the prediction horizon
            max_lookahead = self.pred_horizon
            # Ensure there's enough data for one window: seq_len for input, pred_horizon for target
            for i in range(len(df) - self.seq_len - max_lookahead + 1):
                self.window_indices.append((file_idx, i))

        if not self.data_cache:
             print(f"Warning: No data successfully loaded from {self.data_dir}. Dataset will be empty.")
        else:
            print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")


    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item_data = self.data_cache[file_idx]
        features_full = item_data["features"]
        n_actual = item_data["num_actual_sensors"]

        # Input sequence
        input_orig = features_full[window_start_idx : window_start_idx + self.seq_len]

        # Last known value from the input sequence (for delta calculation if needed or for recon)
        last_known_val_at_input_end = np.zeros(self.max_sensors_global, dtype=np.float32)
        if input_orig.shape[0] > 0: # Should always be true given window creation logic
             last_known_val_at_input_end[:n_actual] = input_orig[-1, :n_actual]


        # Pad input features to max_sensors_global
        padded_input = np.zeros((self.seq_len, self.max_sensors_global), dtype=np.float32)
        padded_input[:, :n_actual] = input_orig

        # Sensor mask
        sensor_mask = np.zeros(self.max_sensors_global, dtype=np.float32)
        sensor_mask[:n_actual] = 1.0

        # Prediction target (delta from last known value)
        delta_target_h5 = np.zeros(self.max_sensors_global, dtype=np.float32) # Target for H=5
        target_idx_h5 = window_start_idx + self.seq_len + self.pred_horizon - 1

        if target_idx_h5 < len(features_full):
            target_value_h5 = features_full[target_idx_h5, :n_actual]
            delta_target_h5[:n_actual] = target_value_h5 - last_known_val_at_input_end[:n_actual]
        # else: target remains zero (e.g. for masking loss later or indicating missing future)

        return {
            "input_features": torch.from_numpy(padded_input),         # [SeqLen, MaxSensors]
            "sensor_mask": torch.from_numpy(sensor_mask),             # [MaxSensors]
            "last_known_values_input": torch.from_numpy(last_known_val_at_input_end),# [MaxSensors]
            "pred_delta_target_h5": torch.from_numpy(delta_target_h5) # [MaxSensors]
        }


# --- Simplified Prediction Model ---
class SimplifiedPredictionModel(nn.Module):
    def __init__(self, max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout, revin_affine=True):
        super().__init__()
        self.max_sensors = max_sensors
        self.seq_len = seq_len
        self.sensor_tcn_out_dim = sensor_tcn_out_dim # Store for potential use

        self.revin_layer = RevIN(num_features=max_sensors, affine=revin_affine)

        self.per_sensor_encoder = PerSensorEncoderTCN(
            input_dim=sensor_input_dim,
            proj_dim=sensor_tcn_proj_dim,
            tcn_out_dim=sensor_tcn_out_dim,
            seq_len=seq_len,
            num_levels=tcn_levels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout
        )

        # Prediction head for a single horizon (H=5)
        # Input: SENSOR_TCN_OUT_DIM features from the last relevant timestep of TCN
        # Output: 1 value (delta prediction for that horizon)
        self.pred_head_h5 = nn.Linear(sensor_tcn_out_dim, 1)

    def forward(self, x_features_orig_scale, sensor_mask):
        # x_features_orig_scale: [Batch, SeqLen, MaxSensors]
        # sensor_mask: [Batch, MaxSensors]

        batch_size, seq_len, _ = x_features_orig_scale.shape

        # Apply RevIN normalization
        # RevIN expects [Batch, NumSensors, SeqLen]
        x_revin_input = x_features_orig_scale.permute(0, 2, 1) # -> [Batch, MaxSensors, SeqLen]
        # Apply mask before stats calculation if zeros from padding should not influence mean/std
        x_revin_input = x_revin_input * sensor_mask.unsqueeze(-1) # Masking along sequence length

        x_norm_revin = self.revin_layer(x_revin_input, mode='norm') # [Batch, MaxSensors, SeqLen]

        # Store the last normalized value for reconstructing absolute prediction
        # This is the normalized value of x_t (last point in input sequence)
        last_val_norm_for_delta_recon = x_norm_revin[:, :, -1] # [Batch, MaxSensors]

        # Prepare for PerSensorEncoderTCN:
        # TCN expects [Batch*MaxSensors, SeqLen, InputDimPerSensor=1]
        # Current x_norm_revin is [Batch, MaxSensors, SeqLen]
        # Permute to [Batch, SeqLen, MaxSensors] then reshape for encoder
        x_for_encoder = x_norm_revin.permute(0, 2, 1) # -> [Batch, SeqLen, MaxSensors]

        # Reshape and add input_dim dimension
        # Each sensor's series is treated independently by the PerSensorEncoder initially
        x_reshaped_for_encoder = x_for_encoder.reshape(batch_size * self.max_sensors, seq_len, SENSOR_INPUT_DIM)
        # The sensor_mask needs to be used to gate/select features after encoding if TCN processes all
        # However, PerSensorEncoderTCN itself doesn't use a sensor_mask internally for its layers.
        # We will use sensor_mask after reshaping back.

        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        # Output: [Batch*MaxSensors, SeqLen, SENSOR_TCN_OUT_DIM]

        # Reshape back to [Batch, MaxSensors, SeqLen, SENSOR_TCN_OUT_DIM]
        sensor_temporal_features = sensor_temporal_features_flat.reshape(
            batch_size, self.max_sensors, seq_len, self.sensor_tcn_out_dim
        )

        # Apply sensor mask to zero out features for non-active/padded sensors
        # Mask shape: [Batch, MaxSensors] -> [Batch, MaxSensors, 1, 1] for broadcasting
        sensor_temporal_features = sensor_temporal_features * sensor_mask.view(batch_size, self.max_sensors, 1, 1)

        # Select features from the last time step for prediction
        # These are the encoded representations of the sequence for each sensor
        features_for_pred_head = sensor_temporal_features[:, :, -1, :] # [Batch, MaxSensors, SENSOR_TCN_OUT_DIM]

        # Predict delta for H=5
        # pred_head_h5 takes [..., SENSOR_TCN_OUT_DIM] and outputs [..., 1]
        pred_delta_norm_h5 = self.pred_head_h5(features_for_pred_head) # [Batch, MaxSensors, 1]

        # Add the delta to the last known normalized value to get absolute normalized prediction
        # last_val_norm_for_delta_recon: [Batch, MaxSensors] -> unsqueeze for broadcasting
        pred_abs_norm_h5 = last_val_norm_for_delta_recon.unsqueeze(-1) + pred_delta_norm_h5
        # pred_abs_norm_h5 shape: [Batch, MaxSensors, 1]

        # Denormalize using RevIN
        # RevIN denorm expects [Batch, NumSensors, NumFeaturesToDenorm=1 in this case]
        pred_abs_denorm_h5 = self.revin_layer(pred_abs_norm_h5, mode='denorm')

        # Apply sensor mask again to ensure predictions for padded sensors are zero
        pred_abs_denorm_h5 = pred_abs_denorm_h5 * sensor_mask.unsqueeze(-1)

        # Final output is prediction for horizon 5
        return pred_abs_denorm_h5 # Shape: [Batch, MaxSensors, 1]


# --- Main execution example (optional, for demonstration) ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # 1. Determine max_sensors from data (or set manually if known)
    # For demonstration, assuming DATA_DIR_EXAMPLE is set and has CSVs
    # This step requires actual data files.
    example_file_paths = glob.glob(os.path.join(DATA_DIR_EXAMPLE, "*.csv"))
    if not example_file_paths:
        print(f"No CSV files found in {DATA_DIR_EXAMPLE}. Using MAX_SENSORS_CAP: {MAX_SENSORS_CAP} as fallback for demo.")
        max_s_overall = MAX_SENSORS_CAP
        if max_s_overall == 0:
            raise ValueError("MAX_SENSORS_CAP is 0 and no files to infer it. Please set a valid MAX_SENSORS_CAP.")
    else:
        max_s_overall = get_max_sensors_from_files(example_file_paths, MAX_SENSORS_CAP)
        if max_s_overall == 0:
             raise ValueError(f"Could not determine a valid number of sensors from files in {DATA_DIR_EXAMPLE}. Check data or get_max_sensors_from_files logic.")
    print(f"Effective max sensors being used: {max_s_overall}")


    # 2. Instantiate Dataset and DataLoader
    # This requires data to be present. If not, this part will fail or yield an empty loader.
    try:
        dataset = MultivariateTimeSeriesDataset(
            data_dir=DATA_DIR_EXAMPLE,
            seq_len=SEQ_LEN,
            pred_horizon=PRED_HORIZON_TARGET,
            max_sensors_global=max_s_overall
        )
        if len(dataset) == 0:
            print("Dataset is empty. Cannot proceed with DataLoader and model instantiation with real data.")
            print("Using dummy data for model demonstration instead.")
            # Fallback to dummy data for model structure check
            BATCH_SIZE_DEMO = 4
            dummy_input_features = torch.randn(BATCH_SIZE_DEMO, SEQ_LEN, max_s_overall).to(DEVICE)
            dummy_sensor_mask = torch.ones(BATCH_SIZE_DEMO, max_s_overall).to(DEVICE)
            # For some sensors, set mask to 0 to simulate padding
            if max_s_overall > 1:
                dummy_sensor_mask[0, -1] = 0 # Example: last sensor of first batch item is padded
        else:
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False) # Small batch for demo
            print(f"Dataset loaded with {len(dataset)} samples.")
            # Get a sample batch for demonstration
            try:
                sample_batch = next(iter(dataloader))
                dummy_input_features = sample_batch["input_features"].to(DEVICE)
                dummy_sensor_mask = sample_batch["sensor_mask"].to(DEVICE)
                print(f"Sample batch input shape: {dummy_input_features.shape}, mask shape: {dummy_sensor_mask.shape}")
            except StopIteration:
                print("DataLoader is empty even though dataset reported non-zero length. Check dataset integrity.")
                raise # re-raise to signal issue

    except Exception as e:
        print(f"Error initializing dataset/dataloader: {e}. Using dummy data for model demonstration.")
        BATCH_SIZE_DEMO = 4
        max_s_overall = MAX_SENSORS_CAP if MAX_SENSORS_CAP > 0 else 10 # Ensure max_s_overall is non-zero
        dummy_input_features = torch.randn(BATCH_SIZE_DEMO, SEQ_LEN, max_s_overall).to(DEVICE)
        dummy_sensor_mask = torch.ones(BATCH_SIZE_DEMO, max_s_overall).to(DEVICE)
        if max_s_overall > 1 : dummy_sensor_mask[0,0] = 0


    # 3. Instantiate Model
    model = SimplifiedPredictionModel(
        max_sensors=max_s_overall,
        seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM,
        sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS,
        tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dropout=TCN_DROPOUT # Set to 0.0 for deterministic output if not using model.eval()
    ).to(DEVICE)
    model.eval() # Set to evaluation mode (affects dropout, etc.)

    # 4. Perform a forward pass (example)
    with torch.no_grad(): # No gradients needed for inference
        prediction_h5 = model(dummy_input_features, dummy_sensor_mask)

    print(f"Model instantiated on {DEVICE}.")
    print(f"Output prediction shape for H={PRED_HORIZON_TARGET}: {prediction_h5.shape}")
    # Expected shape: [BatchSize, MaxSensors, 1] (since we predict only one horizon)

    # You can inspect a prediction:
    # print("Example prediction for first item in batch, first active sensor:")
    # print(prediction_h5[0, torch.where(dummy_sensor_mask[0]>0)[0][0], 0].item())