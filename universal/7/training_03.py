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

# StandardScaler will not be used per-file anymore, but logic for global stats is similar
# from sklearn.preprocessing import StandardScaler # No longer needed for per-file scaling

# --- Configuration ---
BASE_DATA_DIR = "../../data/time_series/1"
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAINING")  # <<< SET THIS
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # <<< SET THIS

SEQ_LEN = 64
PRED_HORIZON_TARGET = 5
MAX_SENSORS_CAP = 20

SENSOR_INPUT_DIM = 1
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 3e-4  # Adjusted from suggestions
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-4
HUBER_DELTA = 5.0  # Adjusted from suggestions
GRADIENT_CLIP_NORM = 1.0  # Added from suggestions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "simplified_timeseries_model_h5_v3.pth"
PREPROCESSOR_SAVE_PATH = "simplified_preprocessor_config_h5_v3.npz"  # Will store global stats


# --- Helper: Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term);
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1), :]


# --- TCN Residual Block (Weight Norm Removed) ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.relu1 = nn.ReLU();
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.padding,
                               dilation=dilation)
        self.relu2 = nn.ReLU();
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x);
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu1(out);
        out = self.dropout1(out)
        out = self.conv2(out);
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu2(out);
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


# --- Per-Sensor TCN Encoder ---
class PerSensorEncoderTCN(nn.Module):
    def __init__(self, input_dim, proj_dim, tcn_out_dim, seq_len, num_levels, kernel_size, dropout):
        super(PerSensorEncoderTCN, self).__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=max(seq_len, 5000))
        tcn_blocks = []
        current_channels = proj_dim
        for i in range(num_levels):
            dilation_size = 2 ** i;
            out_channels_block = tcn_out_dim
            tcn_blocks.append(
                TemporalBlock(current_channels, out_channels_block, kernel_size, stride=1, dilation=dilation_size,
                              dropout=dropout))
            current_channels = out_channels_block
        self.tcn_network = nn.Sequential(*tcn_blocks)
        self.final_norm = nn.LayerNorm(tcn_out_dim)

    def forward(self, x):
        x = self.input_proj(x);
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1);
        x_tcn_out = self.tcn_network(x)
        x_permuted_back = x_tcn_out.permute(0, 2, 1)
        return self.final_norm(x_permuted_back)


# --- Data Handling Utilities ---
def get_sensor_columns(df_peek):
    """Helper to infer sensor columns if not explicitly named Sensor*."""
    sensor_cols = [c for c in df_peek.columns if c.startswith("Sensor")]
    if not sensor_cols:
        potential_cols = [c for c in df_peek.columns if df_peek[c].dtype in [np.float64, np.int64, np.float32]]
        sensor_cols = [c for c in potential_cols if
                       not any(kw in c.lower() for kw in ['time', 'date', 'label', 'failure', 'id', 'current_failure'])]
    return sensor_cols


def get_max_sensors_and_common_cols(file_paths, cap=MAX_SENSORS_CAP):
    max_s = 0
    all_sensor_col_sets = []
    processed_files = 0
    for fp in file_paths:
        try:
            df_peek = pd.read_csv(fp, nrows=1)
            sensor_cols = get_sensor_columns(df_peek)
            if sensor_cols:
                all_sensor_col_sets.append(set(sensor_cols))
                max_s = max(max_s, len(sensor_cols))
                processed_files += 1
        except Exception as e:
            print(f"Warning: Could not read {fp} for sensor discovery: {e}"); continue

    if processed_files == 0: return 0, []

    final_max_s = min(max_s, cap) if cap > 0 and max_s > cap else max_s

    # For simplicity in this version, we'll assume the first 'final_max_s'
    # columns encountered that are common across most files, or simply
    # rely on the dataset to pick the first N columns if names vary wildly.
    # A more robust approach would find the intersection of column names from all_sensor_col_sets
    # or use a predefined list. For now, we'll use the count.
    # The dataset will use its logic to select sensor columns up to final_max_s.
    # The global stats will be calculated based on the columns selected by the dataset.
    # This part is tricky without a strict schema. Let's assume `get_sensor_columns` used by
    # dataset and global stats calculation is consistent.
    # For global stats, we need a fixed list of target column *names* up to final_max_s.
    # We'll determine this during the global stats calculation pass.
    return final_max_s


def calculate_global_stats(file_paths, determined_max_sensors):
    print(f"Calculating global statistics for {determined_max_sensors} sensors...")
    # Initialize accumulators for sum, sum_sq, and count for each of the 'determined_max_sensors'
    # These will correspond to the first 'determined_max_sensors' sensor columns found and used.
    sums = np.zeros(determined_max_sensors, dtype=np.float64)
    sum_sqs = np.zeros(determined_max_sensors, dtype=np.float64)
    counts = np.zeros(determined_max_sensors, dtype=np.int64)
    # Store the canonical sensor column names that these stats correspond to
    canonical_sensor_names = None

    for fp_idx, fp in enumerate(file_paths):
        try:
            df = pd.read_csv(fp, low_memory=False)  # Read full file for stats
            current_sensor_cols = get_sensor_columns(df)

            if not current_sensor_cols: continue

            # On the first file, establish the canonical sensor names up to determined_max_sensors
            if canonical_sensor_names is None:
                canonical_sensor_names = current_sensor_cols[:determined_max_sensors]
                if len(canonical_sensor_names) < determined_max_sensors:
                    print(
                        f"Warning: First file {fp} has only {len(canonical_sensor_names)} sensors, less than determined_max_sensors {determined_max_sensors}. Stats will be based on fewer sensors.")
                    # Adjust determined_max_sensors if the very first file limits it. This is a simplification.
                    # A more robust way would be to find common columns across all files first.
                    sums = sums[:len(canonical_sensor_names)]
                    sum_sqs = sum_sqs[:len(canonical_sensor_names)]
                    counts = counts[:len(canonical_sensor_names)]
                    determined_max_sensors = len(canonical_sensor_names)
                    if determined_max_sensors == 0:
                        print("Error: No sensor columns usable from the first file for global stats.")
                        return None, None, []

            # Select data only for the canonical sensor names
            # Ensure all canonical names are present, or handle missing ones (e.g. skip file for that sensor)
            # For simplicity, we assume files consistently have these columns if they have data for them.
            # We take the first 'determined_max_sensors' from the current file's sensor_cols,
            # assuming they align with the canonical_sensor_names order.

            # Use only the columns that are part of the canonical set and present in the current file
            # This requires careful alignment if column order or presence varies.
            # Simplified: use the first 'num_to_process' columns from this file.
            num_to_process = min(len(current_sensor_cols), determined_max_sensors)

            # Data for selected columns from current file
            # Ensure we only try to access columns that exist in the dataframe 'df'
            # And align them with the 'canonical_sensor_names' indices

            # More robust:
            data_for_stats = np.full((len(df), determined_max_sensors), np.nan, dtype=np.float32)
            for i, name in enumerate(canonical_sensor_names):
                if name in df.columns:
                    data_for_stats[:, i] = df[name].values.astype(np.float32)

            for i in range(determined_max_sensors):
                col_data = data_for_stats[:, i]
                valid_data = col_data[~np.isnan(col_data)]
                if len(valid_data) > 0:
                    sums[i] += valid_data.sum()
                    sum_sqs[i] += (valid_data ** 2).sum()
                    counts[i] += len(valid_data)
        except Exception as e:
            print(f"Warning: Skipping file {fp} during global stats calculation due to error: {e}")
            continue
        if (fp_idx + 1) % 10 == 0: print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for stats...")

    if canonical_sensor_names is None or determined_max_sensors == 0:
        print("Error: Could not establish canonical sensor names or no data for stats.")
        return None, None, []

    # Replace counts of 0 with 1 to avoid division by zero, though mean/std would be ill-defined.
    # Filter out sensors with no data.
    valid_sensor_indices = [i for i, count in enumerate(counts) if count > 0]
    if len(valid_sensor_indices) < determined_max_sensors:
        print(
            f"Warning: Global stats could only be computed for {len(valid_sensor_indices)} out of {determined_max_sensors} target sensors due to missing data.")

    final_canonical_names = [canonical_sensor_names[i] for i in valid_sensor_indices]
    final_sums = sums[valid_sensor_indices]
    final_sum_sqs = sum_sqs[valid_sensor_indices]
    final_counts = counts[valid_sensor_indices]

    if len(final_counts) == 0:
        print("Error: No valid data found to calculate any global statistics.")
        return None, None, []

    means = final_sums / final_counts
    # Variance = E[X^2] - (E[X])^2
    stds = np.sqrt(final_sum_sqs / final_counts - means ** 2)
    stds[stds < 1e-8] = 1e-8  # Add epsilon to stds to prevent division by zero

    print("Global statistics calculated.")
    return means, stds, final_canonical_names


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizon, max_sensors_global, global_means, global_stds,
                 canonical_sensor_names):
        self.data_dir = data_dir;
        self.seq_len = seq_len;
        self.pred_horizon = pred_horizon;
        self.max_sensors_global = max_sensors_global  # This is the target dimension after padding
        self.global_means = global_means
        self.global_stds = global_stds
        self.canonical_sensor_names = canonical_sensor_names  # Names corresponding to global_means/stds

        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"));
        self.data_cache = [];
        self.window_indices = []
        if not self.file_paths: print(f"ERROR: No CSV files found in directory: {data_dir}.")
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(f"Loading and normalizing data from {self.data_dir} using global stats...")
        # The number of sensors for global stats is len(self.canonical_sensor_names)
        # self.max_sensors_global is the target dimension for padding.
        num_canonical_sensors = len(self.canonical_sensor_names)

        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping file {fp} due to loading error: {e}"); continue

            # Initialize a features array for this file based on canonical sensor order, padded with NaN
            # This features_normalized will have num_canonical_sensors columns
            features_normalized = np.full((len(df), num_canonical_sensors), np.nan, dtype=np.float32)
            num_actual_sensors_in_file_for_this_pass = 0

            for i, name in enumerate(self.canonical_sensor_names):
                if name in df.columns:
                    col_data = df[name].values.astype(np.float32)
                    features_normalized[:, i] = (col_data - self.global_means[i]) / self.global_stds[i]
                    num_actual_sensors_in_file_for_this_pass += 1  # Counts how many of the canonical sensors are in this file

            if num_actual_sensors_in_file_for_this_pass == 0:
                # print(f"Warning: File {fp} contained none of the canonical sensor columns. Skipping.")
                continue

            # Replace any NaNs that might have resulted from missing values in original data (not from missing columns)
            # with 0 AFTER normalization. This means 0 is the mean for that sensor.
            # However, if a whole column was NaN (missing canonical sensor), it stays NaN here.
            # For TCN input, NaNs will be an issue. They should be 0 for masked sensors.
            # The sensor_mask handles which of the self.max_sensors_global sensors are active.

            # features_normalized is now [len(df), num_canonical_sensors]
            # We need to pad/truncate this to self.max_sensors_global for the model input tensor.
            # This step is combined with windowing.

            self.data_cache.append({"features_normalized_globally": features_normalized,
                                    "num_canonical_sensors_present_in_file": num_canonical_sensors
                                    # Used for creating the mask correctly
                                    })
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
        item_data = self.data_cache[file_idx]
        # features_normalized_globally is [file_len, num_canonical_sensors]
        features_normalized_globally = item_data["features_normalized_globally"]

        # num_canonical_sensors is len(self.global_means)
        num_canonical_sensors = len(self.global_means)

        # Input sequence from the globally normalized data
        # This slice will be [seq_len, num_canonical_sensors]
        input_normalized_orig = features_normalized_globally[window_start_idx: window_start_idx + self.seq_len]

        # Pad/truncate to self.max_sensors_global
        # This is the fixed dimension the model expects.
        padded_input = np.zeros((self.seq_len, self.max_sensors_global), dtype=np.float32)

        # How many sensors to copy from input_normalized_orig to padded_input
        num_sensors_to_copy = min(num_canonical_sensors, self.max_sensors_global)
        padded_input[:, :num_sensors_to_copy] = input_normalized_orig[:, :num_sensors_to_copy]

        # Handle NaNs in padded_input (e.g. from sensors not in canonical_sensor_names but within max_sensors_global, or original NaNs)
        # For TCN, input should be numeric. If a canonical sensor was missing in a file, its column in features_normalized_globally is NaN.
        # These NaNs, when copied to padded_input, should become 0 because the sensor_mask will mark them as inactive.
        padded_input[np.isnan(padded_input)] = 0.0

        sensor_mask = np.zeros(self.max_sensors_global, dtype=np.float32)
        # The mask indicates which of the self.max_sensors_global slots are active.
        # This depends on how many of the canonical_sensors we decided to use, up to max_sensors_global.
        # If num_canonical_sensors < self.max_sensors_global, only first num_canonical_sensors are potentially active.
        # If num_canonical_sensors >= self.max_sensors_global, only first self.max_sensors_global are potentially active.
        # The mask should reflect which of the copied sensors were NOT originally NaN columns.

        # Simpler mask: active if it's one of the `num_sensors_to_copy` AND the original data for that sensor at that step wasn't all NaN.
        # For simplicity here: mark the first `num_sensors_to_copy` as active.
        # A more precise mask would check if `input_normalized_orig[:, :num_sensors_to_copy]` had non-NaN data for that sensor.
        # Given early masking for TCN (zeros for inactive), this simplified mask is okay.
        sensor_mask[:num_sensors_to_copy] = 1.0
        # Refine mask: if a column in input_normalized_orig (up to num_sensors_to_copy) was all NaNs for this window, it's effectively inactive.
        for k in range(num_sensors_to_copy):
            if np.all(
                    np.isnan(input_normalized_orig[:, k])):  # if entire window for this sensor is NaN (was not in file)
                sensor_mask[k] = 0.0

        last_known_val_at_input_end = np.zeros(self.max_sensors_global, dtype=np.float32)
        # input_orig is already globally normalized and potentially padded/truncated to max_sensors_global
        # use padded_input which is correctly shaped and NaN-handled
        last_known_val_at_input_end[:self.max_sensors_global] = padded_input[-1, :self.max_sensors_global]

        delta_target_h5 = np.zeros(self.max_sensors_global, dtype=np.float32)  # Delta in globally normalized space
        target_idx_h5 = window_start_idx + self.seq_len + self.pred_horizon - 1

        if target_idx_h5 < features_normalized_globally.shape[0]:
            # target_value_normalized is [num_canonical_sensors]
            target_value_normalized = features_normalized_globally[target_idx_h5, :]

            for k in range(num_sensors_to_copy):
                if sensor_mask[k] > 0 and not np.isnan(
                        target_value_normalized[k]):  # If sensor is active and target is not NaN
                    # last_known_val_at_input_end[k] is from padded_input, which had NaNs replaced by 0 if masked.
                    # So, use padded_input[-1, k] which is the value for active sensors.
                    delta_target_h5[k] = target_value_normalized[k] - padded_input[-1, k]
                # If sensor_mask[k] is 0, or target is NaN, delta_target_h5[k] remains 0. This is fine as loss will be masked.

        return {"input_features": torch.from_numpy(padded_input), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_input": torch.from_numpy(last_known_val_at_input_end),  # globally normalized
                "pred_delta_target_h5": torch.from_numpy(delta_target_h5)}  # globally normalized delta


# --- Simplified Prediction Model (RevIN completely removed, Early Masking) ---
class SimplifiedPredictionModel(nn.Module):
    def __init__(self, max_sensors, seq_len, sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout):
        super().__init__()
        self.max_sensors = max_sensors;
        self.seq_len = seq_len;
        self.sensor_tcn_out_dim = sensor_tcn_out_dim
        print("RevIN layer completely REMOVED. Model relies on external global standardization.")
        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)
        self.pred_head_h5 = nn.Linear(sensor_tcn_out_dim, 1)

    def forward(self, x_features_globally_std, sensor_mask):
        # x_features_globally_std: [B, SeqLen, MaxSensors], already globally standardized
        batch_size, seq_len, num_model_sensors = x_features_globally_std.shape

        # 1. Early Masking: Permute for sensor-wise ops, mask, then permute back for TCN input prep
        # Input for TCN should be 0 for masked sensors.
        x_input_masked = x_features_globally_std.permute(0, 2, 1)  # -> [B, MaxSensors, SeqLen]
        x_input_masked = x_input_masked * sensor_mask.unsqueeze(-1)  # Apply mask sensor-wise along sequence

        # This is the globally standardized and correctly masked input data.
        last_val_for_delta_recon = x_input_masked[:, :, -1]  # [B, MaxSensors], last value of each sensor's series

        # Prepare for PerSensorEncoderTCN:
        x_permuted_for_tcn_input = x_input_masked.permute(0, 2, 1)  # -> [B, SeqLen, MaxSensors]
        x_reshaped_for_encoder = x_permuted_for_tcn_input.reshape(batch_size * num_model_sensors, seq_len,
                                                                  SENSOR_INPUT_DIM)

        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, num_model_sensors, seq_len,
                                                                         self.sensor_tcn_out_dim)

        # Masking TCN output is still good practice to ensure padded sensor outputs are zero if TCN biases affected them.
        sensor_temporal_features = sensor_temporal_features * sensor_mask.view(batch_size, num_model_sensors, 1, 1)

        features_for_pred_head = sensor_temporal_features[:, :, -1, :]
        pred_delta_in_normalized_space = self.pred_head_h5(features_for_pred_head)
        pred_abs_in_normalized_space = last_val_for_delta_recon.unsqueeze(-1) + pred_delta_in_normalized_space

        # Final output is in the globally standardized space. Mask for safety.
        final_pred = pred_abs_in_normalized_space * sensor_mask.unsqueeze(-1)
        return final_pred


# --- Training Function (Adjusted parameters, Grad Clipping) ---
def train_and_save_model():
    print(f"Using device: {DEVICE}")
    print(f"TRAIN_DIR: {TRAIN_DIR}");
    print(f"VALID_DIR: {VALID_DIR}")
    if not (os.path.exists(TRAIN_DIR) and os.path.isdir(TRAIN_DIR)): print(
        f"ERROR: TRAIN_DIR '{TRAIN_DIR}' missing."); return
    if not (os.path.exists(VALID_DIR) and os.path.isdir(VALID_DIR)): print(
        f"ERROR: VALID_DIR '{VALID_DIR}' missing."); return

    train_file_paths = glob.glob(os.path.join(TRAIN_DIR, "*.csv"))
    if not train_file_paths: print("ERROR: No CSV files in TRAIN_DIR for global stats."); return

    # Determine max_s_overall from all data (train+valid) for consistent padding dimension
    all_file_paths = train_file_paths + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_file_paths: print("ERROR: No CSV files in total for model dimensioning."); return

    # This max_s_overall is the dimension for model's 'max_sensors' parameter
    model_max_sensors_dim = get_max_sensors_and_common_cols(all_file_paths, MAX_SENSORS_CAP)
    if model_max_sensors_dim == 0: print("ERROR: model_max_sensors_dim is 0."); return
    print(f"Model will be dimensioned for max_sensors_overall: {model_max_sensors_dim}")

    # Calculate global stats ONLY from training data, using model_max_sensors_dim as the target number of sensor features
    global_means, global_stds, canonical_sensor_names = calculate_global_stats(train_file_paths, model_max_sensors_dim)
    if global_means is None or len(canonical_sensor_names) == 0:
        print("ERROR: Failed to calculate global statistics or no canonical sensors found.");
        return

    num_globally_normed_features = len(canonical_sensor_names)
    print(
        f"Global stats calculated for {num_globally_normed_features} canonical sensor features: {canonical_sensor_names}")

    np.savez(PREPROCESSOR_SAVE_PATH,
             global_means=global_means, global_stds=global_stds,
             canonical_sensor_names=np.array(canonical_sensor_names, dtype=object),  # Save as object array for names
             model_max_sensors_dim=model_max_sensors_dim,  # The padding dimension
             seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON_TARGET)
    print(f"Preprocessor config (global stats & names) saved to {PREPROCESSOR_SAVE_PATH}")

    model = SimplifiedPredictionModel(
        max_sensors=model_max_sensors_dim,  # Model is padded/handles up to this many
        seq_len=SEQ_LEN, sensor_input_dim=SENSOR_INPUT_DIM,
        sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM, sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT
    ).to(DEVICE)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZON_TARGET, model_max_sensors_dim,
                                                  global_means, global_stds, canonical_sensor_names)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZON_TARGET, model_max_sensors_dim,
                                                  global_means, global_stds,
                                                  canonical_sensor_names)  # Use same train stats for valid
    if len(train_dataset) == 0: print("ERROR: Training dataset empty."); return
    if len(valid_dataset) == 0: print("Warning: Validation dataset empty.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=DEVICE.type == 'cuda')
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=DEVICE.type == 'cuda') if len(valid_dataset) > 0 else None

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')

    print("Starting training with global stats, no RevIN, early masking, new LR/Delta/Clip...")
    for epoch in range(EPOCHS):
        model.train();
        total_train_loss_epoch = 0;
        num_train_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(DEVICE)  # Globally standardized
            sensor_mask = batch["sensor_mask"].to(DEVICE)
            last_known = batch["last_known_values_input"].to(DEVICE)
            delta_target = batch["pred_delta_target_h5"].to(DEVICE)
            optimizer.zero_grad()
            pred_abs_globally_std = model(input_features, sensor_mask)
            abs_target_globally_std = last_known.unsqueeze(-1) + delta_target.unsqueeze(-1)
            loss_elements = loss_fn(pred_abs_globally_std, abs_target_globally_std)
            active_loss_elements = loss_elements[sensor_mask.bool().unsqueeze(-1)]
            batch_loss = active_loss_elements.mean() if active_loss_elements.numel() > 0 else torch.tensor(0.0,
                                                                                                           device=DEVICE,
                                                                                                           requires_grad=True)
            if torch.isnan(batch_loss) or torch.isinf(batch_loss): print(
                f"Warning: NaN/Inf loss E{epoch + 1} B{batch_idx + 1}. Skip."); continue
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)  # Gradient Clipping
            optimizer.step()
            total_train_loss_epoch += batch_loss.item();
            num_train_batches += 1
            if batch_idx % (len(train_loader) // 5 if len(train_loader) >= 5 else 1) == 0: print(
                f"E{epoch + 1}/{EPOCHS} B{batch_idx + 1}/{len(train_loader)} | Train Loss: {batch_loss.item():.4f}")
        avg_train_loss = total_train_loss_epoch / num_train_batches if num_train_batches > 0 else float('nan')
        print(f"E{epoch + 1}/{EPOCHS} | Avg Train Loss: {avg_train_loss:.4f}")
        if valid_loader:
            model.eval();
            total_valid_loss_epoch = 0;
            num_valid_batches = 0
            with torch.no_grad():
                for batch in valid_loader:
                    input_features = batch["input_features"].to(DEVICE);
                    sensor_mask = batch["sensor_mask"].to(DEVICE)
                    last_known = batch["last_known_values_input"].to(DEVICE);
                    delta_target = batch["pred_delta_target_h5"].to(DEVICE)
                    pred_abs_globally_std = model(input_features, sensor_mask)
                    abs_target_globally_std = last_known.unsqueeze(-1) + delta_target.unsqueeze(-1)
                    loss_elements = loss_fn(pred_abs_globally_std, abs_target_globally_std)
                    active_loss_elements = loss_elements[sensor_mask.bool().unsqueeze(-1)]
                    if active_loss_elements.numel() > 0:
                        batch_loss = active_loss_elements.mean()
                        if not (torch.isnan(batch_loss) or torch.isinf(
                            batch_loss)): total_valid_loss_epoch += batch_loss.item(); num_valid_batches += 1
            avg_valid_loss = total_valid_loss_epoch / num_valid_batches if num_valid_batches > 0 else float('nan')
            print(f"E{epoch + 1}/{EPOCHS} | Avg Valid Loss: {avg_valid_loss:.4f}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model: {MODEL_SAVE_PATH}, Preprocessor (global stats): {PREPROCESSOR_SAVE_PATH}")


if __name__ == '__main__':
    print("--- Script Version 3: Global Stats, No RevIN, Early Mask, LR/Delta/Clip Updated ---")
    print(f"IMPORTANT: TRAIN_DIR ('{TRAIN_DIR}') and VALID_DIR ('{VALID_DIR}') must be set correctly.")
    if BASE_DATA_DIR == "../../data/time_series/1": print(
        "\nWARNING: Using default example BASE_DATA_DIR. Paths might be incorrect.\n")
    train_and_save_model()