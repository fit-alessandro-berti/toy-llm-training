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

# --- Configuration ---
BASE_DATA_DIR = "../../data/time_series/1"  # Example, adjust if needed
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAINING")  # <<< SET THIS
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # <<< SET THIS

# Model & Data Parameters
SEQ_LEN = 64
PRED_HORIZON_TARGET = 5  # Focusing on a single prediction horizon
MAX_SENSORS_CAP = 20  # Max sensors to consider from CSV (used for model_max_sensors_dim)

# Architectural Params
SENSOR_INPUT_DIM = 1

# I. Per-sensor Temporal Encoder (TCN based)
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

# II. Inter-sensor Representation (Reintroduced)
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

# Training Parameters (from previous successful iteration)
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-4
HUBER_DELTA = 5.0
GRADIENT_CLIP_NORM = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "ts_transformer_model_h5.pth"
PREPROCESSOR_SAVE_PATH = "ts_transformer_preprocessor_h5.npz"


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

    def forward(self, x):  # x: [Batch*MaxSensors, SeqLen, InputDim]
        x = self.input_proj(x);
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1);
        x_tcn_out = self.tcn_network(x)  # [B*MS, ProjDim, SL] -> [B*MS, TCN_OUT_DIM, SL]
        x_permuted_back = x_tcn_out.permute(0, 2, 1)  # [B*MS, SL, TCN_OUT_DIM]
        return self.final_norm(x_permuted_back)


# --- Inter-Sensor Transformer (Restored from original) ---
class InterSensorTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, max_sensors):
        super().__init__()
        # Positional encoding for sensors (learnable)
        self.pos_encoder_inter_sensor = nn.Parameter(torch.zeros(1, max_sensors, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True,
                                                   dim_feedforward=embed_dim * 2,
                                                   norm_first=True)  # norm_first is good practice
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask):  # x: [Batch, MaxSensors, embed_dim]
        # Add learnable positional encoding for each sensor
        x = x + self.pos_encoder_inter_sensor[:, :x.size(1), :]  # Ensure slicing if x.size(1) < max_sensors
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_norm(x)


# --- Data Handling Utilities ---
def get_sensor_columns(df_peek):
    sensor_cols = [c for c in df_peek.columns if c.startswith("Sensor")]
    if not sensor_cols:
        potential_cols = [c for c in df_peek.columns if df_peek[c].dtype in [np.float64, np.int64, np.float32]]
        sensor_cols = [c for c in potential_cols if
                       not any(kw in c.lower() for kw in ['time', 'date', 'label', 'failure', 'id', 'current_failure'])]
    return sensor_cols


def get_model_max_sensors_dim(file_paths, cap=MAX_SENSORS_CAP):
    max_s = 0;
    processed_files = 0
    for fp in file_paths:
        try:
            df_peek = pd.read_csv(fp, nrows=1)
            sensor_cols = get_sensor_columns(df_peek)
            if sensor_cols: max_s = max(max_s, len(sensor_cols)); processed_files += 1
        except Exception:
            continue  # Skip files that can't be read
    if processed_files == 0: return 0
    return min(max_s, cap) if cap > 0 and max_s > cap else max_s


def calculate_global_stats(file_paths, target_num_sensors, predefined_sensor_names=None):
    print(f"Calculating global statistics for up to {target_num_sensors} sensors...")
    sums = np.zeros(target_num_sensors, dtype=np.float64)
    sum_sqs = np.zeros(target_num_sensors, dtype=np.float64)
    counts = np.zeros(target_num_sensors, dtype=np.int64)

    # Establish canonical sensor names if not predefined
    # This version assumes we use the first `target_num_sensors` columns found in the first valid file
    # and hope other files are consistent or subset of these.
    # A truly robust solution would involve finding common column names or using a schema.
    canonical_sensor_names = list(predefined_sensor_names) if predefined_sensor_names else []
    first_file_processed_for_names = False

    for fp_idx, fp in enumerate(file_paths):
        try:
            df = pd.read_csv(fp, low_memory=False)
            current_file_sensor_cols = get_sensor_columns(df)
            if not current_file_sensor_cols: continue

            if not predefined_sensor_names and not first_file_processed_for_names:
                canonical_sensor_names = current_file_sensor_cols[:target_num_sensors]
                if len(canonical_sensor_names) < target_num_sensors:
                    print(
                        f"Warning: First file {fp} only has {len(canonical_sensor_names)} relevant sensor columns. Reducing target_num_sensors for stats to this amount.")
                    target_num_sensors = len(canonical_sensor_names)
                    # Resize accumulators if target_num_sensors changed
                    sums = sums[:target_num_sensors];
                    sum_sqs = sum_sqs[:target_num_sensors];
                    counts = counts[:target_num_sensors]
                if target_num_sensors == 0:
                    print(f"Error: No usable sensor columns from first file {fp}. Cannot calculate global stats.");
                    return None, None, []
                first_file_processed_for_names = True

            # Accumulate stats based on the order of canonical_sensor_names
            for i, name in enumerate(
                    canonical_sensor_names):  # Iterate up to the (potentially reduced) target_num_sensors
                if name in df.columns:
                    col_data = df[name].values.astype(np.float32)
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0:
                        sums[i] += valid_data.sum()
                        sum_sqs[i] += (valid_data ** 2).sum()
                        counts[i] += len(valid_data)
        except Exception as e:
            print(f"Warning: Skipping file {fp} during stats calc: {e}"); continue
        if (fp_idx + 1) % 20 == 0: print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for stats...")

    if not canonical_sensor_names or target_num_sensors == 0:
        print("Error: No canonical sensor names established or target_num_sensors is 0.");
        return None, None, []
    if np.sum(counts) == 0:
        print("Error: No data accumulated for global stats (all counts are zero).");
        return None, None, canonical_sensor_names

    # Filter out sensors that had no data points across all files
    valid_indices = counts > 0
    if not np.all(valid_indices):
        print(
            f"Warning: Some sensors had no data points. Stats computed for {np.sum(valid_indices)}/{len(counts)} sensors.")

    final_canonical_names = [name for i, name in enumerate(canonical_sensor_names) if valid_indices[i]]
    final_means = sums[valid_indices] / counts[valid_indices]
    final_stds = np.sqrt(sum_sqs[valid_indices] / counts[valid_indices] - final_means ** 2)
    final_stds[final_stds < 1e-8] = 1e-8

    if len(final_canonical_names) == 0: print("Error: No valid global statistics computed."); return None, None, []
    print(f"Global statistics calculated for {len(final_canonical_names)} sensors.")
    return final_means, final_stds, final_canonical_names


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizon, model_max_sensors_dim, global_means, global_stds,
                 canonical_sensor_names):
        self.data_dir = data_dir;
        self.seq_len = seq_len;
        self.pred_horizon = pred_horizon
        self.model_max_sensors_dim = model_max_sensors_dim  # Max dimension for padding model input
        self.global_means = global_means;
        self.global_stds = global_stds
        self.canonical_sensor_names = canonical_sensor_names  # Names corresponding to global_means/stds
        self.num_globally_normed_features = len(canonical_sensor_names)

        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"));
        self.data_cache = [];
        self.window_indices = []
        if not self.file_paths: print(f"ERROR: No CSV files found in directory: {data_dir}.")
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(
            f"Loading & normalizing data from {self.data_dir} using {self.num_globally_normed_features} global features ({self.canonical_sensor_names[:3]}...). Target model dim: {self.model_max_sensors_dim}")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping file {fp}: {e}"); continue

            features_normalized = np.full((len(df), self.num_globally_normed_features), np.nan, dtype=np.float32)
            sensors_present_in_this_file_count = 0
            for i, name in enumerate(self.canonical_sensor_names):
                if name in df.columns:
                    col_data = df[name].values.astype(np.float32)
                    features_normalized[:, i] = (col_data - self.global_means[i]) / self.global_stds[i]
                    sensors_present_in_this_file_count += 1

            if sensors_present_in_this_file_count == 0: continue

            self.data_cache.append({"features_normalized_globally": features_normalized})
            max_lookahead = self.pred_horizon
            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        if not self.data_cache: print(f"CRITICAL WARNING: No data loaded from {self.data_dir}."); return
        print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item_data = self.data_cache[file_idx]
        features_normalized_globally = item_data[
            "features_normalized_globally"]  # [file_len, num_globally_normed_features]

        input_slice_normed = features_normalized_globally[
                             window_start_idx: window_start_idx + self.seq_len]  # [SeqLen, num_globally_normed_features]

        # Initialize padded input for the model dimension
        padded_input = np.zeros((self.seq_len, self.model_max_sensors_dim), dtype=np.float32)
        sensor_mask = np.zeros(self.model_max_sensors_dim, dtype=np.float32)

        # Copy available globally normed features, up to model_max_sensors_dim
        num_to_copy = min(self.num_globally_normed_features, self.model_max_sensors_dim)
        padded_input[:, :num_to_copy] = input_slice_normed[:, :num_to_copy]

        # Create mask: a sensor is active if its data (after slicing and copying) is not all NaN for the window
        for k in range(num_to_copy):
            if not np.all(np.isnan(input_slice_normed[:, k])):  # If any value in the window for this sensor is not NaN
                sensor_mask[k] = 1.0

        # Replace any NaNs in padded_input with 0 (especially for sensors that were masked out or originally NaN)
        padded_input[np.isnan(padded_input)] = 0.0

        last_known_val = padded_input[-1, :].copy()  # From the NaN-handled, padded input

        delta_target = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        target_idx = window_start_idx + self.seq_len + self.pred_horizon - 1
        if target_idx < features_normalized_globally.shape[0]:
            target_values_all_normed_features = features_normalized_globally[target_idx,
                                                :]  # [num_globally_normed_features]
            for k in range(num_to_copy):  # Iterate over sensors we are actually using in the model input
                if sensor_mask[k] > 0 and not np.isnan(target_values_all_normed_features[k]):
                    delta_target[k] = target_values_all_normed_features[k] - last_known_val[k]

        return {"input_features": torch.from_numpy(padded_input), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_input": torch.from_numpy(last_known_val),
                "pred_delta_target_h5": torch.from_numpy(delta_target)}


# --- Time Series Transformer Model (TCN + InterSensorTransformer) ---
class TimeSeriesTransformerModel(nn.Module):
    def __init__(self, model_max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout,
                 transformer_d_model, transformer_nhead, transformer_nlayers):
        super().__init__()
        self.model_max_sensors = model_max_sensors  # Max number of sensors model is built for (padding dimension)
        self.seq_len = seq_len
        self.sensor_tcn_out_dim = sensor_tcn_out_dim
        self.transformer_d_model = transformer_d_model

        print(
            f"TimeSeriesTransformerModel init: model_max_sensors={model_max_sensors}, tcn_out_dim={sensor_tcn_out_dim}, transformer_d_model={transformer_d_model}")

        self.per_sensor_encoder = PerSensorEncoderTCN(
            sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
            seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)

        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_tcn_out_dim, transformer_d_model) \
            if sensor_tcn_out_dim != transformer_d_model else nn.Identity()

        self.inter_sensor_transformer = InterSensorTransformer(
            transformer_d_model, transformer_nhead, transformer_nlayers, model_max_sensors)

        # Prediction head input: TCN output (last step) + Inter-sensor Transformer output
        pred_head_input_dim = sensor_tcn_out_dim + transformer_d_model
        self.pred_head = nn.Linear(pred_head_input_dim, 1)  # Predicts 1 value (delta for H=5)

    def forward(self, x_features_globally_std, sensor_mask):
        # x_features_globally_std: [B, SeqLen, model_max_sensors], already globally standardized
        batch_size, seq_len, _ = x_features_globally_std.shape

        # 1. Early Masking for TCN input
        x_input_masked_for_tcn = x_features_globally_std.permute(0, 2, 1)  # -> [B, model_max_sensors, SeqLen]
        x_input_masked_for_tcn = x_input_masked_for_tcn * sensor_mask.unsqueeze(-1)  # Apply mask

        last_val_for_delta_recon = x_input_masked_for_tcn[:, :, -1].clone()  # [B, model_max_sensors]

        x_permuted_for_tcn_input = x_input_masked_for_tcn.permute(0, 2, 1)  # -> [B, SeqLen, model_max_sensors]
        x_reshaped_for_encoder = x_permuted_for_tcn_input.reshape(
            batch_size * self.model_max_sensors, seq_len, SENSOR_INPUT_DIM)

        # 2. Per-Sensor Encoding (TCN)
        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        # Output: [B*model_max_sensors, SeqLen, SENSOR_TCN_OUT_DIM]
        sensor_temporal_features = sensor_temporal_features_flat.reshape(
            batch_size, self.model_max_sensors, seq_len, self.sensor_tcn_out_dim)
        sensor_temporal_features = sensor_temporal_features * sensor_mask.view(batch_size, self.model_max_sensors, 1,
                                                                               1)  # Mask TCN output

        # 3. Prepare for Inter-Sensor Transformer
        # Pool TCN features over time for each sensor
        pooled_sensor_features = torch.mean(sensor_temporal_features,
                                            dim=2)  # [B, model_max_sensors, SENSOR_TCN_OUT_DIM]
        projected_for_inter_sensor = self.pooled_to_transformer_dim_proj(
            pooled_sensor_features)  # [B, model_max_sensors, TRANSFORMER_D_MODEL]

        # 4. Inter-Sensor Transformer
        transformer_padding_mask = (sensor_mask == 0)  # True for padded sensors
        cross_sensor_context = self.inter_sensor_transformer(projected_for_inter_sensor,
                                                             transformer_padding_mask)  # [B, model_max_sensors, TRANSFORMER_D_MODEL]
        cross_sensor_context = cross_sensor_context * sensor_mask.unsqueeze(-1)  # Mask transformer output

        # 5. Combine Features for Prediction Head
        tcn_features_last_step = sensor_temporal_features[:, :, -1, :]  # [B, model_max_sensors, SENSOR_TCN_OUT_DIM]
        combined_features_for_head = torch.cat([tcn_features_last_step, cross_sensor_context], dim=-1)
        # Shape: [B, model_max_sensors, SENSOR_TCN_OUT_DIM + TRANSFORMER_D_MODEL]

        # 6. Prediction Head
        pred_delta_normalized = self.pred_head(combined_features_for_head)  # [B, model_max_sensors, 1]

        # 7. Final Prediction (absolute value in globally standardized space)
        pred_abs_normalized = last_val_for_delta_recon.unsqueeze(-1) + pred_delta_normalized
        final_pred = pred_abs_normalized * sensor_mask.unsqueeze(-1)  # Mask final output

        return final_pred


# --- Training Function ---
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

    all_file_paths = train_file_paths + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_file_paths: print("ERROR: No CSV files in total for model dimensioning."); return

    model_max_sensors_dim = get_model_max_sensors_dim(all_file_paths, MAX_SENSORS_CAP)
    if model_max_sensors_dim == 0: print("ERROR: model_max_sensors_dim is 0."); return
    print(f"Model will be built for model_max_sensors_dim: {model_max_sensors_dim}")

    global_means, global_stds, canonical_sensor_names = calculate_global_stats(train_file_paths, model_max_sensors_dim)
    if global_means is None or len(canonical_sensor_names) == 0:
        print("ERROR: Failed to calculate global statistics.");
        return

    num_globally_normed_features = len(canonical_sensor_names)  # Actual number of features with stats
    print(f"Global stats computed for {num_globally_normed_features} features: {canonical_sensor_names[:5]}...")

    np.savez(PREPROCESSOR_SAVE_PATH,
             global_means=global_means, global_stds=global_stds,
             canonical_sensor_names=np.array(canonical_sensor_names, dtype=object),
             model_max_sensors_dim=model_max_sensors_dim,
             seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON_TARGET)
    print(f"Preprocessor config saved to {PREPROCESSOR_SAVE_PATH}")

    model = TimeSeriesTransformerModel(
        model_max_sensors=model_max_sensors_dim, seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM, tcn_levels=TCN_LEVELS,
        tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS
    ).to(DEVICE)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZON_TARGET, model_max_sensors_dim,
                                                  global_means, global_stds, canonical_sensor_names)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZON_TARGET, model_max_sensors_dim,
                                                  global_means, global_stds, canonical_sensor_names)
    if len(train_dataset) == 0: print("ERROR: Training dataset empty."); return
    if len(valid_dataset) == 0: print("Warning: Validation dataset empty.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=DEVICE.type == 'cuda')
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=DEVICE.type == 'cuda') if len(valid_dataset) > 0 else None

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')

    print("Starting training: TCN + InterSensorTransformer for H=5 prediction...")
    for epoch in range(EPOCHS):
        model.train();
        total_train_loss_epoch = 0;
        num_train_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(DEVICE)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_train_loss_epoch += batch_loss.item();
            num_train_batches += 1
            if batch_idx > 0 and batch_idx % (len(train_loader) // 5 if len(train_loader) >= 5 else 1) == 0: print(
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
    print(f"Training complete. Model: {MODEL_SAVE_PATH}, Preprocessor: {PREPROCESSOR_SAVE_PATH}")


if __name__ == '__main__':
    print("--- Script Version 4: TCN + InterSensorTransformer, Global Stats, No RevIN, Early Mask, Updated Params ---")
    print(f"IMPORTANT: TRAIN_DIR ('{TRAIN_DIR}') and VALID_DIR ('{VALID_DIR}') must be set correctly.")
    if BASE_DATA_DIR == "../../data/time_series/1": print(
        "\nWARNING: Using default example BASE_DATA_DIR. Paths might be incorrect.\n")
    train_and_save_model()
