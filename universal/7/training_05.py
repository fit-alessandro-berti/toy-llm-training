import os
import glob
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# --- Configuration ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1"  # Example, adjust if needed
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAINING")  # <<< SET THIS
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # <<< SET THIS

# Model & Task Parameters (from original, adapted where necessary)
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]  # For forecasting task
FAIL_HORIZONS = [3, 5, 10]  # For failure prediction task
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0]  # For RCA task, using the first fail horizon
MAX_SENSORS_CAP = 20

# Architectural Params
SENSOR_INPUT_DIM = 1

# I. Per-sensor Temporal Encoder (TCN based)
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1  # Maintained from converging script

# II. Inter-sensor Representation
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

# III. Mixture-of-Experts (MoE) - Restored
MOE_GLOBAL_INPUT_DIM = TRANSFORMER_D_MODEL  # Input to MoE gating/experts
NUM_EXPERTS_PER_TASK = 8
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64  # Output dim of each expert's processing
EXPERT_DROPOUT_RATE = 0.1
AUX_LOSS_COEFF = 0.01

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 40  # Maintained from converging script (original was 3, can be adjusted)
LEARNING_RATE = 3e-4  # Maintained from converging script (original was 3e-4)
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2  # From original (converging script used 1e-4)
GRAD_CLIP_MAX_NORM = 1.0  # From original
WARMUP_RATIO = 0.05  # For learning rate scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == 'cuda'  # Enable AMP if CUDA is available

MODEL_SAVE_PATH = "foundation_multitask_model.pth"
PREPROCESSOR_SAVE_PATH = "foundation_multitask_preprocessor.npz"

# Loss Function Parameters
HUBER_DELTA = 1.0  # For forecasting loss (original had 1.0, converging used 5.0. Reverting to original for multi-task)
FOCAL_ALPHA_PARAM = 0.25  # For Focal Loss
FOCAL_GAMMA = 2.0  # For Focal Loss

# Loss weights (from original)
W_PRED = 1.0
W_FAIL = 1.0
W_RCA = 0.5


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


# --- TCN Residual Block (Weight Norm REMOVED for convergence) ---
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
        x_tcn_out = self.tcn_network(x)
        x_permuted_back = x_tcn_out.permute(0, 2, 1)
        return self.final_norm(x_permuted_back)


# --- Inter-Sensor Transformer ---
class InterSensorTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, max_sensors):
        super().__init__()
        self.pos_encoder_inter_sensor = nn.Parameter(torch.zeros(1, max_sensors, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True,
                                                   dim_feedforward=embed_dim * 2, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask):
        x = x + self.pos_encoder_inter_sensor[:, :x.size(1), :]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_norm(x)


# --- MoE Components (Restored) ---
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__();
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x): return self.fc(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__();
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x): return self.fc(x)


# --- Focal Loss (Restored) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha_param = alpha;
        self.gamma = gamma;
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')  # Important for element-wise before reduction

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets);
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha_param + (1.0 - targets) * (1.0 - self.alpha_param)
        focal_term = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        return focal_term  # Return element-wise if no reduction specified for custom handling


# --- Data Handling Utilities ---
def get_sensor_columns(df_peek):
    sensor_cols = [c for c in df_peek.columns if c.startswith("Sensor")]
    if not sensor_cols:
        potential_cols = [c for c in df_peek.columns if df_peek[c].dtype in [np.float64, np.int64, np.float32]]
        sensor_cols = [c for c in potential_cols if
                       not any(kw in c.lower() for kw in ['time', 'date', 'label', 'failure', 'id', 'current_failure'])]
    return sensor_cols


def get_model_max_sensors_dim(file_paths, cap=MAX_SENSORS_CAP):  # Renamed from get_max_sensors_and_common_cols
    max_s = 0;
    processed_files = 0
    for fp in file_paths:
        try:
            df_peek = pd.read_csv(fp, nrows=1);
            sensor_cols = get_sensor_columns(df_peek)
            if sensor_cols: max_s = max(max_s, len(sensor_cols)); processed_files += 1
        except Exception:
            continue
    if processed_files == 0: return 0
    return min(max_s, cap) if cap > 0 and max_s > cap else max_s


def calculate_global_stats(file_paths,
                           target_num_sensors):  # Simplified from previous, assuming target_num_sensors is the number of columns for stats
    print(f"Calculating global statistics for up to {target_num_sensors} sensors...")
    sums = np.zeros(target_num_sensors, dtype=np.float64)
    sum_sqs = np.zeros(target_num_sensors, dtype=np.float64)
    counts = np.zeros(target_num_sensors, dtype=np.int64)
    canonical_sensor_names = None  # Will be established from the first file

    for fp_idx, fp in enumerate(file_paths):
        try:
            df = pd.read_csv(fp, low_memory=False)
            current_file_sensor_cols = get_sensor_columns(df)
            if not current_file_sensor_cols: continue

            if canonical_sensor_names is None:  # Establish from first valid file
                canonical_sensor_names = current_file_sensor_cols[:target_num_sensors]
                if len(canonical_sensor_names) < target_num_sensors:
                    print(
                        f"Warning: First file for stats ({fp}) has {len(canonical_sensor_names)} sensors, less than target {target_num_sensors}. Adjusting.")
                    target_num_sensors = len(canonical_sensor_names)
                    sums = sums[:target_num_sensors];
                    sum_sqs = sum_sqs[:target_num_sensors];
                    counts = counts[:target_num_sensors]
                if target_num_sensors == 0: print(
                    "Error: No usable sensor columns from first file."); return None, None, []

            # Use the established canonical_sensor_names order for accumulation
            for i, name in enumerate(canonical_sensor_names):
                if name in df.columns:
                    col_data = df[name].values.astype(np.float32)
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0:
                        sums[i] += valid_data.sum();
                        sum_sqs[i] += (valid_data ** 2).sum();
                        counts[i] += len(valid_data)
        except Exception as e:
            print(f"Warning: Skipping file {fp} during stats calc: {e}"); continue
        if (fp_idx + 1) % 20 == 0: print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for stats...")

    if canonical_sensor_names is None or target_num_sensors == 0 or np.sum(counts) == 0:
        print("Error: Failed to calculate global stats (no data or no canonical names).");
        return None, None, []

    valid_indices = counts > 0
    final_canonical_names = [name for i, name in enumerate(canonical_sensor_names) if valid_indices[i]]
    final_means = sums[valid_indices] / counts[valid_indices]
    final_stds = np.sqrt(sum_sqs[valid_indices] / counts[valid_indices] - final_means ** 2)
    final_stds[final_stds < 1e-8] = 1e-8
    if len(final_canonical_names) == 0: print("Error: No valid global stats computed."); return None, None, []
    print(f"Global stats calculated for {len(final_canonical_names)} sensors.")
    return final_means, final_stds, final_canonical_names


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizons, fail_horizons, rca_failure_lookahead,
                 model_max_sensors_dim, global_means, global_stds, canonical_sensor_names):
        self.data_dir = data_dir;
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons;
        self.fail_horizons = fail_horizons
        self.rca_failure_lookahead = rca_failure_lookahead
        self.model_max_sensors_dim = model_max_sensors_dim
        self.global_means = global_means;
        self.global_stds = global_stds
        self.canonical_sensor_names = canonical_sensor_names
        self.num_globally_normed_features = len(canonical_sensor_names)

        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"));
        self.data_cache = [];
        self.window_indices = []
        if not self.file_paths: print(f"ERROR: No CSV files in {data_dir}.")
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(
            f"Loading & normalizing data from {self.data_dir} using {self.num_globally_normed_features} global features. Target model dim: {self.model_max_sensors_dim}")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping file {fp}: {e}"); continue

            raw_features_from_canonical_cols = np.full((len(df), self.num_globally_normed_features), np.nan,
                                                       dtype=np.float32)
            for i, name in enumerate(self.canonical_sensor_names):
                if name in df.columns: raw_features_from_canonical_cols[:, i] = df[name].values.astype(np.float32)

            if np.all(np.isnan(raw_features_from_canonical_cols)): continue  # Skip if no canonical sensors found

            # Normalize the raw features using global stats
            features_normalized_globally = np.full_like(raw_features_from_canonical_cols, np.nan)
            for i in range(self.num_globally_normed_features):
                valid_mask = ~np.isnan(raw_features_from_canonical_cols[:, i])
                features_normalized_globally[valid_mask, i] = \
                    (raw_features_from_canonical_cols[valid_mask, i] - self.global_means[i]) / self.global_stds[i]

            failure_flags = None
            if "CURRENT_FAILURE" in df.columns:
                failure_flags = df["CURRENT_FAILURE"].values.astype(np.int64)
            else:
                print(
                    f"Warning: 'CURRENT_FAILURE' column not found in {fp}. Failure/RCA targets will be zero."); failure_flags = np.zeros(
                    len(df), dtype=np.int64)

            self.data_cache.append({
                "raw_features_globally_aligned": raw_features_from_canonical_cols,
                # Store raw for RCA calculation if needed by original logic
                "features_normalized_globally": features_normalized_globally,
                "failure_flags": failure_flags
            })
            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons), self.rca_failure_lookahead)
            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        if not self.data_cache: print(f"CRITICAL WARNING: No data loaded from {self.data_dir}."); return
        print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item_data = self.data_cache[file_idx]
        raw_features_aligned = item_data["raw_features_globally_aligned"]  # [file_len, num_globally_normed_features]
        features_normalized_aligned = item_data[
            "features_normalized_globally"]  # [file_len, num_globally_normed_features]
        flags_full = item_data["failure_flags"]

        input_slice_normed = features_normalized_aligned[window_start_idx: window_start_idx + self.seq_len]

        padded_input_normed = np.zeros((self.seq_len, self.model_max_sensors_dim), dtype=np.float32)
        sensor_mask = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        num_to_copy = min(self.num_globally_normed_features, self.model_max_sensors_dim)

        padded_input_normed[:, :num_to_copy] = input_slice_normed[:, :num_to_copy]
        for k in range(num_to_copy):
            if not np.all(np.isnan(input_slice_normed[:, k])): sensor_mask[k] = 1.0
        padded_input_normed[np.isnan(padded_input_normed)] = 0.0
        last_known_normed = padded_input_normed[-1, :].copy()

        # Forecasting Targets (delta in normalized space)
        delta_targets_normed = np.zeros((self.model_max_sensors_dim, len(self.pred_horizons)), dtype=np.float32)
        for i_h, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < features_normalized_aligned.shape[0]:
                target_values_all_normed = features_normalized_aligned[target_idx, :]
                for k in range(num_to_copy):
                    if sensor_mask[k] > 0 and not np.isnan(target_values_all_normed[k]):
                        delta_targets_normed[k, i_h] = target_values_all_normed[k] - last_known_normed[k]

        # Failure Targets
        fail_targets = np.zeros(len(self.fail_horizons), dtype=np.float32)
        for i_fh, fh in enumerate(self.fail_horizons):
            start, end = window_start_idx + self.seq_len, window_start_idx + self.seq_len + fh
            if end <= len(flags_full) and np.any(flags_full[start:end]):
                fail_targets[i_fh] = 1.0

        # RCA Targets (based on original script's logic, using raw features for deviation)
        rca_targets = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        start_r, end_r = window_start_idx + self.seq_len, window_start_idx + self.seq_len + self.rca_failure_lookahead
        if end_r <= len(flags_full) and np.any(
                flags_full[start_r:end_r]):  # If a failure occurs in the lookahead window
            # Use raw_features_aligned for RCA logic to match original script's intent (deviation from historical raw values)
            current_window_raw = raw_features_aligned[window_start_idx: window_start_idx + self.seq_len,
                                 :]  # [SeqLen, num_globally_normed_features]
            future_lookahead_raw = raw_features_aligned[start_r:end_r, :]  # [Lookahead, num_globally_normed_features]

            for k in range(num_to_copy):  # For sensors active in the model input
                if sensor_mask[k] > 0:  # Only for active sensors
                    sensor_data_current_window_raw = current_window_raw[:, k]
                    sensor_data_future_lookahead_raw = future_lookahead_raw[:, k]

                    valid_current_raw = sensor_data_current_window_raw[~np.isnan(sensor_data_current_window_raw)]
                    valid_future_raw = sensor_data_future_lookahead_raw[~np.isnan(sensor_data_future_lookahead_raw)]

                    if len(valid_current_raw) > 0 and len(valid_future_raw) > 0:
                        mean_current_raw = np.mean(valid_current_raw)
                        std_current_raw = np.std(valid_current_raw)
                        std_current_raw = max(std_current_raw, 1e-6)  # Avoid division by zero

                        # Check if any point in the lookahead window deviates significantly
                        if np.any(np.abs(valid_future_raw - mean_current_raw) > 3 * std_current_raw):
                            rca_targets[k] = 1.0

        return {"input_features": torch.from_numpy(padded_input_normed), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_globally_std": torch.from_numpy(last_known_normed),
                "pred_delta_targets_globally_std": torch.from_numpy(delta_targets_normed),
                "fail_targets": torch.from_numpy(fail_targets), "rca_targets": torch.from_numpy(rca_targets)}


# --- Foundational Multi-Task Model (Restored and Adapted) ---
class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, model_max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 moe_global_input_dim, num_experts_per_task, moe_hidden_dim_expert, moe_output_dim, expert_dropout_rate,
                 pred_horizons_len, fail_horizons_len):  # Removed revin_affine
        super().__init__()
        self.model_max_sensors = model_max_sensors;
        self.seq_len = seq_len;
        self.num_experts_per_task = num_experts_per_task;
        self.moe_output_dim = moe_output_dim

        # RevIN layer is REMOVED. Input data is expected to be globally standardized.
        print("FoundationalTimeSeriesModel: RevIN is REMOVED. Expecting globally standardized input.")

        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)
        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_tcn_out_dim, transformer_d_model) \
            if sensor_tcn_out_dim != transformer_d_model else nn.Identity()
        self.inter_sensor_transformer = InterSensorTransformer(transformer_d_model, transformer_nhead,
                                                               transformer_nlayers, model_max_sensors)
        # MoE Components
        self.experts_forecast = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_forecast = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.experts_fail = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_fail = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.experts_rca = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_rca = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.expert_dropout = nn.Dropout(expert_dropout_rate)

        # Task-specific Heads
        final_combined_feat_dim_last_step = sensor_tcn_out_dim + transformer_d_model  # Features from TCN (last step) + Transformer
        self.pred_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)  # Failure head uses only MoE output
        self.rca_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, 1)  # RCA per sensor

    def _apply_moe_switch(self, global_moe_input, gating_network, expert_pool):
        gating_logits = gating_network(global_moe_input)  # [B, num_experts]
        router_probs = torch.softmax(gating_logits, dim=-1)

        # For training, select top-1 expert per sample (Switch Transformer like)
        # For simplicity and original script behavior, we use argmax based one-hot selection.
        chosen_expert_indices = torch.argmax(gating_logits, dim=-1)  # [B]
        one_hot_selection = F.one_hot(chosen_expert_indices, num_classes=len(expert_pool)).float()  # [B, num_experts]

        all_expert_outputs = torch.stack([expert(global_moe_input) for expert in expert_pool],
                                         dim=1)  # [B, num_experts, moe_output_dim]

        moe_task_output = torch.sum(all_expert_outputs * one_hot_selection.unsqueeze(-1), dim=1)  # [B, moe_output_dim]

        if self.training and self.expert_dropout.p > 0:  # Check if dropout is actually enabled
            moe_task_output = self.expert_dropout(moe_task_output)

        # For auxiliary loss (load balancing)
        fi = one_hot_selection.mean(dim=0)  # Fraction of examples routed to expert i
        Pi = router_probs.mean(dim=0)  # Average router probability for expert i
        return moe_task_output, fi, Pi

    def forward(self, x_features_globally_std, sensor_mask):
        batch_size, seq_len, _ = x_features_globally_std.shape

        # Input x_features_globally_std is already standardized and has early masking applied before TCN
        x_input_masked_for_tcn = x_features_globally_std.permute(0, 2, 1)  # -> [B, model_max_sensors, SeqLen]
        x_input_masked_for_tcn = x_input_masked_for_tcn * sensor_mask.unsqueeze(-1)  # Early mask

        last_val_globally_std = x_input_masked_for_tcn[:, :, -1].clone()  # [B, model_max_sensors]

        x_permuted_for_tcn_input = x_input_masked_for_tcn.permute(0, 2, 1)
        x_reshaped_for_encoder = x_permuted_for_tcn_input.reshape(batch_size * self.model_max_sensors, seq_len,
                                                                  SENSOR_INPUT_DIM)

        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, self.model_max_sensors, seq_len,
                                                                         SENSOR_TCN_OUT_DIM)
        sensor_temporal_features = sensor_temporal_features * sensor_mask.view(batch_size, self.model_max_sensors, 1, 1)

        pooled_sensor_features = torch.mean(sensor_temporal_features, dim=2)
        projected_for_inter_sensor = self.pooled_to_transformer_dim_proj(pooled_sensor_features)
        transformer_padding_mask = (sensor_mask == 0)
        cross_sensor_context = self.inter_sensor_transformer(projected_for_inter_sensor, transformer_padding_mask)
        cross_sensor_context = cross_sensor_context * sensor_mask.unsqueeze(
            -1)  # [B, model_max_sensors, TRANSFORMER_D_MODEL]

        # Global MoE input: average of active sensor representations from transformer
        global_moe_input_sum = (cross_sensor_context * sensor_mask.unsqueeze(-1)).sum(dim=1)  # Sum over sensors
        active_sensors_per_batch = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_moe_input = global_moe_input_sum / active_sensors_per_batch  # [B, TRANSFORMER_D_MODEL]

        # Apply MoE for each task
        moe_forecast_output, fi_f, Pi_f = self._apply_moe_switch(global_moe_input, self.gating_forecast,
                                                                 self.experts_forecast)
        moe_fail_output, fi_fail, Pi_fail = self._apply_moe_switch(global_moe_input, self.gating_fail,
                                                                   self.experts_fail)
        moe_rca_output, fi_rca, Pi_rca = self._apply_moe_switch(global_moe_input, self.gating_rca, self.experts_rca)
        aux_loss_terms = {"forecast": (fi_f, Pi_f), "fail": (fi_fail, Pi_fail), "rca": (fi_rca, Pi_rca)}

        # Prepare features for prediction heads
        tcn_features_last_step = sensor_temporal_features[:, :, -1, :]  # [B, model_max_sensors, SENSOR_TCN_OUT_DIM]
        # Features for heads that use per-sensor combined info:
        combined_sensor_features_last_step = torch.cat([tcn_features_last_step, cross_sensor_context], dim=-1)

        # Forecasting head
        moe_f_expanded = moe_forecast_output.unsqueeze(1).expand(-1, self.model_max_sensors,
                                                                 -1)  # Expand to each sensor
        pred_head_input = torch.cat([combined_sensor_features_last_step, moe_f_expanded], dim=-1)
        pred_delta_globally_std = self.pred_head(pred_head_input)  # [B, model_max_sensors, pred_horizons_len]
        # Output is delta in globally standardized space. Add last known globally standardized value.
        pred_abs_globally_std = last_val_globally_std.unsqueeze(-1) + pred_delta_globally_std
        pred_abs_globally_std = pred_abs_globally_std * sensor_mask.unsqueeze(-1)  # Mask final forecast

        # Failure prediction head (uses only MoE output)
        fail_logits = self.fail_head(moe_fail_output)  # [B, fail_horizons_len]

        # RCA prediction head
        moe_r_expanded = moe_rca_output.unsqueeze(1).expand(-1, self.model_max_sensors, -1)
        rca_head_input = torch.cat([combined_sensor_features_last_step, moe_r_expanded], dim=-1)
        rca_logits = self.rca_head(rca_head_input).squeeze(-1)  # [B, model_max_sensors]
        # rca_logits are per sensor, apply mask in loss calculation if needed for inactive sensors

        return pred_abs_globally_std, fail_logits, rca_logits, aux_loss_terms


# --- Training Function ---
def train_and_save_model():
    print(f"Device: {DEVICE}, AMP Enabled: {AMP_ENABLED}")
    print(f"TRAIN_DIR: {TRAIN_DIR}");
    print(f"VALID_DIR: {VALID_DIR}")
    if not (os.path.exists(TRAIN_DIR) and os.path.isdir(TRAIN_DIR)): print(
        f"ERROR: TRAIN_DIR '{TRAIN_DIR}' missing."); return
    if not (os.path.exists(VALID_DIR) and os.path.isdir(VALID_DIR)): print(
        f"ERROR: VALID_DIR '{VALID_DIR}' missing."); return

    train_file_paths = glob.glob(os.path.join(TRAIN_DIR, "*.csv"))
    if not train_file_paths: print("ERROR: No CSV files in TRAIN_DIR."); return
    all_file_paths = train_file_paths + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_file_paths: print("ERROR: No CSV files in total."); return

    model_max_sensors_dim = get_model_max_sensors_dim(all_file_paths, MAX_SENSORS_CAP)
    if model_max_sensors_dim == 0: print("ERROR: model_max_sensors_dim is 0."); return
    print(f"Model built for model_max_sensors_dim: {model_max_sensors_dim}")

    global_means, global_stds, canonical_sensor_names = calculate_global_stats(train_file_paths, model_max_sensors_dim)
    if global_means is None or len(canonical_sensor_names) == 0: print(
        "ERROR: Failed to calculate global statistics."); return

    num_globally_normed_features = len(canonical_sensor_names)
    print(
        f"Global stats for {num_globally_normed_features} features: {canonical_sensor_names[:min(5, len(canonical_sensor_names))]}...")

    np.savez(PREPROCESSOR_SAVE_PATH, global_means=global_means, global_stds=global_stds,
             canonical_sensor_names=np.array(canonical_sensor_names, dtype=object),
             model_max_sensors_dim=model_max_sensors_dim, seq_len=SEQ_LEN,
             pred_horizons=np.array(PRED_HORIZONS), fail_horizons=np.array(FAIL_HORIZONS),
             rca_failure_lookahead=RCA_FAILURE_LOOKAHEAD)
    print(f"Preprocessor config saved to {PREPROCESSOR_SAVE_PATH}")

    model = FoundationalTimeSeriesModel(
        model_max_sensors=model_max_sensors_dim, seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM, tcn_levels=TCN_LEVELS,
        tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        moe_global_input_dim=MOE_GLOBAL_INPUT_DIM, num_experts_per_task=NUM_EXPERTS_PER_TASK,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        expert_dropout_rate=EXPERT_DROPOUT_RATE,
        pred_horizons_len=len(PRED_HORIZONS), fail_horizons_len=len(FAIL_HORIZONS)
    ).to(DEVICE)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD,
                                                  model_max_sensors_dim, global_means, global_stds,
                                                  canonical_sensor_names)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD,
                                                  model_max_sensors_dim, global_means, global_stds,
                                                  canonical_sensor_names)
    if len(train_dataset) == 0: print("ERROR: Training dataset empty."); return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=AMP_ENABLED)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=AMP_ENABLED) if len(valid_dataset) > 0 else None

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)

    # LR Scheduler (from original)
    num_training_steps = EPOCHS * len(train_loader)
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(enabled=AMP_ENABLED)

    huber_loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')  # For forecasting
    focal_loss_elementwise = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA,
                                       reduction='none')  # For fail/RCA elements
    focal_loss_mean = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA, reduction='mean')  # For fail task reduction

    print("Starting multi-task training (Forecast, Fail, RCA) with MoE...")
    for epoch in range(EPOCHS):
        model.train();
        total_loss_epoch, total_lp_epoch, total_lf_epoch, total_lr_epoch, total_laux_epoch = 0, 0, 0, 0, 0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            input_feat = batch["input_features"].to(DEVICE)
            sensor_m = batch["sensor_mask"].to(DEVICE)
            last_k_std = batch["last_known_values_globally_std"].to(DEVICE)
            delta_tgt_std = batch["pred_delta_targets_globally_std"].to(DEVICE)
            fail_tgt = batch["fail_targets"].to(DEVICE)
            rca_tgt = batch["rca_targets"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)  # set_to_none for potentially better performance
            with autocast(enabled=AMP_ENABLED):
                pred_abs_std, fail_logits, rca_logits, aux_terms = model(input_feat, sensor_m)

                # 1. Forecasting Loss (Lp)
                abs_target_std = last_k_std.unsqueeze(
                    -1) + delta_tgt_std  # Target is [B, model_max_sensors, pred_horizons_len]
                lp_elements = huber_loss_fn(pred_abs_std, abs_target_std)  # [B, model_max_sensors, pred_horizons_len]
                lp_masked = lp_elements * sensor_m.unsqueeze(-1)  # Apply sensor mask
                # Average over active sensor-horizon predictions
                num_active_forecast_elements = (sensor_m.sum() * len(PRED_HORIZONS)).clamp(min=1e-9)
                lp = lp_masked.sum() / num_active_forecast_elements

                # 2. Failure Prediction Loss (Lf) - Global for the sample
                lf = focal_loss_mean(fail_logits, fail_tgt)  # Already mean reduced

                # 3. RCA Loss (Lr) - Per sensor
                lr_elements = focal_loss_elementwise(rca_logits, rca_tgt)  # [B, model_max_sensors]
                lr_masked = lr_elements * sensor_m  # Apply sensor mask
                lr = lr_masked.sum() / sensor_m.sum().clamp(min=1e-9)  # Average over active sensors

                # 4. MoE Auxiliary Loss (Laux)
                l_aux = torch.tensor(0.0, device=DEVICE)
                if AUX_LOSS_COEFF > 0 and aux_terms:
                    for task_name, (fi, Pi) in aux_terms.items():
                        l_aux += NUM_EXPERTS_PER_TASK * torch.sum(fi * Pi)  # Encourages load balancing
                    l_aux *= AUX_LOSS_COEFF

                combined_loss = W_PRED * lp + W_FAIL * lf + W_RCA * lr + l_aux

            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(
                    f"Warning: NaN/Inf loss. Lp:{lp.item():.3f}, Lf:{lf.item():.3f}, Lr:{lr.item():.3f}, Aux:{l_aux.item():.3f}. Skipping update.");
                continue

            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)  # Before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # LR scheduler step

            total_loss_epoch += combined_loss.item();
            total_lp_epoch += lp.item();
            total_lf_epoch += lf.item()
            total_lr_epoch += lr.item();
            total_laux_epoch += l_aux.item() if isinstance(l_aux, torch.Tensor) else l_aux
            num_batches += 1

            if batch_idx > 0 and batch_idx % (len(train_loader) // 10 if len(train_loader) >= 10 else 1) == 0:
                lr_val = optimizer.param_groups[0]['lr']
                print(
                    f"E{epoch + 1} B{batch_idx + 1}/{len(train_loader)} LR:{lr_val:.1e} | L:{combined_loss.item():.3f} (P:{lp.item():.3f} F:{lf.item():.3f} R:{lr.item():.3f} Aux:{l_aux.item() if isinstance(l_aux, torch.Tensor) else l_aux:.3f})")

        print_avg_losses(epoch, "Train", num_batches, total_loss_epoch, total_lp_epoch, total_lf_epoch, total_lr_epoch,
                         total_laux_epoch)

        if valid_loader:
            model.eval();
            total_val_loss, total_val_lp, total_val_lf, total_val_lr, total_val_laux = 0, 0, 0, 0, 0
            num_val_batches = 0
            with torch.no_grad():
                for batch in valid_loader:
                    input_feat = batch["input_features"].to(DEVICE);
                    sensor_m = batch["sensor_mask"].to(DEVICE)
                    last_k_std = batch["last_known_values_globally_std"].to(DEVICE);
                    delta_tgt_std = batch["pred_delta_targets_globally_std"].to(DEVICE)
                    fail_tgt = batch["fail_targets"].to(DEVICE);
                    rca_tgt = batch["rca_targets"].to(DEVICE)

                    with autocast(enabled=AMP_ENABLED):
                        pred_abs_std, fail_logits, rca_logits, aux_terms_val = model(input_feat, sensor_m)
                        abs_target_std = last_k_std.unsqueeze(-1) + delta_tgt_std
                        lp_val_el = huber_loss_fn(pred_abs_std, abs_target_std)
                        lp_val = (lp_val_el * sensor_m.unsqueeze(-1)).sum() / (
                                    sensor_m.sum() * len(PRED_HORIZONS)).clamp(min=1e-9)
                        lf_val = focal_loss_mean(fail_logits, fail_tgt)
                        lr_val_el = focal_loss_elementwise(rca_logits, rca_tgt)
                        lr_val = (lr_val_el * sensor_m).sum() / sensor_m.sum().clamp(min=1e-9)
                        l_aux_val = torch.tensor(0.0, device=DEVICE)
                        if AUX_LOSS_COEFF > 0 and aux_terms_val:
                            for _, (fi, Pi) in aux_terms_val.items(): l_aux_val += NUM_EXPERTS_PER_TASK * torch.sum(
                                fi * Pi)
                            l_aux_val *= AUX_LOSS_COEFF
                        val_loss = W_PRED * lp_val + W_FAIL * lf_val + W_RCA * lr_val + l_aux_val

                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        total_val_loss += val_loss.item();
                        total_val_lp += lp_val.item();
                        total_val_lf += lf_val.item()
                        total_val_lr += lr_val.item();
                        total_val_laux += l_aux_val.item() if isinstance(l_aux_val, torch.Tensor) else l_aux_val
                        num_val_batches += 1
            print_avg_losses(epoch, "Valid", num_val_batches, total_val_loss, total_val_lp, total_val_lf, total_val_lr,
                             total_val_laux)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model: {MODEL_SAVE_PATH}, Preprocessor: {PREPROCESSOR_SAVE_PATH}")


def print_avg_losses(epoch, phase, num_batches, total_loss, lp, lf, lr, laux):
    if num_batches > 0:
        print(
            f"E{epoch + 1} Avg {phase} L: {total_loss / num_batches:.3f} (P:{lp / num_batches:.3f} F:{lf / num_batches:.3f} R:{lr / num_batches:.3f} Aux:{laux / num_batches:.3f})")
    else:
        print(f"E{epoch + 1} No batches processed for {phase} phase.")


if __name__ == '__main__':
    print(
        "--- Script Version 5: Multi-Task Foundational Model with MoE, Global Stats, No RevIN, Early Mask, Scheduler, AMP ---")
    print(f"IMPORTANT: TRAIN_DIR ('{TRAIN_DIR}') and VALID_DIR ('{VALID_DIR}') must be set correctly.")
    if BASE_DATA_DIR == "../../data/time_series/1": print(
        "\nWARNING: Using default example BASE_DATA_DIR. Paths might be incorrect.\n")
    train_and_save_model()
