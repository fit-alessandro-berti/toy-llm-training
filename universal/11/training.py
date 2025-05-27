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
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_ema import ExponentialMovingAverage  # Added for EMA

# --- Configuration ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1"
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAINING")
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")

# Model & Task Parameters
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]
FAIL_HORIZONS = [3, 5, 10]  # Still used for multi-horizon evaluation
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0]  # Used for novelty definition in RCA
MAX_SENSORS_CAP = 20

# Architectural Params
SENSOR_INPUT_DIM = 1

# I. Per-sensor Temporal Encoder (TCN based)
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

# II. Inter-sensor Representation
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

# III. Mixture-of-Experts (MoE) - Refactored
NUM_SHARED_EXPERTS = 8
MOE_EXPERT_INPUT_DIM = TRANSFORMER_D_MODEL
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64

MOE_TOP_K = 2
MOE_NOISE_STD = 0.3
AUX_LOSS_COEFF = 0.01  # For MoE load balancing
ENTROPY_REG_COEFF = 0.001

# --- Survival Head Parameters ---
HAZARD_HEAD_MAX_HORIZON = max(FAIL_HORIZONS) if FAIL_HORIZONS else 10
# RCA_SURVIVAL_THRESHOLD = 0.7 # Will be scheduled per epoch

# --- Auxiliary Novelty Pre-text Task ---
SENSOR_DRIFT_THRESHOLD_STD = 2.0  # Number of stds for |Δx| to be considered drift

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 5e-4  # Adjusted for OneCycleLR, acts as max_lr
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2
GRAD_CLIP_MAX_NORM = 1.0
# WARMUP_RATIO = 0.05 # Not directly used by OneCycleLR in this setup, it has its own warmup
GRAD_ACCUMULATION_STEPS = 4  # New parameter
EMA_ALPHA = 0.999  # New parameter for EMA decay

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == 'cuda'

MODEL_SAVE_PATH = "foundation_multitask_model_v6_survival_enhanced.pth"
PREPROCESSOR_SAVE_PATH = "foundation_multitask_preprocessor_v6_survival_enhanced.npz"

# Loss Function Parameters
HUBER_DELTA = 1.0
FOCAL_ALPHA_PARAM = 0.25
FOCAL_GAMMA = 2.0

# Loss weights
W_PRED = 1.0
W_FAIL_BASE = 2.5
W_RCA_BASE = 1.0
W_DRIFT = 0.5  # Weight for the new sensor drift auxiliary loss
ANNEAL_LOSS_WEIGHTS_START_EPOCH = 10
ANNEAL_LOSS_WEIGHTS_FACTOR = 0.9


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

    def forward(self, x): return x + self.pe[:, :x.size(1), :]


# --- TCN Residual Block ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.padding,
                               dilation=dilation)
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=max(seq_len + 1, 5000))

        tcn_blocks = []
        current_channels = proj_dim
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels_block = tcn_out_dim
            tcn_blocks.append(
                TemporalBlock(current_channels, out_channels_block, kernel_size, stride=1, dilation=dilation_size,
                              dropout=dropout)
            )
            current_channels = out_channels_block
        self.tcn_network = nn.Sequential(*tcn_blocks)
        self.final_norm = nn.LayerNorm(tcn_out_dim)
        self.tcn_out_dim = tcn_out_dim

    def forward(self, x):
        x_proj = self.input_proj(x)
        cls_tokens_expanded = self.cls_token.expand(x_proj.size(0), -1, -1)
        x_with_cls = torch.cat((cls_tokens_expanded, x_proj), dim=1)
        x_pos_encoded = self.pos_encoder(x_with_cls)
        x_permuted_for_tcn = x_pos_encoded.permute(0, 2, 1)
        x_tcn_out_with_cls = self.tcn_network(x_permuted_for_tcn)
        x_tcn_out_permuted_back = x_tcn_out_with_cls.permute(0, 2, 1)
        h_cls = x_tcn_out_permuted_back[:, 0]
        main_sequence_features = x_tcn_out_permuted_back[:, 1:]
        main_sequence_normed = self.final_norm(main_sequence_features)
        h_cls_normed = self.final_norm(h_cls.unsqueeze(1)).squeeze(1)
        return main_sequence_normed, h_cls_normed


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


# --- MoE Components ---
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x): return self.fc(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x): return self.fc(x)


# --- Discrete-Time Survival (Hazard) Head ---
class HazardHead(nn.Module):
    def __init__(self, in_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.fc = nn.Linear(in_dim, horizon)

    def forward(self, z):
        return torch.sigmoid(self.fc(z))


# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha_param = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha_param + (1.0 - targets) * (1.0 - self.alpha_param)
        focal_term = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        return focal_term


# --- Survival Loss Function ---
def compute_survival_loss(hazard_rates, event_observed, time_to_event, device, epsilon=1e-8):
    batch_size, horizon_len = hazard_rates.shape
    time_to_event_0idx = (time_to_event - 1).long()

    # Term 1: event_observed_smoothed * log(h_{t_i})
    log_h_ti = torch.log(hazard_rates[torch.arange(batch_size), time_to_event_0idx] + epsilon)
    term1 = event_observed * log_h_ti  # event_observed is now potentially smoothed (e.g. 0.05 for censored)

    # Term 2: (1-event_observed_smoothed) * sum_{t <= t_i} log(1-h_t)
    log_1_minus_h = torch.log(1.0 - hazard_rates + epsilon)
    current_sum_mask = torch.arange(horizon_len, device=device).unsqueeze(0) <= time_to_event_0idx.unsqueeze(1)
    sum_log_1_minus_h_up_to_ti = (log_1_minus_h * current_sum_mask).sum(dim=1)
    term2 = (1.0 - event_observed) * sum_log_1_minus_h_up_to_ti

    # Following the logic from the original prompt:
    # L = - sum_i [ y_i * log h_{t_i} + (1-y_i) * sum_{t<=t_i}log(1-h_t) ]
    # With label smoothing, y_i (event_observed) can be 0.05.
    # So, for a censored event (original y_i=0, now event_observed=0.05):
    # loss_i = - [0.05 * log h_{t_i} + 0.95 * sum_{t<=t_i}log(1-h_t)]
    # For an observed event (original y_i=1, event_observed=1.0):
    # loss_i = - [1.0 * log h_{t_i} + 0.0 * sum_{t<=t_i}log(1-h_t)] = -log h_{t_i}
    # The implementation needs to be careful to match this interpretation.
    # The existing structure where event_observed directly multiplies log_h_ti and (1-event_observed) multiplies the sum of log(1-h)
    # naturally handles the smoothing.

    log_likelihood = term1 + term2  # This sums the two components for each sample.
    # For y_i=1 (observed), term2 becomes 0. For y_i=0 (censored, raw), term1 becomes 0.
    # With smoothing (e.g., y_i=0.05 for censored), both terms contribute.

    return -log_likelihood.mean()


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
            df_peek = pd.read_csv(fp, nrows=1);
            sensor_cols = get_sensor_columns(df_peek)
            if sensor_cols: max_s = max(max_s, len(sensor_cols)); processed_files += 1
        except Exception:
            continue
    if processed_files == 0: return 0
    return min(max_s, cap) if cap > 0 and max_s > cap else max_s


def calculate_global_stats(file_paths, target_num_sensors):
    print(f"Calculating global statistics for up to {target_num_sensors} sensors...")
    sums = np.zeros(target_num_sensors, dtype=np.float64)
    sum_sqs = np.zeros(target_num_sensors, dtype=np.float64)
    counts = np.zeros(target_num_sensors, dtype=np.int64)
    canonical_sensor_names = None
    for fp_idx, fp in enumerate(file_paths):
        try:
            df = pd.read_csv(fp, low_memory=False);
            current_file_sensor_cols = get_sensor_columns(df)
            if not current_file_sensor_cols: continue
            if canonical_sensor_names is None:
                canonical_sensor_names = current_file_sensor_cols[:target_num_sensors]
                if len(canonical_sensor_names) < target_num_sensors:
                    target_num_sensors = len(canonical_sensor_names)
                    sums = sums[:target_num_sensors];
                    sum_sqs = sum_sqs[:target_num_sensors];
                    counts = counts[:target_num_sensors]
                if target_num_sensors == 0: print(
                    "Error: No usable sensor columns from first file."); return None, None, []
            for i, name in enumerate(canonical_sensor_names):
                if name in df.columns:
                    col_data = df[name].values.astype(np.float32);
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0: sums[i] += valid_data.sum(); sum_sqs[i] += (valid_data ** 2).sum(); counts[
                        i] += len(valid_data)
        except Exception as e:
            print(f"Warning: Skipping file {fp} during stats calc: {e}");
            continue
        if (fp_idx + 1) % 20 == 0: print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for stats...")
    if canonical_sensor_names is None or target_num_sensors == 0 or np.sum(counts) == 0: return None, None, []
    valid_indices = counts > 0
    final_canonical_names = [name for i, name in enumerate(canonical_sensor_names) if valid_indices[i]]
    final_means = sums[valid_indices] / counts[valid_indices]
    final_stds = np.sqrt(sum_sqs[valid_indices] / counts[valid_indices] - final_means ** 2)
    final_stds[final_stds < 1e-8] = 1e-8
    if len(final_canonical_names) == 0: return None, None, []
    print(f"Global stats calculated for {len(final_canonical_names)} sensors.")
    return final_means, final_stds, final_canonical_names


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizons, eval_fail_horizons, hazard_max_horizon,
                 rca_failure_lookahead, model_max_sensors_dim,
                 global_means, global_stds, canonical_sensor_names, sensor_drift_std_thresh):
        self.data_dir = data_dir;
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons;
        self.eval_fail_horizons = eval_fail_horizons
        self.hazard_max_horizon = hazard_max_horizon
        self.rca_failure_lookahead = rca_failure_lookahead
        self.model_max_sensors_dim = model_max_sensors_dim
        self.global_means = global_means;
        self.global_stds = global_stds
        self.canonical_sensor_names = canonical_sensor_names
        self.num_globally_normed_features = len(canonical_sensor_names)
        self.sensor_drift_std_thresh = sensor_drift_std_thresh  # New
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
                print(f"Warning: Skipping file {fp}: {e}");
                continue
            raw_features_from_canonical_cols = np.full((len(df), self.num_globally_normed_features), np.nan,
                                                       dtype=np.float32)
            for i, name in enumerate(self.canonical_sensor_names):
                if name in df.columns: raw_features_from_canonical_cols[:, i] = df[name].values.astype(np.float32)
            if np.all(np.isnan(raw_features_from_canonical_cols)): continue
            features_normalized_globally = np.full_like(raw_features_from_canonical_cols, np.nan)
            for i in range(self.num_globally_normed_features):
                valid_mask = ~np.isnan(raw_features_from_canonical_cols[:, i])
                features_normalized_globally[valid_mask, i] = (raw_features_from_canonical_cols[valid_mask, i] -
                                                               self.global_means[i]) / self.global_stds[i]
            failure_flags = df["CURRENT_FAILURE"].values.astype(
                np.int64) if "CURRENT_FAILURE" in df.columns else np.zeros(len(df), dtype=np.int64)
            self.data_cache.append({"raw_features_globally_aligned": raw_features_from_canonical_cols,
                                    "features_normalized_globally": features_normalized_globally,
                                    "failure_flags": failure_flags})
            max_lookahead = max(max(self.pred_horizons) if self.pred_horizons else 0,
                                max(self.eval_fail_horizons) if self.eval_fail_horizons else 0,
                                self.hazard_max_horizon,
                                self.rca_failure_lookahead)
            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        if not self.data_cache: print(f"CRITICAL WARNING: No data loaded from {self.data_dir}."); return
        print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item_data = self.data_cache[file_idx]
        raw_features_aligned = item_data["raw_features_globally_aligned"]
        features_normalized_aligned = item_data["features_normalized_globally"]
        flags_full = item_data["failure_flags"]
        input_slice_normed_full_potential = features_normalized_aligned[
                                            window_start_idx: window_start_idx + self.seq_len]

        padded_input_normed = np.zeros((self.seq_len, self.model_max_sensors_dim), dtype=np.float32)
        sensor_mask = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        num_to_copy = min(self.num_globally_normed_features, self.model_max_sensors_dim)

        current_input_slice_normed = input_slice_normed_full_potential[:, :num_to_copy]
        padded_input_normed[:, :num_to_copy] = current_input_slice_normed

        for k_idx in range(num_to_copy):
            if not np.all(np.isnan(current_input_slice_normed[:, k_idx])): sensor_mask[k_idx] = 1.0
        padded_input_normed[np.isnan(padded_input_normed)] = 0.0  # Impute NaNs after alignment

        last_known_normed = padded_input_normed[-1, :].copy()
        delta_targets_normed = np.zeros((self.model_max_sensors_dim, len(self.pred_horizons)), dtype=np.float32)
        for i_h, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < features_normalized_aligned.shape[0]:
                target_values_all_normed = features_normalized_aligned[target_idx, :]
                for k_idx in range(num_to_copy):
                    if sensor_mask[k_idx] > 0 and not np.isnan(target_values_all_normed[k_idx]):
                        delta_targets_normed[k_idx, i_h] = target_values_all_normed[k_idx] - last_known_normed[k_idx]

        eval_fail_targets = np.zeros(len(self.eval_fail_horizons), dtype=np.float32)
        for i_fh, fh in enumerate(self.eval_fail_horizons):
            start, end = window_start_idx + self.seq_len, window_start_idx + self.seq_len + fh
            if end <= len(flags_full) and np.any(flags_full[start:end]): eval_fail_targets[i_fh] = 1.0

        # --- Survival Label Modification ---
        time_to_event = self.hazard_max_horizon
        is_event_observed_hard = 0  # Original hard label
        failure_window_start = window_start_idx + self.seq_len
        failure_window_end = failure_window_start + self.hazard_max_horizon
        if failure_window_end <= len(flags_full):
            actual_failure_flags_in_window = flags_full[failure_window_start:failure_window_end]
            if np.any(actual_failure_flags_in_window):
                is_event_observed_hard = 1
                time_to_event = np.argmax(actual_failure_flags_in_window) + 1

        # Apply label smoothing for censored samples
        event_observed_smoothed = 1.0 if is_event_observed_hard == 1 else 0.05

        rca_targets = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        start_r, end_r = window_start_idx + self.seq_len, window_start_idx + self.seq_len + self.rca_failure_lookahead
        if end_r <= len(flags_full) and np.any(flags_full[start_r:end_r]):
            current_window_raw_slice = raw_features_aligned[window_start_idx: window_start_idx + self.seq_len,
                                       :num_to_copy]
            future_lookahead_raw_slice = raw_features_aligned[start_r:end_r, :num_to_copy]
            for k_idx in range(num_to_copy):
                if sensor_mask[k_idx] > 0:
                    sensor_data_current_window_raw = current_window_raw_slice[:, k_idx]
                    sensor_data_future_lookahead_raw = future_lookahead_raw_slice[:, k_idx]
                    valid_current_raw = sensor_data_current_window_raw[~np.isnan(sensor_data_current_window_raw)]
                    valid_future_raw = sensor_data_future_lookahead_raw[~np.isnan(sensor_data_future_lookahead_raw)]
                    if len(valid_current_raw) > 0 and len(valid_future_raw) > 0:
                        mean_current_raw = np.mean(valid_current_raw);
                        std_current_raw = np.std(valid_current_raw);
                        std_current_raw = max(std_current_raw, 1e-6)
                        if np.any(np.abs(valid_future_raw - mean_current_raw) > 3 * std_current_raw): rca_targets[
                            k_idx] = 1.0

        # --- Sensor Drift Labels ---
        sensor_drift_labels = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        # Use the normalized input window (padded_input_normed already has relevant part)
        input_slice_for_drift_calc = padded_input_normed[:, :num_to_copy]
        for k_idx in range(num_to_copy):
            if sensor_mask[k_idx] > 0:
                sensor_trace_in_window = input_slice_for_drift_calc[:, k_idx]
                if len(sensor_trace_in_window) < 2: continue  # Need at least 2 points for a difference

                sigma_sensor_window = np.std(sensor_trace_in_window)
                if sigma_sensor_window < 1e-6: sigma_sensor_window = 1e-6

                deltas_in_window = np.diff(sensor_trace_in_window)
                if np.any(np.abs(deltas_in_window) > self.sensor_drift_std_thresh * sigma_sensor_window):
                    sensor_drift_labels[k_idx] = 1.0

        return {"input_features": torch.from_numpy(padded_input_normed), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_globally_std": torch.from_numpy(last_known_normed),
                "pred_delta_targets_globally_std": torch.from_numpy(delta_targets_normed),
                "eval_fail_targets": torch.from_numpy(eval_fail_targets),
                "survival_targets": {"event_observed": torch.tensor(event_observed_smoothed, dtype=torch.float),
                                     "time_to_event": torch.tensor(time_to_event, dtype=torch.long)},
                "rca_targets": torch.from_numpy(rca_targets),
                "sensor_drift_labels": torch.from_numpy(sensor_drift_labels)}  # New output


# --- Foundational Multi-Task Model ---
class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, model_max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 num_shared_experts, moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim,
                 pred_horizons_len, hazard_max_horizon,
                 moe_top_k, moe_noise_std, aux_loss_coeff, entropy_reg_coeff):
        super().__init__()
        self.model_max_sensors = model_max_sensors;
        self.seq_len = seq_len
        self.num_shared_experts = num_shared_experts;
        self.moe_output_dim = moe_output_dim
        self.transformer_d_model = transformer_d_model;
        self.sensor_tcn_out_dim = sensor_tcn_out_dim
        self.moe_top_k = moe_top_k;
        self.moe_noise_std = moe_noise_std
        self.aux_loss_coeff = aux_loss_coeff;  # For MoE load balancing
        self.entropy_reg_coeff = entropy_reg_coeff
        self.hazard_max_horizon = hazard_max_horizon

        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)
        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_tcn_out_dim,
                                                        transformer_d_model) if sensor_tcn_out_dim != transformer_d_model else nn.Identity()
        self.inter_sensor_transformer = InterSensorTransformer(transformer_d_model, transformer_nhead,
                                                               transformer_nlayers, model_max_sensors)
        self.experts_shared = nn.ModuleList(
            [Expert(moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_shared_experts)])
        gate_input_dim_global_forecast_fail = transformer_d_model * 2 + sensor_tcn_out_dim
        gate_input_dim_rca_token = transformer_d_model
        self.gates = nn.ModuleDict({
            "forecast": GatingNetwork(gate_input_dim_global_forecast_fail, num_shared_experts),
            "fail": GatingNetwork(gate_input_dim_global_forecast_fail, num_shared_experts),
            "rca": GatingNetwork(gate_input_dim_rca_token, num_shared_experts)
        })
        self.pred_head = nn.Linear(sensor_tcn_out_dim + transformer_d_model + moe_output_dim, pred_horizons_len)
        self.hazard_head = HazardHead(moe_output_dim, self.hazard_max_horizon)
        rca_head_input_dim = sensor_tcn_out_dim + transformer_d_model + moe_output_dim + 1
        self.rca_head = nn.Linear(rca_head_input_dim, 1)

        # New Sensor Drift Head
        self.sensor_drift_head = nn.Linear(sensor_tcn_out_dim + transformer_d_model, 1)

    def _apply_moe_topk(self, x_expert_input, gate_input, gate_network, experts_modulelist, k, noise_std):
        logits = gate_network(gate_input)
        if self.training and noise_std > 0: logits = logits + torch.randn_like(logits) * noise_std
        num_experts = len(experts_modulelist);
        eff_k = min(k, num_experts)
        topk_val, topk_idx = torch.topk(logits, eff_k, dim=-1);
        topk_w = torch.softmax(topk_val, dim=-1)

        # Optimized expert processing for valid inputs only if x_expert_input can be sparse (e.g. due to masking)
        # However, typical MoE expert_input is dense batch.
        all_out_list = []
        for i in range(x_expert_input.shape[0]):  # Iterate over batch or tokens
            current_x = x_expert_input[i:i + 1]  # Keep dim
            current_topk_idx = topk_idx[i]  # Indices for this item

            # Only compute for selected experts for this item
            # This is more complex than just running all experts then gathering.
            # The original code runs all experts on the full batch/token list then gathers. This is simpler.

        all_out = torch.stack([e(x_expert_input) for e in experts_modulelist],
                              dim=1)  # [Batch_or_Tokens, NumExperts, ExpertOutDim]

        expert_output_dim = all_out.size(-1)

        # Correct gather_idx expansion based on x_expert_input leading dimensions
        # If x_expert_input is [B, D], gather_idx from topk_idx [B, K] needs to be [B, K, ExpertOutDim]
        # If x_expert_input is [N_Tokens, D], gather_idx from topk_idx [N_Tokens, K] needs to be [N_Tokens, K, ExpertOutDim]
        gather_idx_shape = list(topk_idx.shape) + [expert_output_dim]
        gather_idx = topk_idx.unsqueeze(-1).expand(gather_idx_shape)

        sel_out = all_out.gather(1, gather_idx);  # [Batch_or_Tokens, K, ExpertOutDim]
        y = (sel_out * topk_w.unsqueeze(-1)).sum(dim=1)  # [Batch_or_Tokens, ExpertOutDim]

        router_prob_for_loss = torch.softmax(logits, -1);
        avg_router_prob = router_prob_for_loss.mean(0)  # Avg prob per expert over batch/tokens

        # Load balance calculation
        # expert_frac: fraction of tokens/samples routed to each expert
        ones_for_scatter = torch.ones_like(topk_idx, dtype=router_prob_for_loss.dtype).reshape(-1)  # Flattened
        num_items_processed = x_expert_input.size(
            0) if x_expert_input.ndim > 1 else 1  # num_items = batch_size or num_tokens

        expert_frac = torch.zeros_like(avg_router_prob).scatter_add_(
            0, topk_idx.reshape(-1), ones_for_scatter  # Flattened indices
        ) / (num_items_processed * eff_k if num_items_processed > 0 else 1.0)

        load_balance_loss = self.num_shared_experts * (avg_router_prob * expert_frac).sum()
        return y, load_balance_loss, logits

    def forward(self, x_features_globally_std, sensor_mask, last_known_values_globally_std_for_novelty):
        batch_size, seq_len_data, _ = x_features_globally_std.shape
        x_input_for_tcn = x_features_globally_std.reshape(batch_size * self.model_max_sensors, seq_len_data,
                                                          SENSOR_INPUT_DIM)
        sensor_temporal_features_flat, h_cls_flat = self.per_sensor_encoder(x_input_for_tcn)
        sensor_temporal_features_main = sensor_temporal_features_flat.reshape(
            batch_size, self.model_max_sensors, seq_len_data, self.sensor_tcn_out_dim
        )
        sensor_temporal_features_main = sensor_temporal_features_main * sensor_mask.view(batch_size,
                                                                                         self.model_max_sensors, 1, 1)
        pooled_sensor_features = torch.mean(sensor_temporal_features_main, dim=2)
        projected_for_inter_sensor = self.pooled_to_transformer_dim_proj(pooled_sensor_features)
        transformer_padding_mask = (sensor_mask == 0)
        cross_sensor_context = self.inter_sensor_transformer(projected_for_inter_sensor, transformer_padding_mask)
        cross_sensor_context_masked = cross_sensor_context * sensor_mask.unsqueeze(-1)

        active_sensors_per_batch = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_ctx_global = (cross_sensor_context_masked).sum(dim=1) / active_sensors_per_batch
        mean_sq_ctx_global = ((cross_sensor_context_masked ** 2) * sensor_mask.unsqueeze(-1)).sum(
            dim=1) / active_sensors_per_batch
        var_ctx_global = mean_sq_ctx_global - mean_ctx_global ** 2
        std_ctx_global = torch.sqrt(var_ctx_global.clamp(min=1e-6))

        h_cls_per_sensor = h_cls_flat.reshape(batch_size, self.model_max_sensors, self.sensor_tcn_out_dim)
        h_cls_per_sensor_masked = h_cls_per_sensor * sensor_mask.unsqueeze(-1)
        h_cls_global_avg = h_cls_per_sensor_masked.sum(dim=1) / active_sensors_per_batch

        router_input_global_forecast_fail = torch.cat([mean_ctx_global, std_ctx_global, h_cls_global_avg], dim=-1)

        moe_forecast_output, aux_f, logits_f = self._apply_moe_topk(
            mean_ctx_global, router_input_global_forecast_fail, self.gates["forecast"], self.experts_shared,
            self.moe_top_k, self.moe_noise_std
        )
        moe_fail_output, aux_fail, logits_fail = self._apply_moe_topk(
            mean_ctx_global, router_input_global_forecast_fail, self.gates["fail"], self.experts_shared,
            self.moe_top_k, self.moe_noise_std
        )

        x_flat_rca_expert_input = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        valid_token_mask_rca = sensor_mask.view(-1).bool()
        x_flat_rca_expert_input_valid = x_flat_rca_expert_input[valid_token_mask_rca]

        moe_rca_output_flat_valid = torch.empty(0, self.moe_output_dim, device=x_features_globally_std.device,
                                                dtype=moe_forecast_output.dtype)
        aux_rca = torch.tensor(0.0, device=x_features_globally_std.device);
        logits_rca_valid = None
        if x_flat_rca_expert_input_valid.size(0) > 0:
            x_flat_rca_gate_input_valid = x_flat_rca_expert_input_valid  # Gating based on the token itself for RCA MoE
            moe_rca_output_flat_valid, aux_rca, logits_rca_valid = self._apply_moe_topk(
                x_flat_rca_expert_input_valid, x_flat_rca_gate_input_valid, self.gates["rca"], self.experts_shared,
                self.moe_top_k, self.moe_noise_std
            )

        moe_rca_output_flat = torch.zeros(batch_size * self.model_max_sensors, self.moe_output_dim,
                                          device=x_features_globally_std.device, dtype=moe_forecast_output.dtype)
        if x_flat_rca_expert_input_valid.size(0) > 0: moe_rca_output_flat[
            valid_token_mask_rca] = moe_rca_output_flat_valid

        # MoE auxiliary loss (load balancing)
        total_moe_aux_loss = self.aux_loss_coeff * (aux_f + aux_fail + aux_rca)

        total_entropy_loss = torch.tensor(0.0, device=x_features_globally_std.device)
        if self.entropy_reg_coeff > 0:
            for gate_logits_set in [logits_f, logits_fail, logits_rca_valid]:
                if gate_logits_set is not None and gate_logits_set.numel() > 0:
                    probs = F.softmax(gate_logits_set, dim=-1);
                    log_probs = F.log_softmax(gate_logits_set, dim=-1)
                    total_entropy_loss -= self.entropy_reg_coeff * (probs * log_probs).sum(dim=-1).mean()

        tcn_features_last_step = sensor_temporal_features_main[:, :, -1, :]
        moe_f_expanded = moe_forecast_output.unsqueeze(1).expand(-1, self.model_max_sensors, -1)
        pred_head_input_features = torch.cat([tcn_features_last_step, cross_sensor_context_masked, moe_f_expanded],
                                             dim=-1)
        pred_delta_globally_std = self.pred_head(pred_head_input_features)

        last_val_observed_for_pred = x_features_globally_std[:, -1, :]
        pred_abs_globally_std = last_val_observed_for_pred.unsqueeze(-1) + pred_delta_globally_std
        pred_abs_globally_std = pred_abs_globally_std * sensor_mask.unsqueeze(-1)

        hazard_rates = self.hazard_head(moe_fail_output)
        log_1_minus_h = torch.log(1.0 - hazard_rates + 1e-8)
        cumulative_log_survival = torch.cumsum(log_1_minus_h, dim=1)
        cumulative_survival_probs = torch.exp(cumulative_log_survival)

        with torch.no_grad():  # As per original, delta_novelty is for RCA head input, not trained directly
            pred_next_h1 = pred_abs_globally_std[:, :, 0]  # Prediction for horizon 1
        delta_novelty = torch.abs(
            pred_next_h1 - last_known_values_globally_std_for_novelty)  # Difference from actual last known from input
        delta_novelty_masked = delta_novelty * sensor_mask;
        delta_novelty_flat = delta_novelty_masked.reshape(-1, 1)

        tcn_flat = tcn_features_last_step.reshape(-1, self.sensor_tcn_out_dim)
        ctx_flat = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        rca_head_input_flat = torch.cat([tcn_flat, ctx_flat, moe_rca_output_flat, delta_novelty_flat], dim=-1)
        rca_logits_flat = self.rca_head(rca_head_input_flat).squeeze(-1)
        rca_logits = rca_logits_flat.view(batch_size, self.model_max_sensors)

        # Sensor Drift Prediction
        # Input: h_cls_per_sensor [B, M, D_tcn_out], cross_sensor_context_masked [B, M, D_transformer]
        sensor_drift_head_input = torch.cat([h_cls_per_sensor_masked, cross_sensor_context_masked], dim=-1)
        sensor_drift_logits = self.sensor_drift_head(sensor_drift_head_input).squeeze(-1)  # Output [B, M]

        return (pred_abs_globally_std, hazard_rates, cumulative_survival_probs, rca_logits,
                total_moe_aux_loss, total_entropy_loss, sensor_drift_logits)  # Added sensor_drift_logits


# --- Training Function ---
def train_and_save_model():
    print(f"Device: {DEVICE}, AMP Enabled: {AMP_ENABLED}")
    print(f"TRAIN_DIR: {TRAIN_DIR}, VALID_DIR: {VALID_DIR}")
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
             model_max_sensors_dim=model_max_sensors_dim, seq_len=SEQ_LEN, pred_horizons=np.array(PRED_HORIZONS),
             fail_horizons=np.array(FAIL_HORIZONS), rca_failure_lookahead=RCA_FAILURE_LOOKAHEAD,
             hazard_max_horizon=HAZARD_HEAD_MAX_HORIZON, sensor_drift_std_thresh=SENSOR_DRIFT_THRESHOLD_STD)
    print(f"Preprocessor config saved to {PREPROCESSOR_SAVE_PATH}")

    model = FoundationalTimeSeriesModel(
        model_max_sensors=model_max_sensors_dim, seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        num_shared_experts=NUM_SHARED_EXPERTS, moe_expert_input_dim=MOE_EXPERT_INPUT_DIM,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        pred_horizons_len=len(PRED_HORIZONS), hazard_max_horizon=HAZARD_HEAD_MAX_HORIZON,
        moe_top_k=MOE_TOP_K, moe_noise_std=MOE_NOISE_STD,
        aux_loss_coeff=AUX_LOSS_COEFF, entropy_reg_coeff=ENTROPY_REG_COEFF
    ).to(DEVICE)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  HAZARD_HEAD_MAX_HORIZON,
                                                  RCA_FAILURE_LOOKAHEAD, model_max_sensors_dim, global_means,
                                                  global_stds, canonical_sensor_names,
                                                  SENSOR_DRIFT_THRESHOLD_STD)  # Pass new param
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  HAZARD_HEAD_MAX_HORIZON,
                                                  RCA_FAILURE_LOOKAHEAD, model_max_sensors_dim, global_means,
                                                  global_stds, canonical_sensor_names,
                                                  SENSOR_DRIFT_THRESHOLD_STD)  # Pass new param
    if len(train_dataset) == 0: print("ERROR: Training dataset empty."); return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=AMP_ENABLED)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=AMP_ENABLED) if len(valid_dataset) > 0 else None

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)

    # OneCycleLR Scheduler
    total_mini_batches = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_mini_batches,
        pct_start=0.3,  # Corresponds to "first 30% of training" for ramp up
        anneal_strategy='cos'  # Default is cosine
    )

    scaler = GradScaler(enabled=AMP_ENABLED)
    ema = ExponentialMovingAverage(model.parameters(), decay=EMA_ALPHA) if EMA_ALPHA > 0 else None

    huber_loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')
    rca_focal_loss_elementwise = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA, reduction='none')
    sensor_drift_focal_loss_elementwise = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA,
                                                    reduction='none')  # For new drift task

    # RCA Survival Threshold Schedule
    tau_schedule = np.linspace(0.95, 0.70, EPOCHS)

    print(
        f"Starting multi-task training. Max Hazard Horizon: {HAZARD_HEAD_MAX_HORIZON}, Grad Accum: {GRAD_ACCUMULATION_STEPS}, EMA: {EMA_ALPHA > 0}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss_epoch, total_lp_epoch, total_lf_epoch, total_lr_epoch, total_l_drift_epoch, total_lmoe_aux_epoch, total_lentr_epoch = 0, 0, 0, 0, 0, 0, 0
        num_optimizer_steps_epoch = 0

        current_rca_survival_threshold = tau_schedule[epoch]
        print(f"Epoch {epoch + 1}/{EPOCHS} - RCA Survival Threshold (tau): {current_rca_survival_threshold:.4f}")

        current_w_fail = W_FAIL_BASE;
        current_w_rca = W_RCA_BASE
        current_w_drift = W_DRIFT  # Added for drift loss
        if epoch >= ANNEAL_LOSS_WEIGHTS_START_EPOCH:
            decay_exponent = epoch - ANNEAL_LOSS_WEIGHTS_START_EPOCH + 1
            decay_multiplier = ANNEAL_LOSS_WEIGHTS_FACTOR ** decay_exponent
            current_w_fail *= decay_multiplier;
            current_w_rca *= decay_multiplier
            current_w_drift *= decay_multiplier  # Anneal drift weight too
            print(
                f"  Annealing W_FAIL to {current_w_fail:.4f}, W_RCA to {current_w_rca:.4f}, W_DRIFT to {current_w_drift:.4f}")

        optimizer.zero_grad()  # Zero grad at the beginning of accumulation cycle

        for batch_idx, batch in enumerate(train_loader):
            input_feat = batch["input_features"].to(DEVICE);
            sensor_m = batch["sensor_mask"].to(DEVICE)
            last_k_std = batch["last_known_values_globally_std"].to(DEVICE)
            delta_tgt_std = batch["pred_delta_targets_globally_std"].to(DEVICE)
            survival_tgt = batch["survival_targets"]
            event_observed = survival_tgt["event_observed"].to(DEVICE)  # Now smoothed
            time_to_event = survival_tgt["time_to_event"].to(DEVICE)
            rca_tgt = batch["rca_targets"].to(DEVICE)
            sensor_drift_tgt = batch["sensor_drift_labels"].to(DEVICE)  # New target

            with autocast(enabled=AMP_ENABLED):
                pred_abs_std, hazard_rates, cumulative_survival_probs, rca_logits, l_moe_aux, l_entropy, sensor_drift_logits = model(
                    input_feat, sensor_m, last_k_std
                )

                abs_target_std = last_k_std.unsqueeze(-1) + delta_tgt_std
                lp_elements = huber_loss_fn(pred_abs_std, abs_target_std)
                lp_masked = lp_elements * sensor_m.unsqueeze(-1)
                num_active_forecast_elements = (sensor_m.sum() * len(PRED_HORIZONS)).clamp(min=1e-9)
                lp = lp_masked.sum() / num_active_forecast_elements

                lf = compute_survival_loss(hazard_rates, event_observed, time_to_event, DEVICE)

                S_H = cumulative_survival_probs[:, -1]
                rca_active_sample_mask = (S_H < current_rca_survival_threshold).float()
                lr_elements = rca_focal_loss_elementwise(rca_logits, rca_tgt)
                lr_masked_elements_gated = (lr_elements * sensor_m) * rca_active_sample_mask.unsqueeze(1)
                num_active_rca_elements = (sensor_m * rca_active_sample_mask.unsqueeze(1)).sum().clamp(min=1e-9)
                lr = lr_masked_elements_gated.sum() / num_active_rca_elements
                if num_active_rca_elements < 1e-8: lr = torch.tensor(0.0, device=DEVICE)

                # Sensor Drift Loss (l_drift)
                l_drift_elements = sensor_drift_focal_loss_elementwise(sensor_drift_logits, sensor_drift_tgt)
                l_drift_masked = l_drift_elements * sensor_m  # Mask inactive sensors
                num_active_drift_elements = sensor_m.sum().clamp(min=1e-9)
                l_drift = l_drift_masked.sum() / num_active_drift_elements
                if num_active_drift_elements < 1e-8: l_drift = torch.tensor(0.0, device=DEVICE)

                # Combined total auxiliary loss for logging (MoE + Drift)
                # Note: l_moe_aux already has aux_loss_coeff applied in model. l_drift will use W_DRIFT.

                combined_loss = (W_PRED * lp +
                                 current_w_fail * lf +
                                 current_w_rca * lr +
                                 current_w_drift * l_drift +  # Weighted drift loss
                                 l_moe_aux +  # MoE load balancing loss
                                 l_entropy)

                # Scale loss for gradient accumulation
                loss_to_backward = combined_loss / GRAD_ACCUMULATION_STEPS

            if torch.isnan(loss_to_backward) or torch.isinf(loss_to_backward):
                print(
                    f"Warning: NaN/Inf loss. Lp:{lp.item():.3f}, Lf:{lf.item():.3f}, Lr:{lr.item():.3f}, LDrift:{l_drift.item():.3f}, LMoeAux:{l_moe_aux.item():.3f}, LEntr:{l_entropy.item():.3f}. Skipping accumulation.")
                # Don't skip optimizer step if this is the Nth batch, just skip this batch's contribution.
                # However, if loss is NaN, backward() will fail. So, best to skip the backward for this batch.
                if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    # If it's an update step, and current batch is NaN, we might have accumulated valid grads before.
                    # This case is tricky. Simplest is to skip update if current loss is NaN.
                    # Or, zero_grad before next accumulation if this one was skipped.
                    # For now, if loss_to_backward is NaN, we skip backward.
                    # If an optimizer step was due, it will proceed with potentially stale or no new grads.
                    pass  # Let the accumulation logic handle the optimizer step
                else:  # Not an optimizer step, just skip this batch's backward
                    continue
            else:
                scaler.scale(loss_to_backward).backward()

            if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                if not (torch.isnan(loss_to_backward) or torch.isinf(
                        loss_to_backward)):  # Only step if current grads are valid
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    if ema: ema.update()
                optimizer.zero_grad(set_to_none=True)  # Zero grad for next accumulation cycle
                num_optimizer_steps_epoch += 1

            # OneCycleLR steps every mini-batch
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            total_loss_epoch += combined_loss.item()  # Log unscaled combined loss
            total_lp_epoch += lp.item();
            total_lf_epoch += lf.item();
            total_lr_epoch += lr.item()
            total_l_drift_epoch += l_drift.item();
            total_lmoe_aux_epoch += l_moe_aux.item();
            total_lentr_epoch += l_entropy.item()

            if batch_idx > 0 and batch_idx % (len(train_loader) // 10 if len(train_loader) >= 10 else 1) == 0:
                lr_val = optimizer.param_groups[0]['lr']
                print(
                    f"E{epoch + 1} B{batch_idx + 1}/{len(train_loader)} LR:{lr_val:.1e} | AccumL:{loss_to_backward.item():.3f} CombL:{combined_loss.item():.3f} (P:{lp.item():.3f} F:{lf.item():.3f} R:{lr.item():.3f} Dr:{l_drift.item():.3f} MoeA:{l_moe_aux.item():.3f} Entr:{l_entropy.item():.3f})")

        avg_num_batches_for_log = len(train_loader)
        print_avg_losses(epoch, "Train", avg_num_batches_for_log, total_loss_epoch, total_lp_epoch, total_lf_epoch,
                         total_lr_epoch, total_l_drift_epoch, total_lmoe_aux_epoch, total_lentr_epoch)

        if valid_loader:
            model_to_eval = model
            if ema:
                ema.store()  # Store original parameters
                ema.copy_to()  # Copy EMA parameters to model for validation
                model_to_eval = model  # EMA weights are now in `model`

            model_to_eval.eval()
            total_val_loss, total_val_lp, total_val_lf, total_val_lr, total_val_ldr, total_val_lmoe_aux, total_val_lentr = 0, 0, 0, 0, 0, 0, 0
            num_val_batches = 0
            all_fail_preds_eval = []
            all_fail_targets_eval = []

            with torch.no_grad():
                for batch_val in valid_loader:
                    input_feat_val = batch_val["input_features"].to(DEVICE);
                    sensor_m_val = batch_val["sensor_mask"].to(DEVICE)
                    last_k_std_val = batch_val["last_known_values_globally_std"].to(DEVICE)
                    delta_tgt_std_val = batch_val["pred_delta_targets_globally_std"].to(DEVICE)
                    survival_tgt_val = batch_val["survival_targets"]
                    event_observed_val = survival_tgt_val["event_observed"].to(DEVICE)
                    time_to_event_val = survival_tgt_val["time_to_event"].to(DEVICE)
                    eval_fail_tgt_val = batch_val["eval_fail_targets"].to(DEVICE)
                    rca_tgt_val = batch_val["rca_targets"].to(DEVICE)
                    sensor_drift_tgt_val = batch_val["sensor_drift_labels"].to(DEVICE)

                    # Mixed-precision validation
                    with autocast(enabled=AMP_ENABLED):
                        pred_abs_std_val, hazard_rates_val, cumulative_survival_probs_val, rca_logits_val, l_moe_aux_val, l_entr_val, sensor_drift_logits_val = model_to_eval(
                            input_feat_val, sensor_m_val, last_k_std_val
                        )
                        abs_target_std_val = last_k_std_val.unsqueeze(-1) + delta_tgt_std_val
                        lp_val_el = huber_loss_fn(pred_abs_std_val, abs_target_std_val)
                        lp_val = (lp_val_el * sensor_m_val.unsqueeze(-1)).sum() / (
                                    sensor_m_val.sum() * len(PRED_HORIZONS)).clamp(min=1e-9)
                        lf_val = compute_survival_loss(hazard_rates_val, event_observed_val, time_to_event_val, DEVICE)

                        S_H_val = cumulative_survival_probs_val[:, -1]
                        rca_active_sample_mask_val = (
                                    S_H_val < current_rca_survival_threshold).float()  # Use current epoch's threshold for consistency
                        lr_val_el = rca_focal_loss_elementwise(rca_logits_val, rca_tgt_val)
                        lr_val_masked_el_gated = (lr_val_el * sensor_m_val) * rca_active_sample_mask_val.unsqueeze(1)
                        num_active_rca_elements_val = (
                                    sensor_m_val * rca_active_sample_mask_val.unsqueeze(1)).sum().clamp(min=1e-9)
                        lr_val = lr_val_masked_el_gated.sum() / num_active_rca_elements_val
                        if num_active_rca_elements_val < 1e-8: lr_val = torch.tensor(0.0, device=DEVICE)

                        l_drift_val_el = sensor_drift_focal_loss_elementwise(sensor_drift_logits_val,
                                                                             sensor_drift_tgt_val)
                        l_drift_val_masked = l_drift_val_el * sensor_m_val
                        num_active_drift_elements_val = sensor_m_val.sum().clamp(min=1e-9)
                        l_drift_val = l_drift_val_masked.sum() / num_active_drift_elements_val
                        if num_active_drift_elements_val < 1e-8: l_drift_val = torch.tensor(0.0, device=DEVICE)

                        val_loss = (W_PRED * lp_val + current_w_fail * lf_val + current_w_rca * lr_val +
                                    current_w_drift * l_drift_val + l_moe_aux_val + l_entr_val)

                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        total_val_loss += val_loss.item();
                        total_val_lp += lp_val.item();
                        total_val_lf += lf_val.item()
                        total_val_lr += lr_val.item();
                        total_val_ldr += l_drift_val.item();
                        total_val_lmoe_aux += l_moe_aux_val.item();
                        total_val_lentr += l_entr_val.item()
                        num_val_batches += 1

                        probs_fail_at_eval_horizons = []
                        for fh_idx, fh_val in enumerate(FAIL_HORIZONS):
                            if fh_val <= HAZARD_HEAD_MAX_HORIZON:
                                prob_fail_by_fh = 1.0 - cumulative_survival_probs_val[:, fh_val - 1]
                                probs_fail_at_eval_horizons.append(prob_fail_by_fh.unsqueeze(1))
                            else:
                                probs_fail_at_eval_horizons.append(
                                    torch.zeros_like(cumulative_survival_probs_val[:, 0]).unsqueeze(1))
                        if probs_fail_at_eval_horizons:
                            preds_for_eval = torch.cat(probs_fail_at_eval_horizons, dim=1)
                            all_fail_preds_eval.append(preds_for_eval.cpu().numpy())
                            all_fail_targets_eval.append(eval_fail_tgt_val.cpu().numpy())

            if ema: ema.restore()  # Restore original model parameters after validation

            print_avg_losses(epoch, "Valid", num_val_batches, total_val_loss, total_val_lp, total_val_lf, total_val_lr,
                             total_val_ldr, total_val_lmoe_aux, total_val_lentr)

            if num_val_batches > 0 and len(all_fail_preds_eval) > 0:
                all_fail_preds_eval_np = np.concatenate(all_fail_preds_eval, axis=0)
                all_fail_targets_eval_np = np.concatenate(all_fail_targets_eval, axis=0)
                if all_fail_targets_eval_np.ndim == 2 and all_fail_preds_eval_np.ndim == 2 and \
                        all_fail_targets_eval_np.shape == all_fail_preds_eval_np.shape and \
                        all_fail_targets_eval_np.shape[1] == len(FAIL_HORIZONS):
                    auroc_scores, pr_auc_scores = [], []
                    for i_fh in range(all_fail_targets_eval_np.shape[1]):
                        targets_fh = all_fail_targets_eval_np[:, i_fh];
                        preds_fh = all_fail_preds_eval_np[:, i_fh]
                        if len(np.unique(targets_fh)) > 1:
                            try:
                                auroc = roc_auc_score(targets_fh, preds_fh);
                                pr_auc = average_precision_score(targets_fh, preds_fh)
                                auroc_scores.append(auroc);
                                pr_auc_scores.append(pr_auc)
                            except ValueError as e:
                                print(f"  Skipping metrics for eval fail horizon {FAIL_HORIZONS[i_fh]}: {e}")
                        else:
                            print(
                                f"  Skipping metrics for eval fail horizon {FAIL_HORIZONS[i_fh]} (single class in targets)")
                    if auroc_scores:
                        print(f"  Avg Valid Fail AUROC: {np.mean(auroc_scores):.4f} (Horizons: {FAIL_HORIZONS})")
                        print(f"  Avg Valid Fail PR-AUC: {np.mean(pr_auc_scores):.4f} (Horizons: {FAIL_HORIZONS})")
                else:
                    print("  Could not compute AUROC/PR-AUC for failure due to shape mismatch or empty arrays.")

    model_to_save = model
    if ema:  # Save EMA weights if available
        ema.store()
        ema.copy_to()
        # model_to_save refers to model which now has EMA weights
    torch.save(model_to_save.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to: {MODEL_SAVE_PATH} (EMA weights saved if EMA was active)")
    print(f"Preprocessor config saved to {PREPROCESSOR_SAVE_PATH}")


def print_avg_losses(epoch, phase, num_batches, total_loss, lp, lf, lr, ldr, lmoe_aux, lentr):
    if num_batches > 0:
        print(
            f"E{epoch + 1} Avg {phase} L: {total_loss / num_batches:.3f} "
            f"(P:{lp / num_batches:.3f} F:{lf / num_batches:.3f} R:{lr / num_batches:.3f} "
            f"Dr:{ldr / num_batches:.3f} MoeA:{lmoe_aux / num_batches:.3f} Entr:{lentr / num_batches:.3f})"
        )
    else:
        print(f"E{epoch + 1} No batches processed for {phase} phase.")


if __name__ == '__main__':
    print("--- Script Version: Multi-Task Foundational Model with Survival Head & Enhancements (v6) ---")
    print(f"IMPORTANT: TRAIN_DIR ('{TRAIN_DIR}') and VALID_DIR ('{VALID_DIR}') must be set correctly.")
    if BASE_DATA_DIR == "../../data/time_series/1": print(
        "\nWARNING: Using default example BASE_DATA_DIR. Paths might be incorrect.\n")
    train_and_save_model()
    print(
        "\nReminder: Survival loss uses label smoothing (0.05 for censored) and follows the prompt's specified formula.")
