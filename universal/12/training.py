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
BASE_DATA_DIR = "../../data/time_series/1"
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAINING")
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")

# Model & Task Parameters
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]
FAIL_HORIZONS = [3, 5, 10]
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0]
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

# New MoE Parameters
MOE_TOP_K = 2
MOE_NOISE_STD = 1.0
AUX_LOSS_COEFF = 0.01
ENTROPY_REG_COEFF = 0.01

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == 'cuda'

MODEL_SAVE_PATH = "foundation_multitask_model_v3_moe_vectorized_cost_sensitive.pth" # Updated model name
PREPROCESSOR_SAVE_PATH = "foundation_multitask_preprocessor_v3.npz"

# Loss Function Parameters
HUBER_DELTA = 1.0
# FOCAL_ALPHA_PARAM = 0.25 # Kept for potential other uses, but fail/RCA use specific alphas
# FOCAL_GAMMA = 2.0      # Kept for potential other uses, but fail/RCA use specific gammas

# Loss weights
W_PRED = 1.0
W_FAIL = 1.0
W_RCA = 0.5

# Cost-sensitive parameters
FAILURE_HEAD_POS_WEIGHT = 1000.0
FAILURE_FOCAL_ALPHA = 0.995
FAILURE_FOCAL_GAMMA = 1.5
FAILURE_FOCAL_BLEND_COEFF = 0.2 # Tune between 0-0.5

RCA_FOCAL_ALPHA = 0.98
RCA_FOCAL_GAMMA = 1.0
RCA_FOCAL_BLEND_COEFF = 0.1 # Tune as needed


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
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=max(seq_len, 5000))
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

    def forward(self, x):  # x: [Batch*MaxSensors, SeqLen, InputDim]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)
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


# --- MoE Components (Refactored) ---
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


# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'): # Default alpha and gamma
        super(FocalLoss, self).__init__()
        self.alpha_param = alpha # Specific alpha for this instance
        self.gamma = gamma       # Specific gamma for this instance
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        # Calculate alpha_t dynamically based on targets
        alpha_t = targets * self.alpha_param + (1.0 - targets) * (1.0 - self.alpha_param)
        focal_term = alpha_t * ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        elif self.reduction == 'none':
            return focal_term
        return focal_term


# --- RCA Positive Weight Calculation ---
def make_rca_pos_weight(rca_targets, sensor_mask, eps=1e-6):
    """
    rca_targets : [B, S]
    sensor_mask : [B, S]   (0 for padded sensors)
    """
    pos = (rca_targets * sensor_mask).sum()
    neg = ((1.0 - rca_targets) * sensor_mask).sum()
    # print(f"RCA pos: {pos.item()}, neg: {neg.item()}, calculated_weight: {(neg + eps) / (pos + eps)}") # For debugging
    return (neg + eps) / (pos + eps)


# --- Data Handling Utilities (mostly unchanged) ---
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
                    print(
                        f"Warning: First file for stats ({fp}) has {len(canonical_sensor_names)} sensors, less than target {target_num_sensors}. Adjusting.")
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
            print(f"Warning: Skipping file {fp} during stats calc: {e}"); continue
        if (fp_idx + 1) % 20 == 0: print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for stats...")
    if canonical_sensor_names is None or target_num_sensors == 0 or np.sum(counts) == 0: print(
        "Error: Failed to calculate global stats."); return None, None, []
    valid_indices = counts > 0
    final_canonical_names = [name for i, name in enumerate(canonical_sensor_names) if valid_indices[i]]
    final_means = sums[valid_indices] / counts[valid_indices]
    final_stds = np.sqrt(sum_sqs[valid_indices] / counts[valid_indices] - final_means ** 2)
    final_stds[final_stds < 1e-8] = 1e-8
    if len(final_canonical_names) == 0: print("Error: No valid global stats computed."); return None, None, []
    print(f"Global stats calculated for {len(final_canonical_names)} sensors.")
    return final_means, final_stds, final_canonical_names


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizons, fail_horizons, rca_failure_lookahead, model_max_sensors_dim,
                 global_means, global_stds, canonical_sensor_names):
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
            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons), self.rca_failure_lookahead)
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
        input_slice_normed = features_normalized_aligned[window_start_idx: window_start_idx + self.seq_len]
        padded_input_normed = np.zeros((self.seq_len, self.model_max_sensors_dim), dtype=np.float32)
        sensor_mask = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        num_to_copy = min(self.num_globally_normed_features, self.model_max_sensors_dim)
        padded_input_normed[:, :num_to_copy] = input_slice_normed[:, :num_to_copy]
        for k_idx in range(num_to_copy):  # Renamed k to k_idx to avoid conflict
            if not np.all(np.isnan(input_slice_normed[:, k_idx])): sensor_mask[k_idx] = 1.0
        padded_input_normed[np.isnan(padded_input_normed)] = 0.0
        last_known_normed = padded_input_normed[-1, :].copy()
        delta_targets_normed = np.zeros((self.model_max_sensors_dim, len(self.pred_horizons)), dtype=np.float32)
        for i_h, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < features_normalized_aligned.shape[0]:
                target_values_all_normed = features_normalized_aligned[target_idx, :]
                for k_idx in range(num_to_copy):
                    if sensor_mask[k_idx] > 0 and not np.isnan(target_values_all_normed[k_idx]):
                        delta_targets_normed[k_idx, i_h] = target_values_all_normed[k_idx] - last_known_normed[k_idx]
        fail_targets = np.zeros(len(self.fail_horizons), dtype=np.float32)
        for i_fh, fh in enumerate(self.fail_horizons):
            start, end = window_start_idx + self.seq_len, window_start_idx + self.seq_len + fh
            if end <= len(flags_full) and np.any(flags_full[start:end]): fail_targets[i_fh] = 1.0
        rca_targets = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        start_r, end_r = window_start_idx + self.seq_len, window_start_idx + self.seq_len + self.rca_failure_lookahead
        if end_r <= len(flags_full) and np.any(flags_full[start_r:end_r]):
            current_window_raw = raw_features_aligned[window_start_idx: window_start_idx + self.seq_len, :]
            future_lookahead_raw = raw_features_aligned[start_r:end_r, :]
            for k_idx in range(num_to_copy):
                if sensor_mask[k_idx] > 0:
                    sensor_data_current_window_raw = current_window_raw[:, k_idx];
                    sensor_data_future_lookahead_raw = future_lookahead_raw[:, k_idx]
                    valid_current_raw = sensor_data_current_window_raw[~np.isnan(sensor_data_current_window_raw)];
                    valid_future_raw = sensor_data_future_lookahead_raw[~np.isnan(sensor_data_future_lookahead_raw)]
                    if len(valid_current_raw) > 0 and len(valid_future_raw) > 0:
                        mean_current_raw = np.mean(valid_current_raw);
                        std_current_raw = np.std(valid_current_raw);
                        std_current_raw = max(std_current_raw, 1e-6)
                        if np.any(np.abs(valid_future_raw - mean_current_raw) > 3 * std_current_raw): rca_targets[
                            k_idx] = 1.0
        return {"input_features": torch.from_numpy(padded_input_normed), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_globally_std": torch.from_numpy(last_known_normed),
                "pred_delta_targets_globally_std": torch.from_numpy(delta_targets_normed),
                "fail_targets": torch.from_numpy(fail_targets), "rca_targets": torch.from_numpy(rca_targets)}


# --- Foundational Multi-Task Model (MoE with Vectorized TopK) ---
class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, model_max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 num_shared_experts, moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim,
                 pred_horizons_len, fail_horizons_len,
                 moe_top_k, moe_noise_std, aux_loss_coeff, entropy_reg_coeff):
        super().__init__()
        self.model_max_sensors = model_max_sensors
        self.seq_len = seq_len
        self.num_shared_experts = num_shared_experts
        self.moe_output_dim = moe_output_dim
        self.transformer_d_model = transformer_d_model
        self.sensor_tcn_out_dim = sensor_tcn_out_dim

        self.moe_top_k = moe_top_k
        self.moe_noise_std = moe_noise_std
        self.aux_loss_coeff = aux_loss_coeff
        self.entropy_reg_coeff = entropy_reg_coeff

        print("FoundationalTimeSeriesModel: Using MMoE with shared experts and VECTORIZED soft top-k routing.")

        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)
        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_tcn_out_dim,
                                                        transformer_d_model) if sensor_tcn_out_dim != transformer_d_model else nn.Identity()
        self.inter_sensor_transformer = InterSensorTransformer(transformer_d_model, transformer_nhead,
                                                               transformer_nlayers, model_max_sensors)

        self.experts_shared = nn.ModuleList(
            [Expert(moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_shared_experts)])

        gate_input_dim_global = transformer_d_model * 2
        gate_input_dim_rca_token = transformer_d_model

        self.gates = nn.ModuleDict({
            "forecast": GatingNetwork(gate_input_dim_global, num_shared_experts),
            "fail": GatingNetwork(gate_input_dim_global, num_shared_experts),
            "rca": GatingNetwork(gate_input_dim_rca_token, num_shared_experts)
        })

        self.pred_head = nn.Linear(sensor_tcn_out_dim + transformer_d_model + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(sensor_tcn_out_dim + transformer_d_model + moe_output_dim, 1)

    def _apply_moe_topk(self, x_expert_input, gate_input, gate_network, experts_modulelist, k, noise_std):
        """
        Applies soft top-k routing for Mixture of Experts (Vectorized version).
        Args:
            x_expert_input: Input tensor for the experts [BatchSize_effective, ExpertInputDim].
            gate_input: Input tensor for the gate network [BatchSize_effective, GateInputDim].
            gate_network: The gating network module.
            experts_modulelist: ModuleList of expert networks.
            k: Number of experts to select.
            noise_std: Standard deviation for noise added to gate logits.
        Returns:
            y: Combined expert output [BatchSize_effective, ExpertOutputDim].
            load_balance_loss: Scalar load balancing loss.
            gate_logits: Logits from the gate [BatchSize_effective, NumExperts], for entropy loss.
        """
        # 1. Gate
        logits = gate_network(gate_input)  # [B_eff, num_experts]
        if self.training and noise_std > 0:  # noise_std check also ensures it's not None
            logits = logits + torch.randn_like(logits) * noise_std

        # Ensure k is not greater than the number of experts
        num_experts = len(experts_modulelist)
        eff_k = min(k, num_experts)

        topk_val, topk_idx = torch.topk(logits, eff_k, dim=-1)  # [B_eff, eff_k]
        topk_w = torch.softmax(topk_val, dim=-1)  # [B_eff, eff_k]

        # 2. Experts (compute once)
        #    Stack: [B_eff, NumExperts, ExpertOutputDim]
        all_out = torch.stack([e(x_expert_input) for e in experts_modulelist], dim=1)

        # 3. Gather the k chosen experts
        expert_output_dim = all_out.size(-1)
        # expand topk_idx to match the dimension of all_out for gather
        # topk_idx: [B_eff, eff_k] -> gather_idx: [B_eff, eff_k, ExpertOutputDim]
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, eff_k, expert_output_dim)

        # sel_out: [B_eff, eff_k, ExpertOutputDim]
        sel_out = all_out.gather(1, gather_idx)

        # Weighted sum of selected expert outputs
        # topk_w: [B_eff, eff_k] -> topk_w.unsqueeze(-1): [B_eff, eff_k, 1]
        y = (sel_out * topk_w.unsqueeze(-1)).sum(dim=1)  # [B_eff, ExpertOutputDim]

        # 4. Load-balance & (optional) entropy
        # P_i: mean router probability for expert i
        router_prob_for_loss = torch.softmax(logits, -1)  # [B_eff, num_experts]
        avg_router_prob = router_prob_for_loss.mean(0)  # [num_experts] (P_i)

        # f_i: fraction of examples routed to expert i (among top-k choices)
        # Create a tensor of ones with the same shape and type as topk_idx for scatter_add
        ones_for_scatter = torch.ones_like(topk_idx, dtype=router_prob_for_loss.dtype).reshape(-1)
        expert_frac = torch.zeros_like(avg_router_prob).scatter_add_(
            0, topk_idx.reshape(-1),  # Flatten indices for scattering
            ones_for_scatter
        ) / (x_expert_input.size(0) * eff_k)  # Normalize by total selections (f_i)

        load_balance_loss = self.num_shared_experts * (avg_router_prob * expert_frac).sum()

        return y, load_balance_loss, logits

    def forward(self, x_features_globally_std, sensor_mask):
        batch_size, seq_len, _ = x_features_globally_std.shape

        x_input_masked_for_tcn = x_features_globally_std.permute(0, 2, 1)
        x_input_masked_for_tcn = x_input_masked_for_tcn * sensor_mask.unsqueeze(-1)
        last_val_globally_std = x_input_masked_for_tcn[:, :, -1].clone()

        x_permuted_for_tcn_input = x_input_masked_for_tcn.permute(0, 2, 1)
        x_reshaped_for_encoder = x_permuted_for_tcn_input.reshape(batch_size * self.model_max_sensors, seq_len,
                                                                  SENSOR_INPUT_DIM)

        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, self.model_max_sensors, seq_len,
                                                                         self.sensor_tcn_out_dim)
        sensor_temporal_features = sensor_temporal_features * sensor_mask.view(batch_size, self.model_max_sensors, 1, 1)

        pooled_sensor_features = torch.mean(sensor_temporal_features, dim=2)
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
        router_input_global = torch.cat([mean_ctx_global, std_ctx_global], dim=-1)

        moe_forecast_output, aux_f, logits_f = self._apply_moe_topk(
            mean_ctx_global, router_input_global, self.gates["forecast"], self.experts_shared,
            self.moe_top_k, self.moe_noise_std
        )
        moe_fail_output, aux_fail, logits_fail = self._apply_moe_topk(
            mean_ctx_global, router_input_global, self.gates["fail"], self.experts_shared,
            self.moe_top_k, self.moe_noise_std
        )

        x_flat_rca_expert_input = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        valid_token_mask_rca = sensor_mask.view(-1).bool()
        x_flat_rca_expert_input_valid = x_flat_rca_expert_input[valid_token_mask_rca]

        moe_rca_output_flat_valid = torch.empty(0, self.moe_output_dim, device=DEVICE,
                                                dtype=moe_forecast_output.dtype)  # Ensure it has a default even if not computed
        aux_rca = torch.tensor(0.0, device=DEVICE)
        logits_rca_valid = None  # Ensure it's defined

        if x_flat_rca_expert_input_valid.size(0) > 0:
            # Gate input is the same as expert input for token-level RCA
            x_flat_rca_gate_input_valid = x_flat_rca_expert_input_valid
            moe_rca_output_flat_valid, aux_rca, logits_rca_valid = self._apply_moe_topk(
                x_flat_rca_expert_input_valid, x_flat_rca_gate_input_valid, self.gates["rca"], self.experts_shared,
                self.moe_top_k, self.moe_noise_std
            )

        moe_rca_output_flat = torch.zeros(batch_size * self.model_max_sensors, self.moe_output_dim, device=DEVICE,
                                          dtype=moe_forecast_output.dtype)
        if x_flat_rca_expert_input_valid.size(0) > 0:
            moe_rca_output_flat[valid_token_mask_rca] = moe_rca_output_flat_valid

        total_aux_loss = self.aux_loss_coeff * (aux_f + aux_fail + aux_rca)

        total_entropy_loss = torch.tensor(0.0, device=DEVICE)
        if self.entropy_reg_coeff > 0:
            for gate_logits_set in [logits_f, logits_fail, logits_rca_valid]:
                if gate_logits_set is not None and gate_logits_set.numel() > 0:
                    probs = F.softmax(gate_logits_set, dim=-1)
                    log_probs = F.log_softmax(gate_logits_set, dim=-1)  # Use log_softmax for stability
                    total_entropy_loss -= self.entropy_reg_coeff * (probs * log_probs).sum(dim=-1).mean()

        tcn_features_last_step = sensor_temporal_features[:, :, -1, :]
        moe_f_expanded = moe_forecast_output.unsqueeze(1).expand(-1, self.model_max_sensors, -1)
        pred_head_input_features = torch.cat([tcn_features_last_step, cross_sensor_context_masked, moe_f_expanded],
                                             dim=-1)
        pred_delta_globally_std = self.pred_head(pred_head_input_features)
        pred_abs_globally_std = last_val_globally_std.unsqueeze(-1) + pred_delta_globally_std
        pred_abs_globally_std = pred_abs_globally_std * sensor_mask.unsqueeze(-1)

        fail_logits = self.fail_head(moe_fail_output)

        tcn_flat = tcn_features_last_step.reshape(-1, self.sensor_tcn_out_dim)
        ctx_flat = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        rca_head_input_flat = torch.cat([tcn_flat, ctx_flat, moe_rca_output_flat], dim=-1)
        rca_logits_flat = self.rca_head(rca_head_input_flat).squeeze(-1)
        rca_logits = rca_logits_flat.view(batch_size, self.model_max_sensors)

        return pred_abs_globally_std, fail_logits, rca_logits, total_aux_loss, total_entropy_loss


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
             model_max_sensors_dim=model_max_sensors_dim, seq_len=SEQ_LEN, pred_horizons=np.array(PRED_HORIZONS),
             fail_horizons=np.array(FAIL_HORIZONS), rca_failure_lookahead=RCA_FAILURE_LOOKAHEAD)
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
        pred_horizons_len=len(PRED_HORIZONS), fail_horizons_len=len(FAIL_HORIZONS),
        moe_top_k=MOE_TOP_K, moe_noise_std=MOE_NOISE_STD,
        aux_loss_coeff=AUX_LOSS_COEFF, entropy_reg_coeff=ENTROPY_REG_COEFF
    ).to(DEVICE)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, model_max_sensors_dim, global_means,
                                                  global_stds, canonical_sensor_names)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, model_max_sensors_dim, global_means,
                                                  global_stds, canonical_sensor_names)
    if len(train_dataset) == 0: print("ERROR: Training dataset empty."); return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=AMP_ENABLED)  # Consider num_workers > 0
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=AMP_ENABLED) if len(valid_dataset) > 0 else None

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)
    num_training_steps = EPOCHS * len(train_loader);
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=AMP_ENABLED)

    huber_loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')

    # ---- Loss objects ----
    # Failure head specific losses
    failure_pos_w = torch.tensor(FAILURE_HEAD_POS_WEIGHT, device=DEVICE)
    bce_fail = nn.BCEWithLogitsLoss(pos_weight=failure_pos_w)
    focal_fail = FocalLoss(alpha=FAILURE_FOCAL_ALPHA, gamma=FAILURE_FOCAL_GAMMA, reduction='mean') # failure only

    # RCA head specific Focal loss (BCE for RCA is created on-the-fly)
    focal_rca = FocalLoss(alpha=RCA_FOCAL_ALPHA, gamma=RCA_FOCAL_GAMMA, reduction='none') # rca only, elementwise for masking

    print("Starting multi-task training (Forecast, Fail, RCA) with VECTORIZED MoE and Cost-Sensitive Losses...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss_epoch, total_lp_epoch, total_lf_epoch, total_lr_epoch, total_laux_epoch, total_lentr_epoch = 0, 0, 0, 0, 0, 0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            input_feat = batch["input_features"].to(DEVICE);
            sensor_m = batch["sensor_mask"].to(DEVICE)
            last_k_std = batch["last_known_values_globally_std"].to(DEVICE);
            delta_tgt_std = batch["pred_delta_targets_globally_std"].to(DEVICE)
            fail_tgt = batch["fail_targets"].to(DEVICE);
            rca_tgt = batch["rca_targets"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=AMP_ENABLED):
                pred_abs_std, fail_logits, rca_logits, l_aux, l_entropy = model(input_feat, sensor_m)

                # Prediction Loss (Huber)
                abs_target_std = last_k_std.unsqueeze(-1) + delta_tgt_std
                lp_elements = huber_loss_fn(pred_abs_std, abs_target_std)
                lp_masked = lp_elements * sensor_m.unsqueeze(-1)
                num_active_forecast_elements = (sensor_m.sum() * len(PRED_HORIZONS)).clamp(min=1e-9)
                lp = lp_masked.sum() / num_active_forecast_elements

                # Failure Loss (Cost-weighted BCE + optional Focal)
                lf_bce = bce_fail(fail_logits, fail_tgt)  # main driver
                lf_focal_term = focal_fail(fail_logits, fail_tgt) * FAILURE_FOCAL_BLEND_COEFF # regularizer
                lf = lf_bce + lf_focal_term

                # RCA Loss (Adaptive Cost-weighted BCE + optional Focal)
                pos_w_rca = make_rca_pos_weight(rca_tgt, sensor_m)
                # Create BCE loss for RCA on-the-fly due to adaptive pos_weight
                bce_rca_dynamic = nn.BCEWithLogitsLoss(pos_weight=pos_w_rca.to(DEVICE)) # Ensure pos_w_rca is on correct device
                lr_bce = bce_rca_dynamic(rca_logits, rca_tgt) # main term

                # Optional RCA focal term (stabiliser)
                lr_focal_elements = focal_rca(rca_logits, rca_tgt) # elementwise
                lr_focal_term = (lr_focal_elements * sensor_m).sum() / sensor_m.sum().clamp(min=1e-9)
                lr = lr_bce + RCA_FOCAL_BLEND_COEFF * lr_focal_term

                combined_loss = W_PRED * lp + W_FAIL * lf + W_RCA * lr + l_aux + l_entropy

            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(
                    f"Warning: NaN/Inf loss. Lp:{lp.item():.3f}, Lf:{lf.item():.3f} (bce:{lf_bce.item():.3f}, foc:{lf_focal_term.item():.3f}), Lr:{lr.item():.3f} (bce:{lr_bce.item():.3f}, foc:{lr_focal_term.item():.3f}), Aux:{l_aux.item():.3f}, Entr:{l_entropy.item():.3f}. Skipping update.")
                continue

            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss_epoch += combined_loss.item();
            total_lp_epoch += lp.item();
            total_lf_epoch += lf.item()
            total_lr_epoch += lr.item();
            total_laux_epoch += l_aux.item();
            total_lentr_epoch += l_entropy.item()
            num_batches += 1

            if batch_idx > 0 and batch_idx % (len(train_loader) // 10 if len(train_loader) >= 10 else 1) == 0:
                lr_val = optimizer.param_groups[0]['lr']
                print(
                    f"E{epoch + 1} B{batch_idx + 1}/{len(train_loader)} LR:{lr_val:.1e} | L:{combined_loss.item():.3f} (P:{lp.item():.3f} F:{lf.item():.3f} R:{lr.item():.3f} Aux:{l_aux.item():.3f} Entr:{l_entropy.item():.3f})")

        print_avg_losses(epoch, "Train", num_batches, total_loss_epoch, total_lp_epoch, total_lf_epoch, total_lr_epoch,
                         total_laux_epoch, total_lentr_epoch)

        if valid_loader:
            model.eval()
            total_val_loss, total_val_lp, total_val_lf, total_val_lr, total_val_laux, total_val_lentr = 0, 0, 0, 0, 0, 0
            num_val_batches = 0
            with torch.no_grad():
                for batch_val in valid_loader:  # Renamed batch to batch_val
                    input_feat_val = batch_val["input_features"].to(DEVICE);
                    sensor_m_val = batch_val["sensor_mask"].to(DEVICE)
                    last_k_std_val = batch_val["last_known_values_globally_std"].to(DEVICE);
                    delta_tgt_std_val = batch_val["pred_delta_targets_globally_std"].to(DEVICE)
                    fail_tgt_val = batch_val["fail_targets"].to(DEVICE);
                    rca_tgt_val = batch_val["rca_targets"].to(DEVICE)

                    with autocast(enabled=AMP_ENABLED):
                        pred_abs_std_val, fail_logits_val, rca_logits_val, l_aux_val, l_entr_val = model(input_feat_val,
                                                                                                         sensor_m_val)
                        # Prediction Loss
                        abs_target_std_val = last_k_std_val.unsqueeze(-1) + delta_tgt_std_val
                        lp_val_el = huber_loss_fn(pred_abs_std_val, abs_target_std_val)
                        lp_val = (lp_val_el * sensor_m_val.unsqueeze(-1)).sum() / (
                                    sensor_m_val.sum() * len(PRED_HORIZONS)).clamp(min=1e-9)

                        # Failure Loss (Validation)
                        lf_bce_val = bce_fail(fail_logits_val, fail_tgt_val)
                        lf_focal_val = focal_fail(fail_logits_val, fail_tgt_val) * FAILURE_FOCAL_BLEND_COEFF
                        lf_val = lf_bce_val + lf_focal_val

                        # RCA Loss (Validation)
                        pos_w_rca_val = make_rca_pos_weight(rca_tgt_val, sensor_m_val)
                        bce_rca_dynamic_val = nn.BCEWithLogitsLoss(pos_weight=pos_w_rca_val.to(DEVICE))
                        lr_bce_val = bce_rca_dynamic_val(rca_logits_val, rca_tgt_val)
                        lr_focal_elements_val = focal_rca(rca_logits_val, rca_tgt_val)
                        lr_focal_val = (lr_focal_elements_val * sensor_m_val).sum() / sensor_m_val.sum().clamp(min=1e-9)
                        lr_val = lr_bce_val + RCA_FOCAL_BLEND_COEFF * lr_focal_val

                        val_loss = W_PRED * lp_val + W_FAIL * lf_val + W_RCA * lr_val + l_aux_val + l_entr_val

                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        total_val_loss += val_loss.item();
                        total_val_lp += lp_val.item();
                        total_val_lf += lf_val.item()
                        total_val_lr += lr_val.item();
                        total_val_laux += l_aux_val.item();
                        total_val_lentr += l_entr_val.item()
                        num_val_batches += 1
            print_avg_losses(epoch, "Valid", num_val_batches, total_val_loss, total_val_lp, total_val_lf, total_val_lr,
                             total_val_laux, total_val_lentr)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model: {MODEL_SAVE_PATH}, Preprocessor: {PREPROCESSOR_SAVE_PATH}")


def print_avg_losses(epoch, phase, num_batches, total_loss, lp, lf, lr, laux, lentr):
    if num_batches > 0:
        print(
            f"E{epoch + 1} Avg {phase} L: {total_loss / num_batches:.3f} (P:{lp / num_batches:.3f} F:{lf / num_batches:.3f} R:{lr / num_batches:.3f} Aux:{laux / num_batches:.3f} Entr:{lentr / num_batches:.3f})")
    else:
        print(f"E{epoch + 1} No batches processed for {phase} phase.")


if __name__ == '__main__':
    print(
        "--- Script Version: Multi-Task Foundational Model with Refactored MoE (Vectorized Soft Top-k, MMoE, Token RCA, Richer Router, Entropy Reg, Cost-Sensitive) ---")
    print(f"IMPORTANT: TRAIN_DIR ('{TRAIN_DIR}') and VALID_DIR ('{VALID_DIR}') must be set correctly.")
    if BASE_DATA_DIR == "../../data/time_series/1": print(
        "\nWARNING: Using default example BASE_DATA_DIR. Paths might be incorrect.\n")
    train_and_save_model()