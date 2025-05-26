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
from sklearn.metrics import roc_auc_score, average_precision_score # Added for Suggestion 7

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
# MOE_NOISE_STD = 1.0 # Original
MOE_NOISE_STD = 0.3    # Suggestion 6: Reduce MOE_NOISE_STD
AUX_LOSS_COEFF = 0.01
# ENTROPY_REG_COEFF = 0.01 # Original
ENTROPY_REG_COEFF = 0.001 # Suggestion 6: Try dropping ENTROPY_REG_COEFF

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 40 # Original value, may need adjustment for annealing
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == 'cuda'

MODEL_SAVE_PATH = "foundation_multitask_model_v4_suggestions.pth" # Updated path
PREPROCESSOR_SAVE_PATH = "foundation_multitask_preprocessor_v4.npz" # Updated path

# Loss Function Parameters
HUBER_DELTA = 1.0
FOCAL_ALPHA_PARAM = 0.25
FOCAL_GAMMA = 2.0

# Loss weights (Suggestion 2)
W_PRED = 1.0
W_FAIL_BASE = 2.5  # Changed from 1.0, chosen from 2.0-3.0
W_RCA_BASE  = 1.0  # Changed from 0.5
ANNEAL_LOSS_WEIGHTS_START_EPOCH = 10 # Anneal after 10 epochs (i.e., starting from epoch 10 if 0-indexed)
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


# --- Per-Sensor TCN Encoder (Modified for Suggestion 3: CLS Token) ---
class PerSensorEncoderTCN(nn.Module):
    def __init__(self, input_dim, proj_dim, tcn_out_dim, seq_len, num_levels, kernel_size, dropout):
        super(PerSensorEncoderTCN, self).__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        # Suggestion 3: Add a learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim))
        # Positional encoding max_len needs to accommodate seq_len + 1 (for CLS token)
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
        self.tcn_out_dim = tcn_out_dim # Store for convenience

    def forward(self, x):  # x: [Batch*MaxSensors, SeqLen, InputDim]
        x_proj = self.input_proj(x) # [B*MS, SeqLen, ProjDim]

        # Suggestion 3: Prepend CLS token
        cls_tokens_expanded = self.cls_token.expand(x_proj.size(0), -1, -1) # [B*MS, 1, ProjDim]
        x_with_cls = torch.cat((cls_tokens_expanded, x_proj), dim=1) # [B*MS, SeqLen+1, ProjDim]

        x_pos_encoded = self.pos_encoder(x_with_cls) # [B*MS, SeqLen+1, ProjDim]
        x_permuted_for_tcn = x_pos_encoded.permute(0, 2, 1) # [B*MS, ProjDim, SeqLen+1]

        x_tcn_out_with_cls = self.tcn_network(x_permuted_for_tcn) # [B*MS, TcnOutDim, SeqLen+1]
        x_tcn_out_permuted_back = x_tcn_out_with_cls.permute(0, 2, 1) # [B*MS, SeqLen+1, TcnOutDim]

        # Suggestion 3: Extract CLS token output and main sequence output
        h_cls = x_tcn_out_permuted_back[:, 0] # [B*MS, TcnOutDim]
        main_sequence_features = x_tcn_out_permuted_back[:, 1:] # [B*MS, SeqLen, TcnOutDim]

        main_sequence_normed = self.final_norm(main_sequence_features)
        # Also norm h_cls for consistency if it's directly used or averaged later
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
        for k_idx in range(num_to_copy):
            if not np.all(np.isnan(input_slice_normed[:, k_idx])): sensor_mask[k_idx] = 1.0
        padded_input_normed[np.isnan(padded_input_normed)] = 0.0
        last_known_normed = padded_input_normed[-1, :].copy() # This is last_val for RCA novelty
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
        self.seq_len = seq_len # Original data sequence length
        self.num_shared_experts = num_shared_experts
        self.moe_output_dim = moe_output_dim
        self.transformer_d_model = transformer_d_model
        self.sensor_tcn_out_dim = sensor_tcn_out_dim

        self.moe_top_k = moe_top_k
        self.moe_noise_std = moe_noise_std
        self.aux_loss_coeff = aux_loss_coeff
        self.entropy_reg_coeff = entropy_reg_coeff

        print("FoundationalTimeSeriesModel: Using MMoE with shared experts and VECTORIZED soft top-k routing.")
        print(f"MoE Noise STD: {self.moe_noise_std}, Entropy Reg Coeff: {self.entropy_reg_coeff}")

        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)
        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_tcn_out_dim,
                                                        transformer_d_model) if sensor_tcn_out_dim != transformer_d_model else nn.Identity()
        self.inter_sensor_transformer = InterSensorTransformer(transformer_d_model, transformer_nhead,
                                                               transformer_nlayers, model_max_sensors)

        self.experts_shared = nn.ModuleList(
            [Expert(moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_shared_experts)])

        # Suggestion 3: Update gate_input_dim_global for CLS token feature
        # Original: transformer_d_model * 2
        # New: transformer_d_model * 2 (mean/std global context) + sensor_tcn_out_dim (global CLS context)
        gate_input_dim_global_forecast_fail = transformer_d_model * 2 + sensor_tcn_out_dim
        gate_input_dim_rca_token = transformer_d_model # Unchanged by CLS for RCA gate

        self.gates = nn.ModuleDict({
            "forecast": GatingNetwork(gate_input_dim_global_forecast_fail, num_shared_experts),
            "fail": GatingNetwork(gate_input_dim_global_forecast_fail, num_shared_experts), # Uses CLS enhanced global input
            "rca": GatingNetwork(gate_input_dim_rca_token, num_shared_experts)
        })

        self.pred_head = nn.Linear(sensor_tcn_out_dim + transformer_d_model + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len) # Input from MoE dedicated to fail

        # Suggestion 5: RCA head input dim increases by 1 (for delta novelty score)
        rca_head_input_dim = sensor_tcn_out_dim + transformer_d_model + moe_output_dim + 1
        self.rca_head = nn.Linear(rca_head_input_dim, 1)


    def _apply_moe_topk(self, x_expert_input, gate_input, gate_network, experts_modulelist, k, noise_std):
        logits = gate_network(gate_input)
        if self.training and noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std

        num_experts = len(experts_modulelist)
        eff_k = min(k, num_experts)

        topk_val, topk_idx = torch.topk(logits, eff_k, dim=-1)
        topk_w = torch.softmax(topk_val, dim=-1)

        all_out = torch.stack([e(x_expert_input) for e in experts_modulelist], dim=1)

        expert_output_dim = all_out.size(-1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, eff_k, expert_output_dim)
        sel_out = all_out.gather(1, gather_idx)
        y = (sel_out * topk_w.unsqueeze(-1)).sum(dim=1)

        router_prob_for_loss = torch.softmax(logits, -1)
        avg_router_prob = router_prob_for_loss.mean(0)
        ones_for_scatter = torch.ones_like(topk_idx, dtype=router_prob_for_loss.dtype).reshape(-1)
        expert_frac = torch.zeros_like(avg_router_prob).scatter_add_(
            0, topk_idx.reshape(-1), ones_for_scatter
        ) / (x_expert_input.size(0) * eff_k if x_expert_input.size(0) > 0 else 1.0) # Avoid div by zero

        load_balance_loss = self.num_shared_experts * (avg_router_prob * expert_frac).sum()
        return y, load_balance_loss, logits

    def forward(self, x_features_globally_std, sensor_mask, last_known_values_globally_std_for_novelty):
        batch_size, seq_len_data, _ = x_features_globally_std.shape # seq_len_data is the original SEQ_LEN

        # Permute for TCN: [B, MaxSensors, SeqLen] -> [B, MaxSensors, FeatDim=1, SeqLen] if needed, or just handle dimensions.
        # Input to per_sensor_encoder should be [B*MaxSensors, SeqLen, InputDim=1]
        x_input_for_tcn = x_features_globally_std.reshape(batch_size * self.model_max_sensors, seq_len_data, SENSOR_INPUT_DIM)

        # Suggestion 3: per_sensor_encoder now returns (main_sequence_features, h_cls_output)
        # main_sequence_features_flat: [B*MS, SeqLen, TcnOutDim]
        # h_cls_flat: [B*MS, TcnOutDim]
        sensor_temporal_features_flat, h_cls_flat = self.per_sensor_encoder(x_input_for_tcn)

        # Reshape main sequence features
        sensor_temporal_features_main = sensor_temporal_features_flat.reshape(
            batch_size, self.model_max_sensors, seq_len_data, self.sensor_tcn_out_dim
        )
        sensor_temporal_features_main = sensor_temporal_features_main * sensor_mask.view(batch_size, self.model_max_sensors, 1, 1)

        # Pool main sequence features for Transformer input
        pooled_sensor_features = torch.mean(sensor_temporal_features_main, dim=2) # [B, MS, TcnOutDim]
        projected_for_inter_sensor = self.pooled_to_transformer_dim_proj(pooled_sensor_features) # [B, MS, TransformerDModel]

        transformer_padding_mask = (sensor_mask == 0)
        cross_sensor_context = self.inter_sensor_transformer(projected_for_inter_sensor, transformer_padding_mask) # [B, MS, TransformerDModel]
        cross_sensor_context_masked = cross_sensor_context * sensor_mask.unsqueeze(-1)

        # --- Global Context Calculation ---
        active_sensors_per_batch = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_ctx_global = (cross_sensor_context_masked).sum(dim=1) / active_sensors_per_batch # [B, TransformerDModel]
        mean_sq_ctx_global = ((cross_sensor_context_masked ** 2) * sensor_mask.unsqueeze(-1)).sum(dim=1) / active_sensors_per_batch
        var_ctx_global = mean_sq_ctx_global - mean_ctx_global ** 2
        std_ctx_global = torch.sqrt(var_ctx_global.clamp(min=1e-6)) # [B, TransformerDModel]

        # Suggestion 3: Incorporate CLS token output into global router input
        h_cls_per_sensor = h_cls_flat.reshape(batch_size, self.model_max_sensors, self.sensor_tcn_out_dim)
        h_cls_per_sensor_masked = h_cls_per_sensor * sensor_mask.unsqueeze(-1)
        h_cls_global_avg = h_cls_per_sensor_masked.sum(dim=1) / active_sensors_per_batch # [B, SensorTcnOutDim]

        router_input_global_forecast_fail = torch.cat([mean_ctx_global, std_ctx_global, h_cls_global_avg], dim=-1)
        # Shape: [B, TransformerDModel*2 + SensorTcnOutDim]

        # --- MoE Routing ---
        # Forecast and Fail gates use the CLS-enhanced global context
        moe_forecast_output, aux_f, logits_f = self._apply_moe_topk(
            mean_ctx_global, router_input_global_forecast_fail, self.gates["forecast"], self.experts_shared,
            self.moe_top_k, self.moe_noise_std
        )
        moe_fail_output, aux_fail, logits_fail = self._apply_moe_topk(
            mean_ctx_global, router_input_global_forecast_fail, self.gates["fail"], self.experts_shared, # Using CLS enhanced input
            self.moe_top_k, self.moe_noise_std
        )

        # RCA gate uses token-level context (cross_sensor_context_masked)
        x_flat_rca_expert_input = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        valid_token_mask_rca = sensor_mask.view(-1).bool()
        x_flat_rca_expert_input_valid = x_flat_rca_expert_input[valid_token_mask_rca]

        moe_rca_output_flat_valid = torch.empty(0, self.moe_output_dim, device=x_features_globally_std.device, dtype=moe_forecast_output.dtype)
        aux_rca = torch.tensor(0.0, device=x_features_globally_std.device)
        logits_rca_valid = None

        if x_flat_rca_expert_input_valid.size(0) > 0:
            # Gate input is the same as expert input for token-level RCA
            x_flat_rca_gate_input_valid = x_flat_rca_expert_input_valid
            moe_rca_output_flat_valid, aux_rca, logits_rca_valid = self._apply_moe_topk(
                x_flat_rca_expert_input_valid, x_flat_rca_gate_input_valid, self.gates["rca"], self.experts_shared,
                self.moe_top_k, self.moe_noise_std
            )

        moe_rca_output_flat = torch.zeros(batch_size * self.model_max_sensors, self.moe_output_dim, device=x_features_globally_std.device, dtype=moe_forecast_output.dtype)
        if x_flat_rca_expert_input_valid.size(0) > 0:
            moe_rca_output_flat[valid_token_mask_rca] = moe_rca_output_flat_valid

        total_aux_loss = self.aux_loss_coeff * (aux_f + aux_fail + aux_rca)

        total_entropy_loss = torch.tensor(0.0, device=x_features_globally_std.device)
        if self.entropy_reg_coeff > 0: # Apply to all gates that produced logits
            for gate_logits_set in [logits_f, logits_fail, logits_rca_valid]:
                if gate_logits_set is not None and gate_logits_set.numel() > 0:
                    probs = F.softmax(gate_logits_set, dim=-1)
                    log_probs = F.log_softmax(gate_logits_set, dim=-1)
                    # Suggestion 6: ENTROPY_REG_COEFF might be lower (e.g. 0.001)
                    total_entropy_loss -= self.entropy_reg_coeff * (probs * log_probs).sum(dim=-1).mean()

        # --- Prediction Head ---
        # Use last step of main sequence features from TCN
        tcn_features_last_step = sensor_temporal_features_main[:, :, -1, :] # [B, MS, TcnOutDim]
        moe_f_expanded = moe_forecast_output.unsqueeze(1).expand(-1, self.model_max_sensors, -1) # [B, MS, MoeOutDim]
        pred_head_input_features = torch.cat([tcn_features_last_step, cross_sensor_context_masked, moe_f_expanded], dim=-1)
        pred_delta_globally_std = self.pred_head(pred_head_input_features) # [B, MS, NumPredHorizons]

        # Calculate absolute predictions
        # last_val_globally_std (from input x_features...) is [B, MS] (value at t=seq_len-1 for each sensor)
        last_val_observed_for_pred = x_features_globally_std[:, -1, :] # [B, MS] assuming SENSOR_INPUT_DIM=1 and we take that one feature
        pred_abs_globally_std = last_val_observed_for_pred.unsqueeze(-1) + pred_delta_globally_std
        pred_abs_globally_std = pred_abs_globally_std * sensor_mask.unsqueeze(-1)

        # --- Failure Head ---
        fail_logits = self.fail_head(moe_fail_output) # [B, NumFailHorizons]

        # --- RCA Head (Suggestion 5: Novelty based) ---
        # Calculate novelty score (delta)
        with torch.no_grad():
            # pred_abs_globally_std is [B, MS, NumPredHorizons]
            # We need prediction for the first horizon (H=1)
            pred_next_h1 = pred_abs_globally_std[:, :, 0] # [B, MS]

        # last_known_values_globally_std_for_novelty is the actual last value from input, passed from training loop
        # It should be [B, MS]
        delta_novelty = torch.abs(pred_next_h1 - last_known_values_globally_std_for_novelty) # [B, MS]
        delta_novelty_masked = delta_novelty * sensor_mask # [B, MS]
        delta_novelty_flat = delta_novelty_masked.reshape(-1, 1) # [B*MS, 1]

        # Prepare RCA head input
        tcn_flat = tcn_features_last_step.reshape(-1, self.sensor_tcn_out_dim) # [B*MS, TcnOutDim]
        ctx_flat = cross_sensor_context_masked.reshape(-1, self.transformer_d_model) # [B*MS, TransformerDModel]
        # moe_rca_output_flat is [B*MS, MoeOutDim]

        rca_head_input_flat = torch.cat([tcn_flat, ctx_flat, moe_rca_output_flat, delta_novelty_flat], dim=-1)
        # Shape: [B*MS, TcnOutDim + TransformerDModel + MoeOutDim + 1]

        rca_logits_flat = self.rca_head(rca_head_input_flat).squeeze(-1) # [B*MS]
        rca_logits = rca_logits_flat.view(batch_size, self.model_max_sensors) # [B, MS]

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
        model_max_sensors=model_max_sensors_dim, seq_len=SEQ_LEN, # Pass original SEQ_LEN
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        num_shared_experts=NUM_SHARED_EXPERTS, moe_expert_input_dim=MOE_EXPERT_INPUT_DIM,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        pred_horizons_len=len(PRED_HORIZONS), fail_horizons_len=len(FAIL_HORIZONS),
        moe_top_k=MOE_TOP_K, moe_noise_std=MOE_NOISE_STD, # Using updated MOE_NOISE_STD
        aux_loss_coeff=AUX_LOSS_COEFF, entropy_reg_coeff=ENTROPY_REG_COEFF # Using updated ENTROPY_REG_COEFF
    ).to(DEVICE)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, model_max_sensors_dim, global_means,
                                                  global_stds, canonical_sensor_names)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, model_max_sensors_dim, global_means,
                                                  global_stds, canonical_sensor_names)
    if len(train_dataset) == 0: print("ERROR: Training dataset empty."); return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=AMP_ENABLED)
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
    focal_loss_elementwise = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA, reduction='none')
    focal_loss_mean = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA, reduction='mean')

    print("Starting multi-task training (Forecast, Fail, RCA) with VECTORIZED MoE and implemented suggestions...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss_epoch, total_lp_epoch, total_lf_epoch, total_lr_epoch, total_laux_epoch, total_lentr_epoch = 0, 0, 0, 0, 0, 0
        num_batches = 0

        # Suggestion 2: Anneal W_FAIL and W_RCA
        current_w_fail = W_FAIL_BASE
        current_w_rca = W_RCA_BASE
        if epoch >= ANNEAL_LOSS_WEIGHTS_START_EPOCH:
            decay_exponent = epoch - ANNEAL_LOSS_WEIGHTS_START_EPOCH +1 # epoch 10 -> 0.9^1, epoch 11 -> 0.9^2
            decay_multiplier = ANNEAL_LOSS_WEIGHTS_FACTOR ** decay_exponent
            current_w_fail *= decay_multiplier
            current_w_rca *= decay_multiplier
            print(f"Epoch {epoch+1}: Annealing W_FAIL to {current_w_fail:.4f}, W_RCA to {current_w_rca:.4f}")


        for batch_idx, batch in enumerate(train_loader):
            input_feat = batch["input_features"].to(DEVICE);
            sensor_m = batch["sensor_mask"].to(DEVICE)
            # last_k_std is used for delta target calculation and for RCA novelty calculation
            last_k_std = batch["last_known_values_globally_std"].to(DEVICE);
            delta_tgt_std = batch["pred_delta_targets_globally_std"].to(DEVICE)
            fail_tgt = batch["fail_targets"].to(DEVICE);
            rca_tgt = batch["rca_targets"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=AMP_ENABLED):
                # Pass last_k_std for novelty calculation in RCA head
                pred_abs_std, fail_logits, rca_logits, l_aux, l_entropy = model(input_feat, sensor_m, last_k_std)

                abs_target_std = last_k_std.unsqueeze(-1) + delta_tgt_std
                lp_elements = huber_loss_fn(pred_abs_std, abs_target_std)
                lp_masked = lp_elements * sensor_m.unsqueeze(-1)
                num_active_forecast_elements = (sensor_m.sum() * len(PRED_HORIZONS)).clamp(min=1e-9)
                lp = lp_masked.sum() / num_active_forecast_elements

                lf = focal_loss_mean(fail_logits, fail_tgt)

                lr_elements = focal_loss_elementwise(rca_logits, rca_tgt)
                lr_masked = lr_elements * sensor_m
                lr = lr_masked.sum() / sensor_m.sum().clamp(min=1e-9)

                # Use current (possibly annealed) loss weights
                combined_loss = W_PRED * lp + current_w_fail * lf + current_w_rca * lr + l_aux + l_entropy

            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(
                    f"Warning: NaN/Inf loss. Lp:{lp.item():.3f}, Lf:{lf.item():.3f}, Lr:{lr.item():.3f}, Aux:{l_aux.item():.3f}, Entr:{l_entropy.item():.3f}. Skipping update.")
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
            all_fail_preds_val = [] # For AUROC/PR-AUC
            all_fail_targets_val = [] # For AUROC/PR-AUC

            with torch.no_grad():
                for batch_val in valid_loader:
                    input_feat_val = batch_val["input_features"].to(DEVICE);
                    sensor_m_val = batch_val["sensor_mask"].to(DEVICE)
                    last_k_std_val = batch_val["last_known_values_globally_std"].to(DEVICE);
                    delta_tgt_std_val = batch_val["pred_delta_targets_globally_std"].to(DEVICE)
                    fail_tgt_val = batch_val["fail_targets"].to(DEVICE);
                    rca_tgt_val = batch_val["rca_targets"].to(DEVICE)
                    with autocast(enabled=AMP_ENABLED):
                        pred_abs_std_val, fail_logits_val, rca_logits_val, l_aux_val, l_entr_val = model(
                            input_feat_val, sensor_m_val, last_k_std_val # Pass last_k_std for novelty
                        )

                        abs_target_std_val = last_k_std_val.unsqueeze(-1) + delta_tgt_std_val
                        lp_val_el = huber_loss_fn(pred_abs_std_val, abs_target_std_val)
                        lp_val = (lp_val_el * sensor_m_val.unsqueeze(-1)).sum() / (
                                    sensor_m_val.sum() * len(PRED_HORIZONS)).clamp(min=1e-9)

                        lf_val = focal_loss_mean(fail_logits_val, fail_tgt_val)

                        lr_val_el = focal_loss_elementwise(rca_logits_val, rca_tgt_val)
                        lr_val = (lr_val_el * sensor_m_val).sum() / sensor_m_val.sum().clamp(min=1e-9)

                        # Use current (possibly annealed) loss weights for val loss reporting consistency
                        val_loss = W_PRED * lp_val + current_w_fail * lf_val + current_w_rca * lr_val + l_aux_val + l_entr_val

                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        total_val_loss += val_loss.item();
                        total_val_lp += lp_val.item();
                        total_val_lf += lf_val.item()
                        total_val_lr += lr_val.item();
                        total_val_laux += l_aux_val.item();
                        total_val_lentr += l_entr_val.item()
                        num_val_batches += 1

                        # Suggestion 7: Collect predictions and targets for AUROC/PR-AUC
                        all_fail_preds_val.append(torch.sigmoid(fail_logits_val).cpu().numpy())
                        all_fail_targets_val.append(fail_tgt_val.cpu().numpy())

            print_avg_losses(epoch, "Valid", num_val_batches, total_val_loss, total_val_lp, total_val_lf, total_val_lr,
                             total_val_laux, total_val_lentr)

            # Suggestion 7: Calculate and print AUROC & PR-AUC for failure head
            if num_val_batches > 0 and len(all_fail_preds_val) > 0:
                all_fail_preds_val = np.concatenate(all_fail_preds_val, axis=0)
                all_fail_targets_val = np.concatenate(all_fail_targets_val, axis=0)

                if all_fail_targets_val.ndim == 2 and all_fail_preds_val.ndim == 2 and \
                   all_fail_targets_val.shape == all_fail_preds_val.shape:
                    auroc_scores = []
                    pr_auc_scores = []
                    for i_fh in range(all_fail_targets_val.shape[1]): # Iterate over fail horizons
                        targets_fh = all_fail_targets_val[:, i_fh]
                        preds_fh = all_fail_preds_val[:, i_fh]
                        if len(np.unique(targets_fh)) > 1: # Check if there are both classes present
                            try:
                                auroc = roc_auc_score(targets_fh, preds_fh)
                                pr_auc = average_precision_score(targets_fh, preds_fh)
                                auroc_scores.append(auroc)
                                pr_auc_scores.append(pr_auc)
                            except ValueError as e:
                                print(f"  Skipping metrics for fail horizon {FAIL_HORIZONS[i_fh]}: {e}")
                        else:
                            print(f"  Skipping metrics for fail horizon {FAIL_HORIZONS[i_fh]} (single class in targets)")

                    if auroc_scores: # if any scores were computed
                         print(f"  Avg Valid Fail AUROC: {np.mean(auroc_scores):.4f} (Horizons: {FAIL_HORIZONS})")
                         print(f"  Avg Valid Fail PR-AUC: {np.mean(pr_auc_scores):.4f} (Horizons: {FAIL_HORIZONS})")
                else:
                    print("  Could not compute AUROC/PR-AUC due to shape mismatch or empty arrays.")


    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model: {MODEL_SAVE_PATH}, Preprocessor: {PREPROCESSOR_SAVE_PATH}")


def print_avg_losses(epoch, phase, num_batches, total_loss, lp, lf, lr, laux, lentr):
    if num_batches > 0:
        print(
            f"E{epoch + 1} Avg {phase} L: {total_loss / num_batches:.3f} (P:{lp / num_batches:.3f} F:{lf / num_batches:.3f} R:{lr / num_batches:.3f} Aux:{laux / num_batches:.3f} Entr:{lentr / num_batches:.3f})")
    else:
        print(f"E{epoch + 1} No batches processed for {phase} phase.")

# Suggestion 7: Temperature Scaling (to be applied post-training)
def temperature_scale(logits, temperature):
    """
    Applies temperature scaling to logits.
    Args:
        logits (torch.Tensor): The logits from the model.
        temperature (torch.nn.Parameter or float): The temperature value.
    Returns:
        torch.Tensor: Calibrated probabilities.
    """
    return torch.softmax(logits / temperature, dim=-1)

# Example of how temperature might be found (very simplified, typically needs a validation set)
# def find_temperature(model, valid_loader, device):
#     model.eval()
#     all_logits = []
#     all_targets = []
#     with torch.no_grad():
#         for batch in valid_loader:
#             # ... get inputs, targets ...
#             # _, fail_logits, _, _, _ = model(input_feat.to(device), sensor_m.to(device), last_k.to(device))
#             # all_logits.append(fail_logits)
#             # all_targets.append(fail_tgt) # Assuming fail_tgt are class indices for NLLLoss
#             pass # Placeholder for actual data loading and model inference
#
#     if not all_logits: return torch.tensor(1.0) # Default temperature
#
#     all_logits = torch.cat(all_logits).to(device)
#     all_targets = torch.cat(all_targets).to(device)
#
#     temperature = nn.Parameter(torch.ones(1).to(device))
#     optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50) # Or Adam, etc.
#     nll_criterion = nn.CrossEntropyLoss() # If targets are class indices
#
#     def eval_temp():
#         optimizer.zero_grad()
#         loss = nll_criterion(all_logits / temperature, all_targets)
#         loss.backward()
#         return loss
#
#     optimizer.step(eval_temp)
#     return temperature.item()


if __name__ == '__main__':
    print(
        "--- Script Version: Multi-Task Foundational Model with Implemented Suggestions (v4) ---")
    print(f"IMPORTANT: TRAIN_DIR ('{TRAIN_DIR}') and VALID_DIR ('{VALID_DIR}') must be set correctly.")
    if BASE_DATA_DIR == "../../data/time_series/1": print(
        "\nWARNING: Using default example BASE_DATA_DIR. Paths might be incorrect.\n")
    train_and_save_model()
    print("\nSuggestion 7 Reminder: Temperature scaling for the failure head's $\phi$-values should be performed post-training.")
    print("A 'temperature_scale' function is provided as a utility. You would typically find the optimal")
    print("temperature value on a validation set after the main training is complete and then apply it.")
