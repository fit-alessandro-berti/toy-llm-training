import os
import glob
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Configuration (MUST MATCH LATEST TRAINING SCRIPT) ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1"  # Example, adjust if needed
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # <<< SET THIS (TestData Source)
OUTPUT_CSV_FILENAME = "output_failure_prediction_fm.csv"  # <<< Name of the output CSV file for failure predictions

# Model & Task Parameters (from latest training script, defaults if not in preprocessor)
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]  # Still needed for dataset structure, though not primary output
FAIL_HORIZONS = [3, 5, 10]  # Used by model's fail_head and hysteresis
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0] if FAIL_HORIZONS else 3

# Architectural Params (ensure these match the trained model's config)
SENSOR_INPUT_DIM = 1
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

# MoE Parameters (ensure these match the trained model's config)
NUM_SHARED_EXPERTS = 8
MOE_EXPERT_INPUT_DIM = TRANSFORMER_D_MODEL
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64
MOE_TOP_K = 2

# Testing Params
TEST_BATCH_SIZE = 4  # Adjust as needed for your system's memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCHES_TO_TEST = float('inf')  # Process all data
SAMPLES_PER_BATCH_TO_PRINT = 2  # For console summary

# --- Paths (Updated to match LATEST training script output) ---
MODEL_LOAD_PATH = "foundation_multitask_model_v3_moe_vectorized_ema_cost_sensitive_updated.pth"
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v3_ema_updated.npz"

# Hysteresis thresholds
HYSTERESIS_HI_THRESH = 0.6
HYSTERESIS_LO_THRESH = 0.4


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

    def forward(self, x):
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
                                                   dim_feedforward=embed_dim * 2,
                                                   norm_first=True)
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


# --- Test Dataset ---
class TestMultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizons, fail_horizons, rca_failure_lookahead,
                 model_max_sensors_dim, global_means, global_stds, canonical_sensor_names):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons
        self.fail_horizons = fail_horizons
        self.rca_failure_lookahead = rca_failure_lookahead
        self.model_max_sensors_dim = model_max_sensors_dim
        self.global_means = global_means
        self.global_stds = global_stds
        self.canonical_sensor_names = canonical_sensor_names
        self.num_globally_normed_features = len(canonical_sensor_names)

        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
        self.data_cache = []
        self.window_indices = []
        if not self.file_paths: print(f"ERROR: No CSV files in test data directory: {data_dir}.")
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(f"Loading test data from {self.data_dir}...")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping test file {fp}: {e}")
                continue

            raw_features_from_canonical_cols = np.full((len(df), self.num_globally_normed_features), np.nan,
                                                       dtype=np.float32)
            for i, name in enumerate(self.canonical_sensor_names):
                if name in df.columns: raw_features_from_canonical_cols[:, i] = df[name].values.astype(np.float32)

            if np.all(np.isnan(raw_features_from_canonical_cols)): continue

            features_normalized_globally = np.full_like(raw_features_from_canonical_cols, np.nan)
            for i in range(self.num_globally_normed_features):
                valid_mask = ~np.isnan(raw_features_from_canonical_cols[:, i])
                if self.global_stds[i] > 1e-8:
                    features_normalized_globally[valid_mask, i] = (raw_features_from_canonical_cols[valid_mask, i] -
                                                                   self.global_means[i]) / self.global_stds[i]
                else:
                    features_normalized_globally[valid_mask, i] = 0.0

            failure_flags = np.zeros(len(df), dtype=np.int64)
            if "CURRENT_FAILURE" in df.columns:
                failure_flags = df["CURRENT_FAILURE"].values.astype(np.int64)
            else:
                print(
                    f"Warning: 'CURRENT_FAILURE' column not found in {fp}. Fail/RCA targets will be zero for this file.")

            self.data_cache.append({
                "raw_features_globally_aligned": raw_features_from_canonical_cols,
                "features_normalized_globally": features_normalized_globally,
                "failure_flags": failure_flags,
                "filepath": fp
            })
            max_lookahead = 0
            if self.pred_horizons: max_lookahead = max(max_lookahead, max(self.pred_horizons))
            if self.fail_horizons: max_lookahead = max(max_lookahead, max(self.fail_horizons))
            if self.rca_failure_lookahead: max_lookahead = max(max_lookahead, self.rca_failure_lookahead)

            for i in range(len(df) - self.seq_len - max_lookahead + 1):
                self.window_indices.append((file_idx, i))
        if not self.data_cache: print(f"CRITICAL WARNING: No test data loaded from {self.data_dir}."); return
        print(f"Loaded {len(self.data_cache)} test files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item_data = self.data_cache[file_idx]
        features_normalized_aligned = item_data["features_normalized_globally"]
        flags_full = item_data["failure_flags"]
        filepath = item_data["filepath"]

        input_slice_normed = features_normalized_aligned[window_start_idx: window_start_idx + self.seq_len]
        padded_input_normed = np.zeros((self.seq_len, self.model_max_sensors_dim), dtype=np.float32)
        sensor_mask = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        num_to_copy = min(self.num_globally_normed_features, self.model_max_sensors_dim)

        padded_input_normed[:, :num_to_copy] = input_slice_normed[:, :num_to_copy]
        for k_idx_sensor in range(num_to_copy):
            if not np.all(np.isnan(input_slice_normed[:, k_idx_sensor])):
                sensor_mask[k_idx_sensor] = 1.0
        padded_input_normed[np.isnan(padded_input_normed)] = 0.0
        last_known_normed = padded_input_normed[-1, :].copy()  # Needed for forecast target calculation

        # Forecast targets (still needed for dataset consistency, though not primary output)
        delta_targets_normed = np.zeros((self.model_max_sensors_dim, len(self.pred_horizons)), dtype=np.float32)
        if self.pred_horizons:
            for i_h, h in enumerate(self.pred_horizons):
                target_idx = window_start_idx + self.seq_len + h - 1
                if target_idx < features_normalized_aligned.shape[0]:
                    target_values_all_normed = features_normalized_aligned[target_idx, :]
                    for k_idx_sensor in range(num_to_copy):
                        if sensor_mask[k_idx_sensor] > 0 and not np.isnan(target_values_all_normed[k_idx_sensor]):
                            delta_targets_normed[k_idx_sensor, i_h] = target_values_all_normed[k_idx_sensor] - \
                                                                      last_known_normed[k_idx_sensor]

        # Failure targets (primary interest for this script's CSV output)
        fail_targets = np.zeros(len(self.fail_horizons), dtype=np.float32)
        if self.fail_horizons:
            for i_fh, fh in enumerate(self.fail_horizons):
                start, end = window_start_idx + self.seq_len, window_start_idx + self.seq_len + fh
                if end <= len(flags_full) and np.any(flags_full[start:end]):
                    fail_targets[i_fh] = 1.0

        # RCA targets (still needed for dataset consistency)
        rca_targets = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        if self.rca_failure_lookahead > 0:
            raw_features_aligned_for_rca = item_data["raw_features_globally_aligned"]
            start_r, end_r = window_start_idx + self.seq_len, window_start_idx + self.seq_len + self.rca_failure_lookahead
            if end_r <= len(flags_full) and np.any(flags_full[start_r:end_r]):
                current_window_raw = raw_features_aligned_for_rca[window_start_idx: window_start_idx + self.seq_len, :]
                future_lookahead_raw = raw_features_aligned_for_rca[start_r:end_r, :]
                for k_idx_sensor in range(num_to_copy):
                    if sensor_mask[k_idx_sensor] > 0:
                        sensor_data_current_window_raw = current_window_raw[:, k_idx_sensor]
                        sensor_data_future_lookahead_raw = future_lookahead_raw[:, k_idx_sensor]
                        valid_current_raw = sensor_data_current_window_raw[~np.isnan(sensor_data_current_window_raw)]
                        valid_future_raw = sensor_data_future_lookahead_raw[~np.isnan(sensor_data_future_lookahead_raw)]
                        if len(valid_current_raw) > 0 and len(valid_future_raw) > 0:
                            mean_current_raw = np.mean(valid_current_raw)
                            std_current_raw = np.std(valid_current_raw)
                            std_current_raw = max(std_current_raw, 1e-6)
                            if np.any(np.abs(valid_future_raw - mean_current_raw) > 3 * std_current_raw):
                                rca_targets[k_idx_sensor] = 1.0

        return {"input_features": torch.from_numpy(padded_input_normed),
                "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_globally_std": torch.from_numpy(last_known_normed),  # For forecast task
                "pred_delta_targets_globally_std": torch.from_numpy(delta_targets_normed),  # For forecast task
                "fail_targets": torch.from_numpy(fail_targets),  # Primary GT for this script
                "rca_targets": torch.from_numpy(rca_targets),  # For RCA task
                "filepath": filepath,
                "window_start_idx": window_start_idx
                }


# --- Foundational Multi-Task Model ---
class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, model_max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 num_shared_experts, moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim,
                 pred_horizons_len, fail_horizons_len,  # Note: these are lengths
                 moe_top_k, moe_noise_std, aux_loss_coeff,
                 entropy_reg_coeff):
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
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)  # Outputs fail_horizons_len logits
        self.rca_head = nn.Linear(sensor_tcn_out_dim + transformer_d_model + moe_output_dim, 1)

    def _apply_moe_topk(self, x_expert_input, gate_input, gate_network, experts_modulelist, k, noise_std):
        logits = gate_network(gate_input)
        if self.training and noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std

        num_experts = len(experts_modulelist)
        eff_k = min(k, num_experts)

        topk_val, topk_idx = torch.topk(logits, eff_k, dim=-1)
        topk_w = torch.softmax(topk_val, dim=-1)

        all_out = torch.stack([e(x_expert_input) for e in experts_modulelist], dim=1)

        gather_idx_shape = list(topk_idx.shape) + [all_out.size(-1)]
        gather_idx = topk_idx.unsqueeze(-1).expand(gather_idx_shape)
        sel_out = all_out.gather(1, gather_idx)
        y = (sel_out * topk_w.unsqueeze(-1)).sum(dim=1)
        # Aux loss components (not primary for this script's output, but model includes them)
        router_prob_for_loss = torch.softmax(logits, -1)
        avg_router_prob = router_prob_for_loss.mean(0)
        ones_for_scatter = torch.ones_like(topk_idx, dtype=router_prob_for_loss.dtype).view(-1)
        num_examples_for_expert_frac = gate_input.size(0) * eff_k if gate_input.size(0) > 0 else 1.0
        expert_frac = torch.zeros_like(avg_router_prob, device=avg_router_prob.device).scatter_add_(
            0, topk_idx.view(-1), ones_for_scatter
        ) / num_examples_for_expert_frac
        load_balance_loss = self.num_shared_experts * (avg_router_prob * expert_frac).sum()
        return y, load_balance_loss, logits

    def forward(self, x_features_globally_std, sensor_mask):
        batch_size, seq_len_dim, _ = x_features_globally_std.shape
        x_permuted = x_features_globally_std.permute(0, 2, 1)
        last_val_globally_std = x_permuted[:, :, -1].clone()

        x_expanded_for_tcn = x_permuted.unsqueeze(-1)
        x_reshaped_for_encoder = x_expanded_for_tcn.reshape(
            batch_size * self.model_max_sensors, seq_len_dim, SENSOR_INPUT_DIM
        )
        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, self.model_max_sensors,
                                                                         seq_len_dim, self.sensor_tcn_out_dim)
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

        current_moe_noise = self.moe_noise_std if self.training else 0.0

        moe_forecast_output, aux_f, logits_f = self._apply_moe_topk(
            mean_ctx_global, router_input_global, self.gates["forecast"], self.experts_shared,
            self.moe_top_k, current_moe_noise
        )
        moe_fail_output, aux_fail, logits_fail = self._apply_moe_topk(  # This is key for failure prediction
            mean_ctx_global, router_input_global, self.gates["fail"], self.experts_shared,
            self.moe_top_k, current_moe_noise
        )

        x_flat_rca_expert_input = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        valid_token_mask_rca = sensor_mask.view(-1).bool()
        x_flat_rca_expert_input_valid = x_flat_rca_expert_input[valid_token_mask_rca]

        moe_rca_output_flat = torch.zeros(batch_size * self.model_max_sensors, self.moe_output_dim,
                                          device=x_features_globally_std.device, dtype=moe_forecast_output.dtype)
        aux_rca = torch.tensor(0.0, device=x_features_globally_std.device)
        logits_rca_valid = None
        if x_flat_rca_expert_input_valid.size(0) > 0:
            x_flat_rca_gate_input_valid = x_flat_rca_expert_input_valid
            moe_rca_output_flat_valid, aux_rca_val, logits_rca_valid_val = self._apply_moe_topk(
                x_flat_rca_expert_input_valid, x_flat_rca_gate_input_valid, self.gates["rca"], self.experts_shared,
                self.moe_top_k, current_moe_noise
            )
            moe_rca_output_flat[valid_token_mask_rca] = moe_rca_output_flat_valid
            aux_rca = aux_rca_val
            # logits_rca_valid = logits_rca_valid_val # Not directly used for CSV

        # --- Prediction Heads ---
        # Forecast head
        tcn_features_last_step = sensor_temporal_features[:, :, -1, :]
        moe_f_expanded = moe_forecast_output.unsqueeze(1).expand(-1, self.model_max_sensors, -1)
        pred_head_input_features = torch.cat([tcn_features_last_step, cross_sensor_context_masked, moe_f_expanded],
                                             dim=-1)
        pred_delta_globally_std = self.pred_head(pred_head_input_features)
        pred_abs_globally_std = last_val_globally_std.unsqueeze(-1) + pred_delta_globally_std
        pred_abs_globally_std = pred_abs_globally_std * sensor_mask.unsqueeze(-1)

        # Failure head (primary interest for this script's CSV)
        failure_logits = self.fail_head(moe_fail_output)  # [B, FailHorizonsLen]

        # RCA head
        tcn_flat = tcn_features_last_step.reshape(-1, self.sensor_tcn_out_dim)
        ctx_flat = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        rca_head_input_flat = torch.cat([tcn_flat, ctx_flat, moe_rca_output_flat], dim=-1)
        rca_logits_flat = self.rca_head(rca_head_input_flat).squeeze(-1)
        rca_logits = rca_logits_flat.view(batch_size, self.model_max_sensors)

        # Aux and Entropy losses are not primary for eval output but are part of model's structure
        total_aux_loss = self.aux_loss_coeff * (aux_f + aux_fail + aux_rca)
        total_entropy_loss = torch.tensor(0.0, device=x_features_globally_std.device)
        # ... (entropy calculation if needed, omitted for brevity as not used in eval output)

        return pred_abs_globally_std, failure_logits, rca_logits, total_aux_loss, total_entropy_loss


# --- Test Script Logic ---
def test_model_failure_prediction():
    print(f"--- Test Script for Failure Prediction Output ---")
    print(f"Using device: {DEVICE}")
    print(f"Attempting to load model from: {MODEL_LOAD_PATH}")
    print(f"Attempting to load preprocessor from: {PREPROCESSOR_LOAD_PATH}")

    output_failure_data = []  # MODIFIED: Initialize list for failure prediction CSV

    # 1. Load Preprocessor Config
    try:
        preprocessor_data = np.load(PREPROCESSOR_LOAD_PATH, allow_pickle=True)
        global_means = preprocessor_data['global_means']  # Needed for dataset
        global_stds = preprocessor_data['global_stds']  # Needed for dataset
        canonical_sensor_names = list(preprocessor_data['canonical_sensor_names'])  # Needed for dataset
        model_max_sensors_dim = int(preprocessor_data['model_max_sensors_dim'])

        loaded_seq_len = int(preprocessor_data.get('seq_len', SEQ_LEN))
        loaded_pred_horizons_np = preprocessor_data.get('pred_horizons', np.array(PRED_HORIZONS))
        loaded_pred_horizons = list(loaded_pred_horizons_np) if isinstance(loaded_pred_horizons_np,
                                                                           np.ndarray) else PRED_HORIZONS

        loaded_fail_horizons_np = preprocessor_data.get('fail_horizons', np.array(FAIL_HORIZONS))
        loaded_fail_horizons = list(loaded_fail_horizons_np) if isinstance(loaded_fail_horizons_np,
                                                                           np.ndarray) else FAIL_HORIZONS  # Key for this script

        default_rca_lookahead = loaded_fail_horizons[0] if loaded_fail_horizons else 3
        loaded_rca_lookahead = int(preprocessor_data.get('rca_failure_lookahead', default_rca_lookahead))

        print(
            f"Preprocessor loaded. Using SeqLen={loaded_seq_len}, PredHorizons={loaded_pred_horizons}, FailHorizons={loaded_fail_horizons}, RCALookahead={loaded_rca_lookahead}")
        if not loaded_fail_horizons:
            print(
                "ERROR: No failure horizons defined (loaded_fail_horizons is empty). Cannot proceed with failure prediction output.")
            return

    except FileNotFoundError:
        print(f"ERROR: Preprocessor file not found at {PREPROCESSOR_LOAD_PATH}. Exiting.")
        return
    except KeyError as e:
        print(f"ERROR: Missing key {e} in preprocessor file. Exiting.")
        return

    # 2. Initialize Dataset and DataLoader
    if not (os.path.exists(VALID_DIR) and os.path.isdir(VALID_DIR)):
        print(f"ERROR: Test data directory (VALID_DIR) '{VALID_DIR}' not found. Exiting.")
        return

    test_dataset = TestMultivariateTimeSeriesDataset(
        data_dir=VALID_DIR,
        seq_len=loaded_seq_len,
        pred_horizons=loaded_pred_horizons,  # Pass all horizons for consistent dataset item structure
        fail_horizons=loaded_fail_horizons,
        rca_failure_lookahead=loaded_rca_lookahead,
        model_max_sensors_dim=model_max_sensors_dim,
        global_means=global_means,
        global_stds=global_stds,
        canonical_sensor_names=canonical_sensor_names
    )
    if len(test_dataset) == 0:
        print(f"ERROR: No data found or no valid windows created in the test dataset from {VALID_DIR}. Exiting.")
        return

    # Batch processing setup (process all batches)
    # shuffle=False can make debugging easier if order matters, but CSV contains identifiers.
    # The original script collected all then shuffled, which is fine if processing all.
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Total batches to process: {len(test_loader)}")
    if len(test_loader) == 0:
        print("No batches generated by DataLoader. Exiting.")
        return

    # 3. Initialize Model
    model = FoundationalTimeSeriesModel(
        model_max_sensors=model_max_sensors_dim,
        seq_len=loaded_seq_len,
        sensor_input_dim=SENSOR_INPUT_DIM,
        sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS,
        tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL,
        transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        num_shared_experts=NUM_SHARED_EXPERTS,
        moe_expert_input_dim=MOE_EXPERT_INPUT_DIM,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT,
        moe_output_dim=MOE_OUTPUT_DIM,
        pred_horizons_len=len(loaded_pred_horizons),
        fail_horizons_len=len(loaded_fail_horizons),  # Critical for failure head
        moe_top_k=MOE_TOP_K,
        moe_noise_std=0.0,
        aux_loss_coeff=0.0,
        entropy_reg_coeff=0.0
    ).to(DEVICE)

    # 4. Load Trained Model Weights
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_LOAD_PATH}. Exiting.")
        return
    except RuntimeError as e:
        print(f"ERROR loading model state_dict: {e}. Check architecture match. Exiting.")
        return

    model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    # 5. Perform Inference and Collect Failure Prediction Results
    with torch.no_grad():
        for batch_idx, batch_content in enumerate(test_loader):
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:  # Print progress
                print(f"\n--- Processing Batch {batch_idx + 1}/{len(test_loader)} ---")

            input_globally_std = batch_content["input_features"].to(DEVICE)
            sensor_m = batch_content["sensor_mask"].to(DEVICE)
            # Ground truths
            gt_fail_targets_batch = batch_content["fail_targets"].to(DEVICE)  # [B, NumFailHorizons]

            # Model predictions (forecast, failure, rca)
            # We are interested in `failure_logits_batch`
            _, failure_logits_batch, rca_logits_batch, _, _ = model(input_globally_std, sensor_m)
            # failure_logits_batch should be [B, NumFailHorizons]

            # Iterate through each sample in the batch
            for i in range(input_globally_std.size(0)):
                sample_fp = batch_content["filepath"][i]
                sample_win_start = batch_content["window_start_idx"][i].item()

                gt_fail_sample_i = gt_fail_targets_batch[i]  # [NumFailHorizons]
                pred_fail_logits_sample_i = failure_logits_batch[i]  # [NumFailHorizons]
                pred_fail_probs_sample_i = torch.sigmoid(pred_fail_logits_sample_i)  # [NumFailHorizons]

                # Apply hysteresis for each horizon for this sample
                current_hysteresis_state = torch.zeros(1, device=DEVICE)  # Reset for each sample

                # Iterate through each defined failure horizon
                for h_idx, horizon_step_val in enumerate(loaded_fail_horizons):
                    gt_failure_status = gt_fail_sample_i[h_idx].item()
                    pred_logit = pred_fail_logits_sample_i[h_idx].item()
                    pred_prob = pred_fail_probs_sample_i[h_idx].item()

                    # Hysteresis calculation for this specific horizon h_idx
                    # The state evolves across horizons FOR A SINGLE SAMPLE if that's the design.
                    # Or, if hysteresis is independent per horizon, reset state.
                    # The original script implies state evolves *across the horizons for a given sample's fail prediction output vector*.
                    # Let's keep that logic.
                    # The state for hysteresis should be per-sample, evolving over its fail_horizons vector.
                    # So, current_hysteresis_state is reset per sample, then evolves.
                    # If hysteresis is calculated independently for each horizon (less common interpretation):
                    #   temp_state = 0; if pred_prob > HI: temp_state = 1; else if pred_prob < LO: temp_state = 0;
                    # Original interpretation:
                    is_state_zero = (
                                current_hysteresis_state == 0)  # current_hysteresis_state is for this sample, up to previous horizon
                    becomes_one = is_state_zero & (
                                pred_fail_probs_sample_i[h_idx] > HYSTERESIS_HI_THRESH)  # Use prob for current horizon
                    remains_one = (~is_state_zero) & (pred_fail_probs_sample_i[h_idx] > HYSTERESIS_LO_THRESH)
                    current_hysteresis_state = (becomes_one | remains_one).float()
                    hysteresis_alarm = current_hysteresis_state.item()

                    output_failure_data.append({
                        'filepath': os.path.basename(sample_fp),
                        'window_start_idx': sample_win_start,
                        'failure_horizon_steps': horizon_step_val,
                        'ground_truth_failure': gt_failure_status,
                        'predicted_failure_logit': pred_logit,
                        'predicted_failure_probability': pred_prob,
                        'predicted_hysteresis_alarm': hysteresis_alarm
                    })

                # --- Console printing for a limited number of SAMPLES (not batches) ---
                # This uses SAMPLES_PER_BATCH_TO_PRINT as an overall limit of samples to print.
                num_samples_printed_so_far = batch_idx * TEST_BATCH_SIZE + i
                if num_samples_printed_so_far < SAMPLES_PER_BATCH_TO_PRINT:
                    print(
                        f"\n  Console Summary for Sample (Overall #{num_samples_printed_so_far + 1}) File: {os.path.basename(sample_fp)}, Window: {sample_win_start}")
                    print("    Failure Predictions (GT | Prob | Hyst_Alarm):")
                    # Re-calculate hysteresis for printing here to ensure it matches what was stored
                    # Or retrieve from what would be stored (but we store per horizon)
                    # For console, let's re-do the hysteresis display loop:
                    temp_hyst_state_console = torch.zeros(1, device=DEVICE)
                    for h_idx_print, fh_print in enumerate(loaded_fail_horizons):
                        prob_print = pred_fail_probs_sample_i[h_idx_print].item()
                        gt_print = gt_fail_sample_i[h_idx_print].item()

                        is_state_zero_print = (temp_hyst_state_console == 0)
                        becomes_one_print = is_state_zero_print & (
                                    pred_fail_probs_sample_i[h_idx_print] > HYSTERESIS_HI_THRESH)
                        remains_one_print = (~is_state_zero_print) & (
                                    pred_fail_probs_sample_i[h_idx_print] > HYSTERESIS_LO_THRESH)
                        temp_hyst_state_console = (becomes_one_print | remains_one_print).float()
                        alarm_print = temp_hyst_state_console.item()

                        print(
                            f"      H={fh_print}: {gt_print:.0f} | {prob_print:.3f} | {alarm_print:.0f} (Logit: {pred_fail_logits_sample_i[h_idx_print].item():.3f})")

                    # Optional: print a few forecast/RCA for these limited samples if desired for quick check
                    # This part is just for quick console view, not for the primary CSV output.
                    # For brevity, keeping console focused on failure for this script.

    # Write collected failure prediction data to CSV
    if output_failure_data:
        output_df = pd.DataFrame(output_failure_data)
        try:
            output_df.to_csv(OUTPUT_CSV_FILENAME, index=False)
            print(f"\nSuccessfully wrote {len(output_df)} failure prediction data points to {OUTPUT_CSV_FILENAME}")
        except Exception as e:
            print(f"\nERROR: Could not write failure prediction outputs to CSV: {e}")
    else:
        print("\nNo failure prediction data collected to write to CSV.")

    print("\n--- Failure Prediction Test Script Finished ---")


if __name__ == '__main__':
    # Basic path checks
    if BASE_DATA_DIR == "../../data/time_series/1":
        default_validation_path = os.path.join("../../data/time_series/1", "VALIDATION")
        if VALID_DIR == default_validation_path:
            print(f"\nWARNING: VALID_DIR is set to the default example path: {VALID_DIR}")
            print("Ensure this points to your actual test data folder if this is not intended.\n")

    if not os.path.exists(MODEL_LOAD_PATH) or not os.path.exists(PREPROCESSOR_LOAD_PATH):
        print(
            f"CRITICAL WARNING: Model ({MODEL_LOAD_PATH}) or Preprocessor ({PREPROCESSOR_LOAD_PATH}) not found. Please check paths. Script will likely fail.\n")

    if not os.path.exists(VALID_DIR) or not os.listdir(VALID_DIR):  # Check if VALID_DIR is empty
        print(f"CRITICAL WARNING: VALID_DIR '{VALID_DIR}' does not exist or is empty. No data to process.\n")
        # Exit if no data, as script will fail in DataLoader or Dataset init.
        exit()

    test_model_failure_prediction()
