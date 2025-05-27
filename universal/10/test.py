import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Configuration (MUST MATCH LATEST TRAINING SCRIPT v5_survival) ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1"  # Example, adjust if needed
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # <<< SET THIS (TestData Source)

# Model & Task Parameters (from latest training script)
SEQ_LEN = 64  # Will be overridden by preprocessor if different
PRED_HORIZONS = [1, 3, 5]  # Will be overridden by preprocessor if different
# FAIL_HORIZONS, RCA_FAILURE_LOOKAHEAD, HAZARD_HEAD_MAX_HORIZON will be loaded from preprocessor
# MAX_SENSORS_CAP will be loaded from preprocessor as model_max_sensors_dim

# Architectural Params (from latest training script)
SENSOR_INPUT_DIM = 1
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

# MoE Parameters (from latest training script)
NUM_SHARED_EXPERTS = 8
MOE_EXPERT_INPUT_DIM = TRANSFORMER_D_MODEL
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64
MOE_TOP_K = 2
MOE_NOISE_STD = 0.0  # For eval, noise is off. Training script used 0.3
AUX_LOSS_COEFF = 0.01
ENTROPY_REG_COEFF = 0.001

# Survival Head Parameters (RCA_SURVIVAL_THRESHOLD is for training loss gating, not directly used in test script prints)
# HAZARD_HEAD_MAX_HORIZON will be loaded.

# Testing Params
TEST_BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCHES_TO_TEST = 5
SAMPLES_PER_BATCH_TO_PRINT = 2

# Adjusted paths to match latest training script output (v5_survival)
MODEL_LOAD_PATH = "foundation_multitask_model_v5_survival.pth"
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v5_survival.npz"


# --- Helper: Positional Encoding (from latest training script) ---
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


# --- TCN Residual Block (from latest training script) ---
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


# --- Per-Sensor TCN Encoder (from latest training script - with CLS Token) ---
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


# --- Inter-Sensor Transformer (from latest training script) ---
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


# --- MoE Components (from latest training script) ---
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


# --- Discrete-Time Survival (Hazard) Head (from latest training script) ---
class HazardHead(nn.Module):
    def __init__(self, in_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.fc = nn.Linear(in_dim, horizon)

    def forward(self, z):
        return torch.sigmoid(self.fc(z))


# --- Test Dataset (Adapted from latest training dataset) ---
class TestMultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizons,  # pred_horizons is fixed
                 eval_fail_horizons,  # These are the original FAIL_HORIZONS from training for comparison
                 rca_failure_lookahead, model_max_sensors_dim,
                 global_means, global_stds, canonical_sensor_names,
                 hazard_max_horizon):  # Added for completeness, though not directly used for target gen here
        self.data_dir = data_dir;
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons
        self.eval_fail_horizons = eval_fail_horizons  # For generating comparable binary fail targets
        self.rca_failure_lookahead = rca_failure_lookahead
        self.model_max_sensors_dim = model_max_sensors_dim
        self.global_means = global_means;
        self.global_stds = global_stds
        self.canonical_sensor_names = canonical_sensor_names
        self.num_globally_normed_features = len(canonical_sensor_names)
        self.hazard_max_horizon = hazard_max_horizon  # Stored

        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"));
        self.data_cache = [];
        self.window_indices = []
        if not self.file_paths: print(f"ERROR: No CSV files in test data directory: {data_dir}.")
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(
            f"Loading test data from {self.data_dir} using {self.num_globally_normed_features} global features. Model dim: {self.model_max_sensors_dim}")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping test file {fp}: {e}"); continue
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

            failure_flags = np.zeros(len(df), dtype=np.int64)
            if "CURRENT_FAILURE" in df.columns:
                failure_flags = df["CURRENT_FAILURE"].values.astype(np.int64)
            else:
                print(f"Warning: 'CURRENT_FAILURE' not in {fp}. Fail/RCA targets will be zero for this file.")

            self.data_cache.append({"raw_features_globally_aligned": raw_features_from_canonical_cols,
                                    "features_normalized_globally": features_normalized_globally,
                                    "failure_flags": failure_flags, "filepath": fp})

            max_pred_h = max(self.pred_horizons) if self.pred_horizons and len(self.pred_horizons) > 0 else 0
            # Use eval_fail_horizons for max lookahead calculation, as hazard_max_horizon might be larger
            # and eval_fail_horizons define the binary targets we generate here for comparison.
            max_eval_fail_h = max(self.eval_fail_horizons) if self.eval_fail_horizons and len(
                self.eval_fail_horizons) > 0 else 0
            # Also consider hazard_max_horizon for windowing if it's the absolute max for any label generation
            max_lookahead = max(max_pred_h, max_eval_fail_h, self.rca_failure_lookahead, self.hazard_max_horizon)

            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        if not self.data_cache: print(f"CRITICAL WARNING: No test data loaded from {self.data_dir}."); return
        print(f"Loaded {len(self.data_cache)} test files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item_data = self.data_cache[file_idx]
        raw_features_aligned = item_data["raw_features_globally_aligned"]
        features_normalized_aligned = item_data["features_normalized_globally"]
        flags_full = item_data["failure_flags"];
        filepath = item_data["filepath"]
        input_slice_normed = features_normalized_aligned[window_start_idx: window_start_idx + self.seq_len]
        padded_input_normed = np.zeros((self.seq_len, self.model_max_sensors_dim), dtype=np.float32)
        sensor_mask = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        num_to_copy = min(self.num_globally_normed_features, self.model_max_sensors_dim)
        padded_input_normed[:, :num_to_copy] = input_slice_normed[:, :num_to_copy]
        for k_idx in range(num_to_copy):
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

        # Binary failure targets for specified evaluation horizons
        eval_fail_targets_binary = np.zeros(len(self.eval_fail_horizons), dtype=np.float32)
        for i_fh, fh in enumerate(self.eval_fail_horizons):
            start, end = window_start_idx + self.seq_len, window_start_idx + self.seq_len + fh
            if end <= len(flags_full) and np.any(flags_full[start:end]): eval_fail_targets_binary[i_fh] = 1.0

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
                        std_current_raw = max(np.std(valid_current_raw), 1e-6)
                        if np.any(np.abs(valid_future_raw - mean_current_raw) > 3 * std_current_raw): rca_targets[
                            k_idx] = 1.0

        return {"input_features": torch.from_numpy(padded_input_normed), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_globally_std": torch.from_numpy(last_known_normed),
                "pred_delta_targets_globally_std": torch.from_numpy(delta_targets_normed),
                "eval_fail_targets_binary": torch.from_numpy(eval_fail_targets_binary),  # For comparison prints
                "rca_targets": torch.from_numpy(rca_targets),
                "filepath": filepath, "window_start_idx": window_start_idx}


# --- Foundational Multi-Task Model (from latest training script v5_survival) ---
class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, model_max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 num_shared_experts, moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim,
                 pred_horizons_len, hazard_max_horizon,  # Changed from fail_horizons_len
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
        self.aux_loss_coeff = aux_loss_coeff;
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
        self.hazard_head = HazardHead(moe_output_dim, self.hazard_max_horizon)  # Updated
        rca_head_input_dim = sensor_tcn_out_dim + transformer_d_model + moe_output_dim + 1
        self.rca_head = nn.Linear(rca_head_input_dim, 1)

    def _apply_moe_topk(self, x_expert_input, gate_input, gate_network, experts_modulelist, k, noise_std):
        logits = gate_network(gate_input)
        # In eval mode (self.training is False), noise_std passed should be 0.0
        if self.training and noise_std > 0: logits = logits + torch.randn_like(logits) * noise_std

        num_experts = len(experts_modulelist);
        eff_k = min(k, num_experts)
        topk_val, topk_idx = torch.topk(logits, eff_k, dim=-1);
        topk_w = torch.softmax(topk_val, dim=-1)
        all_out = torch.stack([e(x_expert_input) for e in experts_modulelist], dim=1)
        expert_output_dim = all_out.size(-1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, eff_k, expert_output_dim)
        sel_out = all_out.gather(1, gather_idx);
        y = (sel_out * topk_w.unsqueeze(-1)).sum(dim=1)
        router_prob_for_loss = torch.softmax(logits, -1);
        avg_router_prob = router_prob_for_loss.mean(0)
        ones_for_scatter = torch.ones_like(topk_idx, dtype=router_prob_for_loss.dtype).reshape(-1)
        expert_frac = torch.zeros_like(avg_router_prob).scatter_add_(
            0, topk_idx.reshape(-1), ones_for_scatter
        ) / (x_expert_input.size(0) * eff_k if x_expert_input.size(0) > 0 else 1.0)
        load_balance_loss = self.num_shared_experts * (avg_router_prob * expert_frac).sum()
        return y, load_balance_loss, logits

    def forward(self, x_features_globally_std, sensor_mask, last_known_values_globally_std_for_novelty):
        batch_size, seq_len_data, _ = x_features_globally_std.shape
        x_input_for_tcn = x_features_globally_std.reshape(batch_size * self.model_max_sensors, seq_len_data,
                                                          SENSOR_INPUT_DIM)
        sensor_temporal_features_flat, h_cls_flat = self.per_sensor_encoder(x_input_for_tcn)
        sensor_temporal_features_main = sensor_temporal_features_flat.reshape(
            batch_size, self.model_max_sensors, seq_len_data, self.sensor_tcn_out_dim)
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

        # For eval mode, self.training is False, so moe_noise_std is effectively 0 if passed directly
        # or explicitly pass 0.0 for noise_std
        moe_noise_effective = 0.0 if not self.training else self.moe_noise_std

        moe_forecast_output, aux_f, logits_f = self._apply_moe_topk(
            mean_ctx_global, router_input_global_forecast_fail, self.gates["forecast"], self.experts_shared,
            self.moe_top_k, moe_noise_effective)
        moe_fail_output, aux_fail, logits_fail = self._apply_moe_topk(
            mean_ctx_global, router_input_global_forecast_fail, self.gates["fail"], self.experts_shared,
            self.moe_top_k, moe_noise_effective)
        x_flat_rca_expert_input = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        valid_token_mask_rca = sensor_mask.view(-1).bool()
        x_flat_rca_expert_input_valid = x_flat_rca_expert_input[valid_token_mask_rca]
        moe_rca_output_flat_valid = torch.empty(0, self.moe_output_dim, device=x_features_globally_std.device,
                                                dtype=moe_forecast_output.dtype)
        aux_rca = torch.tensor(0.0, device=x_features_globally_std.device);
        logits_rca_valid = None
        if x_flat_rca_expert_input_valid.size(0) > 0:
            x_flat_rca_gate_input_valid = x_flat_rca_expert_input_valid
            moe_rca_output_flat_valid, aux_rca, logits_rca_valid = self._apply_moe_topk(
                x_flat_rca_expert_input_valid, x_flat_rca_gate_input_valid, self.gates["rca"], self.experts_shared,
                self.moe_top_k, moe_noise_effective)
        moe_rca_output_flat = torch.zeros(batch_size * self.model_max_sensors, self.moe_output_dim,
                                          device=x_features_globally_std.device, dtype=moe_forecast_output.dtype)
        if x_flat_rca_expert_input_valid.size(0) > 0: moe_rca_output_flat[
            valid_token_mask_rca] = moe_rca_output_flat_valid

        total_aux_loss = self.aux_loss_coeff * (aux_f + aux_fail + aux_rca)
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

        hazard_rates = self.hazard_head(moe_fail_output)  # [B, HazardMaxHorizon]
        log_1_minus_h = torch.log(1.0 - hazard_rates + 1e-8)
        cumulative_log_survival = torch.cumsum(log_1_minus_h, dim=1)
        cumulative_survival_probs = torch.exp(cumulative_log_survival)  # S_t values [B, HazardMaxHorizon]

        with torch.no_grad():
            pred_next_h1 = pred_abs_globally_std[:, :, 0]
        delta_novelty = torch.abs(pred_next_h1 - last_known_values_globally_std_for_novelty)
        delta_novelty_masked = delta_novelty * sensor_mask;
        delta_novelty_flat = delta_novelty_masked.reshape(-1, 1)
        tcn_flat = tcn_features_last_step.reshape(-1, self.sensor_tcn_out_dim)
        ctx_flat = cross_sensor_context_masked.reshape(-1, self.transformer_d_model)
        rca_head_input_flat = torch.cat([tcn_flat, ctx_flat, moe_rca_output_flat, delta_novelty_flat], dim=-1)
        rca_logits_flat = self.rca_head(rca_head_input_flat).squeeze(-1)
        rca_logits = rca_logits_flat.view(batch_size, self.model_max_sensors)

        return pred_abs_globally_std, hazard_rates, cumulative_survival_probs, rca_logits, total_aux_loss, total_entropy_loss


# --- Test Script Logic ---
def test_model():
    print(f"--- Test Script for Foundational Multi-Task MoE Model (Version 5 Survival) ---")
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_LOAD_PATH}")
    print(f"Loading preprocessor from: {PREPROCESSOR_LOAD_PATH}")

    # 1. Load Preprocessor Config
    try:
        preprocessor_data = np.load(PREPROCESSOR_LOAD_PATH, allow_pickle=True)
        global_means = preprocessor_data['global_means']
        global_stds = preprocessor_data['global_stds']
        canonical_sensor_names = list(preprocessor_data['canonical_sensor_names'])
        model_max_sensors_dim = int(preprocessor_data['model_max_sensors_dim'])
        loaded_seq_len = int(preprocessor_data['seq_len'])
        loaded_pred_horizons = list(preprocessor_data['pred_horizons'])
        # fail_horizons from training are the eval_fail_horizons for the test script
        loaded_eval_fail_horizons = list(preprocessor_data.get('fail_horizons', [3, 5, 10]))
        loaded_rca_lookahead = int(preprocessor_data.get('rca_failure_lookahead', loaded_eval_fail_horizons[
            0] if loaded_eval_fail_horizons else 3))
        # Load hazard_max_horizon (new in v5)
        loaded_hazard_max_horizon = int(preprocessor_data.get('hazard_max_horizon',
                                                              max(loaded_eval_fail_horizons) if loaded_eval_fail_horizons else 10))

        # Override script constants with loaded values for consistency
        script_seq_len = loaded_seq_len
        script_pred_horizons = loaded_pred_horizons
        if list(PRED_HORIZONS) != script_pred_horizons: print(
            f"Warning: PRED_HORIZONS mismatch. Loaded: {script_pred_horizons}, Script: {PRED_HORIZONS}. Using loaded.")
        if SEQ_LEN != script_seq_len: print(
            f"Warning: SEQ_LEN mismatch. Loaded: {script_seq_len}, Script: {SEQ_LEN}. Using loaded.")

    except FileNotFoundError:
        print(f"ERROR: Preprocessor file not found at {PREPROCESSOR_LOAD_PATH}. Exiting."); return
    except KeyError as e:
        print(
            f"ERROR: Missing key {e} in preprocessor file '{PREPROCESSOR_LOAD_PATH}'. Ensure it's from the v5 training. Exiting."); return

    print(
        f"Preprocessor loaded: model_max_sensors_dim={model_max_sensors_dim}, {len(canonical_sensor_names)} canonical sensors.")
    print(
        f"Using Eval Fail Horizons: {loaded_eval_fail_horizons}, RCA Lookahead: {loaded_rca_lookahead}, Hazard Max Horizon: {loaded_hazard_max_horizon}")

    # 2. Initialize Dataset and DataLoader
    if not (os.path.exists(VALID_DIR) and os.path.isdir(VALID_DIR)):
        print(f"ERROR: Test data directory '{VALID_DIR}' missing or not a directory.");
        return

    test_dataset = TestMultivariateTimeSeriesDataset(
        data_dir=VALID_DIR, seq_len=script_seq_len, pred_horizons=script_pred_horizons,
        eval_fail_horizons=loaded_eval_fail_horizons,  # For GT comparison
        rca_failure_lookahead=loaded_rca_lookahead,
        model_max_sensors_dim=model_max_sensors_dim, global_means=global_means,
        global_stds=global_stds, canonical_sensor_names=canonical_sensor_names,
        hazard_max_horizon=loaded_hazard_max_horizon
    )
    if len(test_dataset) == 0: print(f"No data in {VALID_DIR} or no valid windows. Exiting."); return
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Initialize Model
    model = FoundationalTimeSeriesModel(
        model_max_sensors=model_max_sensors_dim, seq_len=script_seq_len,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM, tcn_levels=TCN_LEVELS,
        tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        num_shared_experts=NUM_SHARED_EXPERTS, moe_expert_input_dim=MOE_EXPERT_INPUT_DIM,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        pred_horizons_len=len(script_pred_horizons),
        hazard_max_horizon=loaded_hazard_max_horizon,  # Updated
        moe_top_k=MOE_TOP_K, moe_noise_std=MOE_NOISE_STD,  # Will be 0.0 in eval
        aux_loss_coeff=AUX_LOSS_COEFF, entropy_reg_coeff=ENTROPY_REG_COEFF
    ).to(DEVICE)

    # 4. Load Trained Model Weights
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {MODEL_LOAD_PATH}. Exiting."); return
    except RuntimeError as e:
        print(f"ERROR loading model state: {e}. Ensure architecture matches. Exiting."); return
    except Exception as e:
        print(f"ERROR loading model state (general): {e}. Exiting."); return
    model.eval();
    print("Model loaded and in evaluation mode.")

    # 5. Perform Inference and Print Results
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_idx >= MAX_BATCHES_TO_TEST: print(
                f"\nReached MAX_BATCHES_TO_TEST ({MAX_BATCHES_TO_TEST}). Stopping."); break
            print(f"\n--- Batch {batch_idx + 1}/{min(MAX_BATCHES_TO_TEST, len(test_loader))} ---")

            input_globally_std = batch_data["input_features"].to(DEVICE)
            sensor_m = batch_data["sensor_mask"].to(DEVICE)
            last_k_std_gt = batch_data["last_known_values_globally_std"].to(DEVICE)
            delta_tgt_std_gt = batch_data["pred_delta_targets_globally_std"].to(DEVICE)
            eval_fail_tgt_binary_gt = batch_data["eval_fail_targets_binary"].to(DEVICE)  # Binary GT
            rca_tgt_gt = batch_data["rca_targets"].to(DEVICE)

            pred_abs_globally_std, hazard_rates, cumulative_survival_probs, rca_logits, _, _ = model(
                input_globally_std, sensor_m, last_k_std_gt)

            actual_abs_targets_std = last_k_std_gt.unsqueeze(-1) + delta_tgt_std_gt

            num_samples_to_print = min(SAMPLES_PER_BATCH_TO_PRINT, input_globally_std.size(0))
            for i in range(num_samples_to_print):
                sample_fp = batch_data["filepath"][i];
                sample_win_start = batch_data["window_start_idx"][i].item()
                print(f"\n  Sample {i + 1} (File: {os.path.basename(sample_fp)}, WinStart: {sample_win_start})")
                active_sensor_indices = torch.where(sensor_m[i] == 1.0)[0].cpu().tolist()
                if not active_sensor_indices: print("    No active sensors in this sample."); continue

                print("    Forecast (Globally Standardized Space):")
                for h_idx, horizon_val in enumerate(script_pred_horizons):
                    print(f"      H={horizon_val}:")
                    for s_local_idx, s_global_model_idx in enumerate(
                            active_sensor_indices[:min(3, len(active_sensor_indices))]):
                        gt_val = actual_abs_targets_std[i, s_global_model_idx, h_idx].item()
                        pred_val = pred_abs_globally_std[i, s_global_model_idx, h_idx].item()
                        s_name = canonical_sensor_names[s_global_model_idx] if s_global_model_idx < len(
                            canonical_sensor_names) else f"Sensor Pad {s_global_model_idx + 1}"
                        print(
                            f"        {s_name} (idx {s_global_model_idx}): GT Abs_Std={gt_val:.3f}, Pred Abs_Std={pred_val:.3f}")

                print("    Failure Prediction (Prob of Fail by Horizon = 1 - S_t):")
                # cumulative_survival_probs is [B, HazardMaxHorizon]
                # loaded_eval_fail_horizons contains the horizons for which we have binary GT
                gt_fail_status_sample = eval_fail_tgt_binary_gt[i].cpu().tolist()

                for h_idx, horizon_val in enumerate(loaded_eval_fail_horizons):
                    if horizon_val <= loaded_hazard_max_horizon:  # Ensure horizon is within prediction range
                        # S_k is survival up to k. So S_k = cumulative_survival_probs[:, k-1] (0-indexed)
                        prob_survival_at_h = cumulative_survival_probs[i, horizon_val - 1].item()
                        prob_fail_by_h = 1.0 - prob_survival_at_h
                        print(
                            f"      Next {horizon_val} steps: GT Fail={gt_fail_status_sample[h_idx]:.0f}, Pred P(Fail)={prob_fail_by_h:.3f} (S_{horizon_val}={prob_survival_at_h:.3f})")
                    else:
                        print(
                            f"      Next {horizon_val} steps: GT Fail={gt_fail_status_sample[h_idx]:.0f}, Pred P(Fail)=N/A (horizon > max hazard pred range)")

                print("    RCA Prediction (Sigmoid Scores):")
                rca_eval_condition = False
                prob_fail_for_rca_horizon = 0.0
                try:
                    # Check failure prob at RCA_FAILURE_LOOKAHEAD
                    if loaded_rca_lookahead <= loaded_hazard_max_horizon:
                        prob_survival_at_rca_lh = cumulative_survival_probs[i, loaded_rca_lookahead - 1].item()
                        prob_fail_for_rca_horizon = 1.0 - prob_survival_at_rca_lh
                        # Find corresponding GT for this lookahead
                        gt_fail_at_rca_lh_idx = loaded_eval_fail_horizons.index(loaded_rca_lookahead)
                        gt_fail_status_rca_lh = gt_fail_status_sample[gt_fail_at_rca_lh_idx]
                        rca_eval_condition = prob_fail_for_rca_horizon > 0.5 or gt_fail_status_rca_lh == 1.0
                    else:
                        print(
                            f"      Warning: RCA_FAILURE_LOOKAHEAD ({loaded_rca_lookahead}) > HAZARD_MAX_HORIZON ({loaded_hazard_max_horizon}). Cannot reliably get P(Fail) for RCA condition.")
                        rca_eval_condition = True  # Default to show
                except (ValueError, IndexError) as e:
                    print(
                        f"      Warning: Could not map RCA_FAILURE_LOOKAHEAD ({loaded_rca_lookahead}) to eval_fail_horizons or hazard predictions. Error: {e}. Showing RCA by default.")
                    rca_eval_condition = True

                if rca_eval_condition:
                    pred_rca_scores = torch.sigmoid(rca_logits[i]).cpu().tolist()
                    print(
                        f"      (RCA relevant for {loaded_rca_lookahead}-step horizon, P(Fail)={prob_fail_for_rca_horizon:.3f}):")
                    for s_local_idx, s_global_model_idx in enumerate(
                            active_sensor_indices[:min(5, len(active_sensor_indices))]):
                        s_name = canonical_sensor_names[s_global_model_idx] if s_global_model_idx < len(
                            canonical_sensor_names) else f"Sensor Pad {s_global_model_idx + 1}"
                        score = pred_rca_scores[s_global_model_idx]
                        gt_rca = rca_tgt_gt[i, s_global_model_idx].item()
                        print(
                            f"        {s_name} (idx {s_global_model_idx}): Pred Score={score:.3f}, GT RCA={gt_rca:.0f}")
                else:
                    print(
                        f"      (RCA not actively evaluated for {loaded_rca_lookahead}-step horizon based on this sample's P(Fail)={prob_fail_for_rca_horizon:.3f})")
    print("\n--- Testing Script Finished ---")


if __name__ == '__main__':
    test_model()
