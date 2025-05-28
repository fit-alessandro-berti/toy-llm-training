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
BASE_DATA_DIR = "../../data/time_series/2"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")
OUTPUT_CSV_FILENAME = "output_rca_foundational_fm.csv"  # <<< Name for RCA output

# Model & Task Parameters (from latest training script, defaults if not in preprocessor)
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]  # Needed for dataset and model structure
FAIL_HORIZONS = [3, 5, 10]
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0] if FAIL_HORIZONS else 3  # Default if FAIL_HORIZONS empty

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

# MoE Parameters
NUM_SHARED_EXPERTS = 8
MOE_EXPERT_INPUT_DIM = TRANSFORMER_D_MODEL
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64
MOE_TOP_K = 2

# Testing Params
TEST_BATCH_SIZE = 4  # Adjust based on your GPU memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCHES_TO_TEST = float('inf')  # Process all data
SAMPLES_PER_BATCH_TO_PRINT = 2  # For console summary

# --- Paths ---
MODEL_LOAD_PATH = "foundation_multitask_model_v3_moe_vectorized_ema_cost_sensitive_updated.pth"
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v3_ema_updated.npz"

# Hysteresis thresholds (not directly for RCA CSV, but part of model script)
HYSTERESIS_HI_THRESH = 0.6
HYSTERESIS_LO_THRESH = 0.4


# --- Model Classes (Copied from your provided foundational model script) ---
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


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): super().__init__(); self.fc = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x): return self.fc(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts): super().__init__(); self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x): return self.fc(x)


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
        print(f"Loading test data from {self.data_dir} for {len(self.file_paths)} files...")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping test file {fp}: {e}"); continue
            raw_features = np.full((len(df), self.num_globally_normed_features), np.nan, dtype=np.float32)
            for i, name in enumerate(self.canonical_sensor_names):
                if name in df.columns: raw_features[:, i] = df[name].values.astype(np.float32)
            if np.all(np.isnan(raw_features)): continue
            features_normed = np.full_like(raw_features, np.nan)
            for i in range(self.num_globally_normed_features):
                valid_mask = ~np.isnan(raw_features[:, i])
                if self.global_stds[i] > 1e-8:
                    features_normed[valid_mask, i] = (raw_features[valid_mask, i] - self.global_means[i]) / \
                                                     self.global_stds[i]
                else:
                    features_normed[valid_mask, i] = 0.0
            failure_flags = np.zeros(len(df), dtype=np.int64)
            if "CURRENT_FAILURE" in df.columns: failure_flags = df["CURRENT_FAILURE"].values.astype(np.int64)
            # else: print(f"Warning: 'CURRENT_FAILURE' column not found in {fp}.") # Less verbose
            self.data_cache.append(
                {"raw_features": raw_features, "features_normed": features_normed, "failure_flags": failure_flags,
                 "filepath": fp})
            max_lookahead = max(max(self.pred_horizons if self.pred_horizons else [0]),
                                max(self.fail_horizons if self.fail_horizons else [0]), self.rca_failure_lookahead or 0)
            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        if not self.data_cache: print(f"CRITICAL WARNING: No test data loaded from {self.data_dir}."); return
        print(f"Loaded {len(self.data_cache)} test files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, win_start_idx = self.window_indices[idx]
        item = self.data_cache[file_idx]
        features_normed_aligned = item["features_normed"]
        flags_full = item["failure_flags"]
        raw_features_aligned = item["raw_features"]  # For RCA GT calculation
        input_slice_normed = features_normed_aligned[win_start_idx: win_start_idx + self.seq_len]
        padded_input_normed = np.zeros((self.seq_len, self.model_max_sensors_dim), dtype=np.float32)
        sensor_mask = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        num_to_copy = min(self.num_globally_normed_features, self.model_max_sensors_dim)
        padded_input_normed[:, :num_to_copy] = input_slice_normed[:, :num_to_copy]
        for k_idx in range(num_to_copy):
            if not np.all(np.isnan(input_slice_normed[:, k_idx])): sensor_mask[k_idx] = 1.0
        padded_input_normed = np.nan_to_num(padded_input_normed, nan=0.0)  # Fill any NaNs from padding
        last_known_normed = padded_input_normed[-1, :].copy()

        delta_targets_normed = np.zeros((self.model_max_sensors_dim, len(self.pred_horizons or [])), dtype=np.float32)
        if self.pred_horizons:
            for i_h, h in enumerate(self.pred_horizons):
                target_idx = win_start_idx + self.seq_len + h - 1
                if target_idx < features_normed_aligned.shape[0]:
                    target_vals = features_normed_aligned[target_idx, :]
                    for k_idx in range(num_to_copy):
                        if sensor_mask[k_idx] > 0 and not np.isnan(target_vals[k_idx]):
                            delta_targets_normed[k_idx, i_h] = target_vals[k_idx] - last_known_normed[k_idx]

        fail_targets = np.zeros(len(self.fail_horizons or []), dtype=np.float32)
        if self.fail_horizons:
            for i_fh, fh in enumerate(self.fail_horizons):
                start, end = win_start_idx + self.seq_len, win_start_idx + self.seq_len + fh
                if end <= len(flags_full) and np.any(flags_full[start:end]): fail_targets[i_fh] = 1.0

        rca_targets = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        if self.rca_failure_lookahead > 0:
            start_r, end_r = win_start_idx + self.seq_len, win_start_idx + self.seq_len + self.rca_failure_lookahead
            if end_r <= len(flags_full) and np.any(flags_full[start_r:end_r]):
                current_win_raw = raw_features_aligned[win_start_idx: win_start_idx + self.seq_len, :]
                future_lookahead_raw = raw_features_aligned[start_r:end_r, :]
                for k_idx in range(num_to_copy):
                    if sensor_mask[k_idx] > 0:
                        sensor_curr_raw = current_win_raw[~np.isnan(current_win_raw[:, k_idx]), k_idx]
                        sensor_future_raw = future_lookahead_raw[~np.isnan(future_lookahead_raw[:, k_idx]), k_idx]
                        if len(sensor_curr_raw) > 0 and len(sensor_future_raw) > 0:
                            mean_curr, std_curr = np.mean(sensor_curr_raw), max(np.std(sensor_curr_raw), 1e-6)
                            if np.any(np.abs(sensor_future_raw - mean_curr) > 3 * std_curr): rca_targets[k_idx] = 1.0

        return {"input_features": torch.from_numpy(padded_input_normed), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_globally_std": torch.from_numpy(last_known_normed),
                "pred_delta_targets_globally_std": torch.from_numpy(delta_targets_normed),
                "fail_targets": torch.from_numpy(fail_targets), "rca_targets": torch.from_numpy(rca_targets),
                "filepath": item["filepath"], "window_start_idx": win_start_idx}


class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, model_max_sensors, seq_len, sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout, transformer_d_model, transformer_nhead, transformer_nlayers,
                 num_shared_experts, moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim,
                 pred_horizons_len, fail_horizons_len, moe_top_k, moe_noise_std=0.0, aux_loss_coeff=0.0,
                 entropy_reg_coeff=0.0):
        super().__init__()
        self.model_max_sensors = model_max_sensors;
        self.seq_len = seq_len;
        self.num_shared_experts = num_shared_experts
        self.moe_output_dim = moe_output_dim;
        self.transformer_d_model = transformer_d_model;
        self.sensor_tcn_out_dim = sensor_tcn_out_dim
        self.moe_top_k = moe_top_k;
        self.moe_noise_std = moe_noise_std;
        self.aux_loss_coeff = aux_loss_coeff;
        self.entropy_reg_coeff = entropy_reg_coeff
        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)
        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_tcn_out_dim,
                                                        transformer_d_model) if sensor_tcn_out_dim != transformer_d_model else nn.Identity()
        self.inter_sensor_transformer = InterSensorTransformer(transformer_d_model, transformer_nhead,
                                                               transformer_nlayers, model_max_sensors)
        self.experts_shared = nn.ModuleList(
            [Expert(moe_expert_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_shared_experts)])
        self.gates = nn.ModuleDict({
            "forecast": GatingNetwork(transformer_d_model * 2, num_shared_experts),
            "fail": GatingNetwork(transformer_d_model * 2, num_shared_experts),
            "rca": GatingNetwork(transformer_d_model, num_shared_experts)})
        self.pred_head = nn.Linear(sensor_tcn_out_dim + transformer_d_model + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(sensor_tcn_out_dim + transformer_d_model + moe_output_dim,
                                  1)  # Output is per sensor, hence 1

    def _apply_moe_topk(self, x_expert_input, gate_input, gate_network, experts_modulelist, k, noise_std):
        logits = gate_network(gate_input)
        if self.training and noise_std > 0: logits = logits + torch.randn_like(logits) * noise_std
        eff_k = min(k, len(experts_modulelist))
        topk_val, topk_idx = torch.topk(logits, eff_k, dim=-1);
        topk_w = torch.softmax(topk_val, dim=-1)
        all_out = torch.stack([e(x_expert_input) for e in experts_modulelist], dim=1)
        sel_out = all_out.gather(1, topk_idx.unsqueeze(-1).expand(list(topk_idx.shape) + [all_out.size(-1)]))
        y = (sel_out * topk_w.unsqueeze(-1)).sum(dim=1)
        # Aux loss calculation (not essential for inference output)
        router_prob_for_loss = torch.softmax(logits, -1);
        avg_router_prob = router_prob_for_loss.mean(0)
        num_examples_for_expert_frac = gate_input.size(0) * eff_k if gate_input.size(0) > 0 else 1.0
        expert_frac = torch.zeros_like(avg_router_prob).scatter_add_(0, topk_idx.view(-1), torch.ones_like(topk_idx,
                                                                                                           dtype=router_prob_for_loss.dtype).view(
            -1)) / num_examples_for_expert_frac
        load_balance_loss = self.num_shared_experts * (avg_router_prob * expert_frac).sum()
        return y, load_balance_loss, logits

    def forward(self, x_features_globally_std, sensor_mask):
        B, S, _ = x_features_globally_std.shape  # B, SeqLen, MaxSensors
        x_p = x_features_globally_std.permute(0, 2, 1)  # B, MaxSensors, SeqLen
        last_val = x_p[:, :, -1].clone()  # B, MaxSensors
        x_enc_in = x_p.unsqueeze(-1).reshape(B * self.model_max_sensors, S, SENSOR_INPUT_DIM)
        s_temporal_f_flat = self.per_sensor_encoder(x_enc_in)  # B*MaxSensors, SeqLen, TcnOutDim
        s_temporal_f = s_temporal_f_flat.reshape(B, self.model_max_sensors, S, self.sensor_tcn_out_dim)
        s_temporal_f = s_temporal_f * sensor_mask.view(B, self.model_max_sensors, 1, 1)
        pooled_s_f = torch.mean(s_temporal_f, dim=2)  # B, MaxSensors, TcnOutDim
        proj_for_tx = self.pooled_to_transformer_dim_proj(pooled_s_f)  # B, MaxSensors, TransformerDModel
        tx_pad_mask = (sensor_mask == 0)  # B, MaxSensors
        cross_s_ctx = self.inter_sensor_transformer(proj_for_tx, tx_pad_mask)  # B, MaxSensors, TransformerDModel
        cross_s_ctx_masked = cross_s_ctx * sensor_mask.unsqueeze(-1)
        active_s_count = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_ctx_global = cross_s_ctx_masked.sum(dim=1) / active_s_count  # B, TransformerDModel
        var_ctx_global = (cross_s_ctx_masked ** 2).sum(dim=1) / active_s_count - mean_ctx_global ** 2
        router_in_global = torch.cat([mean_ctx_global, torch.sqrt(var_ctx_global.clamp(min=1e-6))], dim=-1)
        noise = self.moe_noise_std if self.training else 0.0
        moe_f_out, aux_f, _ = self._apply_moe_topk(mean_ctx_global, router_in_global, self.gates["forecast"],
                                                   self.experts_shared, self.moe_top_k, noise)
        moe_fail_out, _, _ = self._apply_moe_topk(mean_ctx_global, router_in_global, self.gates["fail"],
                                                  self.experts_shared, self.moe_top_k, noise)

        # RCA MoE (token-level)
        rca_expert_in_flat = cross_s_ctx_masked.reshape(-1, self.transformer_d_model)  # B*MaxSensors, D_tx
        valid_token_mask_rca = sensor_mask.view(-1).bool()  # B*MaxSensors
        rca_expert_in_valid = rca_expert_in_flat[valid_token_mask_rca]  # NumValidTokens, D_tx
        moe_rca_out_flat = torch.zeros(B * self.model_max_sensors, self.moe_output_dim,
                                       device=x_features_globally_std.device, dtype=moe_f_out.dtype)
        if rca_expert_in_valid.size(0) > 0:
            # Gate input for RCA is just the token embedding itself
            moe_rca_out_valid, _, _ = self._apply_moe_topk(rca_expert_in_valid, rca_expert_in_valid, self.gates["rca"],
                                                           self.experts_shared, self.moe_top_k, noise)
            moe_rca_out_flat[valid_token_mask_rca] = moe_rca_out_valid

        # Prediction Heads
        s_temporal_last = s_temporal_f[:, :, -1, :]  # B, MaxSensors, TcnOutDim
        pred_head_in = torch.cat(
            [s_temporal_last, cross_s_ctx_masked, moe_f_out.unsqueeze(1).expand(-1, self.model_max_sensors, -1)],
            dim=-1)
        pred_delta_std = self.pred_head(pred_head_in)  # B, MaxSensors, PredHorizonsLen
        pred_abs_std = last_val.unsqueeze(-1) + pred_delta_std;
        pred_abs_std = pred_abs_std * sensor_mask.unsqueeze(-1)
        fail_logits = self.fail_head(moe_fail_out)  # B, FailHorizonsLen
        rca_head_in_flat = torch.cat([s_temporal_last.reshape(-1, self.sensor_tcn_out_dim),
                                      cross_s_ctx_masked.reshape(-1, self.transformer_d_model),
                                      moe_rca_out_flat], dim=-1)  # B*MaxSensors, Dim
        rca_logits_flat = self.rca_head(rca_head_in_flat).squeeze(-1)  # B*MaxSensors
        rca_logits = rca_logits_flat.view(B, self.model_max_sensors)  # B, MaxSensors
        return pred_abs_std, fail_logits, rca_logits, torch.tensor(0.0), torch.tensor(
            0.0)  # Aux/Entropy loss not needed for eval


# --- Main Test Script Logic ---
def test_foundational_rca():
    print(f"--- Test Script for Foundational Model RCA Output ---")
    print(f"Using device: {DEVICE}")

    output_rca_data = []

    # 1. Load Preprocessor Config
    try:
        preprocessor_data = np.load(PREPROCESSOR_LOAD_PATH, allow_pickle=True)
        global_means = preprocessor_data['global_means']
        global_stds = preprocessor_data['global_stds']
        canonical_sensor_names = list(preprocessor_data['canonical_sensor_names'])
        model_max_sensors_dim = int(preprocessor_data['model_max_sensors_dim'])
        loaded_seq_len = int(preprocessor_data.get('seq_len', SEQ_LEN))
        loaded_pred_horizons_np = preprocessor_data.get('pred_horizons', np.array(PRED_HORIZONS))
        loaded_pred_horizons = list(loaded_pred_horizons_np) if isinstance(loaded_pred_horizons_np,
                                                                           np.ndarray) else PRED_HORIZONS
        loaded_fail_horizons_np = preprocessor_data.get('fail_horizons', np.array(FAIL_HORIZONS))
        loaded_fail_horizons = list(loaded_fail_horizons_np) if isinstance(loaded_fail_horizons_np,
                                                                           np.ndarray) else FAIL_HORIZONS
        default_rca_lh = loaded_fail_horizons[0] if loaded_fail_horizons else RCA_FAILURE_LOOKAHEAD
        loaded_rca_lookahead = int(preprocessor_data.get('rca_failure_lookahead', default_rca_lh))
        print(f"Preprocessor loaded. RCA Lookahead: {loaded_rca_lookahead}, MaxSensors: {model_max_sensors_dim}")
    except FileNotFoundError:
        print(f"ERROR: Preprocessor file '{PREPROCESSOR_LOAD_PATH}' not found."); return
    except KeyError as e:
        print(f"ERROR: Missing key {e} in preprocessor file."); return

    # Determine index for RCA lookahead in fail_horizons list for context
    rca_lh_fail_horizon_idx = None
    if loaded_fail_horizons and loaded_rca_lookahead in loaded_fail_horizons:
        try:
            rca_lh_fail_horizon_idx = loaded_fail_horizons.index(loaded_rca_lookahead)
        except ValueError:
            pass  # rca_lookahead is not in fail_horizons

    # 2. Initialize Dataset and DataLoader
    if not (os.path.exists(VALID_DIR) and os.path.isdir(VALID_DIR)): print(
        f"ERROR: VALID_DIR '{VALID_DIR}' not found."); return
    test_dataset = TestMultivariateTimeSeriesDataset(
        data_dir=VALID_DIR, seq_len=loaded_seq_len, pred_horizons=loaded_pred_horizons,
        fail_horizons=loaded_fail_horizons, rca_failure_lookahead=loaded_rca_lookahead,
        model_max_sensors_dim=model_max_sensors_dim, global_means=global_means,
        global_stds=global_stds, canonical_sensor_names=canonical_sensor_names)
    if len(test_dataset) == 0: print(f"ERROR: No data in TestDataset from {VALID_DIR}."); return

    # Process all data
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Total batches to process: {len(test_loader)}")

    # 3. Initialize Model
    model = FoundationalTimeSeriesModel(
        model_max_sensors=model_max_sensors_dim, seq_len=loaded_seq_len, sensor_input_dim=SENSOR_INPUT_DIM,
        sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM, sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        num_shared_experts=NUM_SHARED_EXPERTS, moe_expert_input_dim=MOE_EXPERT_INPUT_DIM,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        pred_horizons_len=len(loaded_pred_horizons or []), fail_horizons_len=len(loaded_fail_horizons or []),
        moe_top_k=MOE_TOP_K
    ).to(DEVICE)

    # 4. Load Trained Model Weights
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model file '{MODEL_LOAD_PATH}' not found."); return
    except RuntimeError as e:
        print(f"ERROR loading model state_dict: {e}. Check architecture match."); return
    model.eval()
    print("Model loaded and in evaluation mode.")

    # 5. Perform Inference and Collect RCA Results
    with torch.no_grad():
        for batch_idx, batch_content in enumerate(test_loader):
            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                print(f"  Processing Batch {batch_idx + 1}/{len(test_loader)}")

            input_std = batch_content["input_features"].to(DEVICE)
            sensor_m = batch_content["sensor_mask"].to(DEVICE)
            gt_fail_targets = batch_content["fail_targets"].to(DEVICE)
            gt_rca_targets = batch_content["rca_targets"].to(DEVICE)

            _, pred_fail_logits, pred_rca_logits, _, _ = model(input_std, sensor_m)

            for i in range(input_std.size(0)):  # Iterate samples in batch
                filepath_sample = batch_content["filepath"][i]
                win_start_sample = batch_content["window_start_idx"][i].item()

                # Get overall failure context for this sample at RCA lookahead horizon
                gt_overall_fail_at_rca_lh = np.nan
                pred_overall_fail_prob_at_rca_lh = np.nan
                if rca_lh_fail_horizon_idx is not None and rca_lh_fail_horizon_idx < gt_fail_targets.shape[1]:
                    gt_overall_fail_at_rca_lh = gt_fail_targets[i, rca_lh_fail_horizon_idx].item()
                    pred_overall_fail_prob_at_rca_lh = torch.sigmoid(
                        pred_fail_logits[i, rca_lh_fail_horizon_idx]).item()

                for s_model_idx in range(model_max_sensors_dim):
                    sensor_name_str = canonical_sensor_names[s_model_idx] if s_model_idx < len(
                        canonical_sensor_names) else f"SensorIDX_{s_model_idx}"
                    is_active = sensor_m[i, s_model_idx].item()

                    # Only log detailed RCA for active sensors or all sensors if desired (here for all canonical ones)
                    # If sensor_mask is 0, its RCA logit might be non-zero but influenced by padding or model structure.
                    # The ground truth rca_target for a non-active sensor should be 0.

                    output_rca_data.append({
                        'filepath': os.path.basename(filepath_sample),
                        'window_start_idx': win_start_sample,
                        'rca_lookahead_horizon_config': loaded_rca_lookahead,
                        'gt_overall_failure_at_rca_lh': gt_overall_fail_at_rca_lh,
                        'pred_overall_failure_prob_at_rca_lh': pred_overall_fail_prob_at_rca_lh,
                        'sensor_name': sensor_name_str,
                        'sensor_model_idx': s_model_idx,
                        'is_sensor_active_in_window': is_active,
                        'ground_truth_sensor_rca_label': gt_rca_targets[i, s_model_idx].item(),
                        'predicted_sensor_rca_logit': pred_rca_logits[i, s_model_idx].item(),
                        'predicted_sensor_rca_probability': torch.sigmoid(pred_rca_logits[i, s_model_idx]).item()
                    })

                # Console print for limited samples (can be adapted from your original script)
                if batch_idx * TEST_BATCH_SIZE + i < SAMPLES_PER_BATCH_TO_PRINT:
                    print(
                        f"\n  Sample (Overall #{batch_idx * TEST_BATCH_SIZE + i + 1}) File: {os.path.basename(filepath_sample)}, Window: {win_start_sample}")
                    print(
                        f"    RCA Lookahead: {loaded_rca_lookahead}, GT Fail at LH: {gt_overall_fail_at_rca_lh}, Pred Fail Prob at LH: {pred_overall_fail_prob_at_rca_lh:.3f}")
                    print(f"    Sensor RCA (Name | GT | Pred Prob):")
                    active_indices = torch.where(sensor_m[i] == 1.0)[0].cpu().tolist()
                    for s_print_idx in active_indices[:min(5, len(active_indices))]:  # Print first 5 active
                        name = canonical_sensor_names[s_print_idx] if s_print_idx < len(
                            canonical_sensor_names) else f"Sensor_{s_print_idx}"
                        gt_val = gt_rca_targets[i, s_print_idx].item()
                        pred_prob_val = torch.sigmoid(pred_rca_logits[i, s_print_idx]).item()
                        print(
                            f"      - {name:<15} | {gt_val:.0f}  | {pred_prob_val:.3f} (Logit: {pred_rca_logits[i, s_print_idx].item():.2f})")

    # 6. Write to CSV
    if output_rca_data:
        output_df = pd.DataFrame(output_rca_data)
        try:
            output_df.to_csv(OUTPUT_CSV_FILENAME, index=False)
            print(f"\nSuccessfully wrote {len(output_df)} RCA data points to {OUTPUT_CSV_FILENAME}")
        except Exception as e:
            print(f"\nERROR: Could not write RCA outputs to CSV: {e}")
    else:
        print("\nNo RCA data collected to write to CSV.")

    print("\n--- Foundational Model RCA Test Script Finished ---")


if __name__ == '__main__':
    if not os.path.exists(PREPROCESSOR_LOAD_PATH): print(
        f"CRITICAL ERROR: Preprocessor '{PREPROCESSOR_LOAD_PATH}' not found."); exit()
    if not os.path.exists(MODEL_LOAD_PATH): print(f"CRITICAL ERROR: Model '{MODEL_LOAD_PATH}' not found."); exit()
    if not os.path.exists(VALID_DIR) or not os.listdir(VALID_DIR): print(
        f"CRITICAL ERROR: Valid dir '{VALID_DIR}' empty/not found."); exit()
    test_foundational_rca()
