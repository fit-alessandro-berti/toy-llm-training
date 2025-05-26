import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Configuration (MUST MATCH V5 TRAINING SCRIPT) ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1"  # Example, adjust if needed
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # <<< SET THIS (TestData Source)

# Model & Task Parameters (from V5 training script)
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]
FAIL_HORIZONS = [3, 5, 10]
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0]  # Default, should match preprocessor if saved
# MAX_SENSORS_CAP will be loaded from preprocessor as model_max_sensors_dim

# Architectural Params (from V5 training script)
SENSOR_INPUT_DIM = 1
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1  # Not active in eval, but part of model def

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

MOE_GLOBAL_INPUT_DIM = TRANSFORMER_D_MODEL
NUM_EXPERTS_PER_TASK = 8
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64
EXPERT_DROPOUT_RATE = 0.1  # Not active in eval

# Testing Params
TEST_BATCH_SIZE = 4  # Can be different from training BATCH_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCHES_TO_TEST = 5
SAMPLES_PER_BATCH_TO_PRINT = 2

MODEL_LOAD_PATH = "foundation_multitask_model.pth"  # <<< Path to your trained V5 model
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor.npz"  # <<< Path to your V5 preprocessor


# --- Helper: Positional Encoding (from V5 training script) ---
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


# --- TCN Residual Block (from V5 training script - NO weight_norm) ---
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


# --- Per-Sensor TCN Encoder (from V5 training script) ---
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


# --- Inter-Sensor Transformer (from V5 training script) ---
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


# --- MoE Components (from V5 training script) ---
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


# --- Test Dataset (Adapted from V5 training dataset) ---
def get_sensor_columns_for_test(df_peek):  # Consistent with training
    sensor_cols = [c for c in df_peek.columns if c.startswith("Sensor")]
    if not sensor_cols:
        potential_cols = [c for c in df_peek.columns if df_peek[c].dtype in [np.float64, np.int64, np.float32]]
        sensor_cols = [c for c in potential_cols if
                       not any(kw in c.lower() for kw in ['time', 'date', 'label', 'failure', 'id', 'current_failure'])]
    return sensor_cols


class TestMultivariateTimeSeriesDataset(Dataset):
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
                features_normalized_globally[valid_mask, i] = \
                    (raw_features_from_canonical_cols[valid_mask, i] - self.global_means[i]) / self.global_stds[i]

            failure_flags = np.zeros(len(df), dtype=np.int64)  # Default if column missing
            if "CURRENT_FAILURE" in df.columns:
                failure_flags = df["CURRENT_FAILURE"].values.astype(np.int64)
            else:
                print(f"Warning: 'CURRENT_FAILURE' not in {fp}. Fail/RCA targets will be zero for this file.")

            self.data_cache.append({
                "raw_features_globally_aligned": raw_features_from_canonical_cols,
                "features_normalized_globally": features_normalized_globally,
                "failure_flags": failure_flags, "filepath": fp
            })
            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons), self.rca_failure_lookahead)
            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        if not self.data_cache: print(f"CRITICAL WARNING: No test data loaded from {self.data_dir}."); return
        print(f"Loaded {len(self.data_cache)} test files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item = self.data_cache[file_idx]
        raw_features_aligned = item["raw_features_globally_aligned"]
        features_normalized_aligned = item["features_normalized_globally"]
        flags_full = item["failure_flags"];
        filepath = item["filepath"]

        input_slice_normed = features_normalized_aligned[window_start_idx: window_start_idx + self.seq_len]
        padded_input_normed = np.zeros((self.seq_len, self.model_max_sensors_dim), dtype=np.float32)
        sensor_mask = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        num_to_copy = min(self.num_globally_normed_features, self.model_max_sensors_dim)
        padded_input_normed[:, :num_to_copy] = input_slice_normed[:, :num_to_copy]
        for k in range(num_to_copy):
            if not np.all(np.isnan(input_slice_normed[:, k])): sensor_mask[k] = 1.0
        padded_input_normed[np.isnan(padded_input_normed)] = 0.0
        last_known_normed = padded_input_normed[-1, :].copy()

        delta_targets_normed = np.zeros((self.model_max_sensors_dim, len(self.pred_horizons)), dtype=np.float32)
        for i_h, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < features_normalized_aligned.shape[0]:
                target_values_all_normed = features_normalized_aligned[target_idx, :]
                for k in range(num_to_copy):
                    if sensor_mask[k] > 0 and not np.isnan(target_values_all_normed[k]):
                        delta_targets_normed[k, i_h] = target_values_all_normed[k] - last_known_normed[k]

        fail_targets = np.zeros(len(self.fail_horizons), dtype=np.float32)
        for i_fh, fh in enumerate(self.fail_horizons):
            start, end = window_start_idx + self.seq_len, window_start_idx + self.seq_len + fh
            if end <= len(flags_full) and np.any(flags_full[start:end]): fail_targets[i_fh] = 1.0

        rca_targets = np.zeros(self.model_max_sensors_dim, dtype=np.float32)
        start_r, end_r = window_start_idx + self.seq_len, window_start_idx + self.seq_len + self.rca_failure_lookahead
        if end_r <= len(flags_full) and np.any(flags_full[start_r:end_r]):
            current_window_raw = raw_features_aligned[window_start_idx: window_start_idx + self.seq_len, :]
            future_lookahead_raw = raw_features_aligned[start_r:end_r, :]
            for k in range(num_to_copy):
                if sensor_mask[k] > 0:
                    sensor_data_current_window_raw = current_window_raw[:, k]
                    sensor_data_future_lookahead_raw = future_lookahead_raw[:, k]
                    valid_current_raw = sensor_data_current_window_raw[~np.isnan(sensor_data_current_window_raw)]
                    valid_future_raw = sensor_data_future_lookahead_raw[~np.isnan(sensor_data_future_lookahead_raw)]
                    if len(valid_current_raw) > 0 and len(valid_future_raw) > 0:
                        mean_current_raw = np.mean(valid_current_raw);
                        std_current_raw = max(np.std(valid_current_raw), 1e-6)
                        if np.any(np.abs(valid_future_raw - mean_current_raw) > 3 * std_current_raw): rca_targets[
                            k] = 1.0

        return {"input_features": torch.from_numpy(padded_input_normed), "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values_globally_std": torch.from_numpy(last_known_normed),  # Ground truth last known
                "pred_delta_targets_globally_std": torch.from_numpy(delta_targets_normed),  # Ground truth deltas
                "fail_targets": torch.from_numpy(fail_targets), "rca_targets": torch.from_numpy(rca_targets),
                "filepath": filepath, "window_start_idx": window_start_idx}


# --- Foundational Multi-Task Model (from V5 training script, NO RevIN) ---
class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, model_max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 tcn_levels, tcn_kernel_size, tcn_dropout,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 moe_global_input_dim, num_experts_per_task, moe_hidden_dim_expert, moe_output_dim, expert_dropout_rate,
                 pred_horizons_len, fail_horizons_len):
        super().__init__()
        self.model_max_sensors = model_max_sensors;
        self.seq_len = seq_len;
        self.num_experts_per_task = num_experts_per_task;
        self.moe_output_dim = moe_output_dim

        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len, tcn_levels, tcn_kernel_size, tcn_dropout)
        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_tcn_out_dim, transformer_d_model) \
            if sensor_tcn_out_dim != transformer_d_model else nn.Identity()
        self.inter_sensor_transformer = InterSensorTransformer(transformer_d_model, transformer_nhead,
                                                               transformer_nlayers, model_max_sensors)
        self.experts_forecast = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_forecast = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.experts_fail = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_fail = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.experts_rca = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_rca = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.expert_dropout = nn.Dropout(expert_dropout_rate)  # Inactive during model.eval()

        final_combined_feat_dim_last_step = sensor_tcn_out_dim + transformer_d_model
        self.pred_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, 1)

    def _apply_moe_switch(self, global_moe_input, gating_network, expert_pool):
        gating_logits = gating_network(global_moe_input)
        chosen_expert_indices = torch.argmax(gating_logits, dim=-1)
        one_hot_selection = F.one_hot(chosen_expert_indices, num_classes=len(expert_pool)).float()
        all_expert_outputs = torch.stack([expert(global_moe_input) for expert in expert_pool], dim=1)
        moe_task_output = torch.sum(all_expert_outputs * one_hot_selection.unsqueeze(-1), dim=1)
        # No dropout or aux loss terms needed for pure inference output
        return moe_task_output

    def forward(self, x_features_globally_std, sensor_mask):
        batch_size, seq_len, _ = x_features_globally_std.shape
        x_input_masked_for_tcn = x_features_globally_std.permute(0, 2, 1) * sensor_mask.unsqueeze(-1)
        last_val_globally_std = x_input_masked_for_tcn[:, :, -1].clone()
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
        cross_sensor_context = cross_sensor_context * sensor_mask.unsqueeze(-1)

        global_moe_input_sum = (cross_sensor_context * sensor_mask.unsqueeze(-1)).sum(dim=1)
        active_sensors_per_batch = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_moe_input = global_moe_input_sum / active_sensors_per_batch

        moe_forecast_output = self._apply_moe_switch(global_moe_input, self.gating_forecast, self.experts_forecast)
        moe_fail_output = self._apply_moe_switch(global_moe_input, self.gating_fail, self.experts_fail)
        moe_rca_output = self._apply_moe_switch(global_moe_input, self.gating_rca, self.experts_rca)

        tcn_features_last_step = sensor_temporal_features[:, :, -1, :]
        combined_sensor_features_last_step = torch.cat([tcn_features_last_step, cross_sensor_context], dim=-1)

        moe_f_expanded = moe_forecast_output.unsqueeze(1).expand(-1, self.model_max_sensors, -1)
        pred_head_input = torch.cat([combined_sensor_features_last_step, moe_f_expanded], dim=-1)
        pred_delta_globally_std = self.pred_head(pred_head_input)
        pred_abs_globally_std = last_val_globally_std.unsqueeze(-1) + pred_delta_globally_std
        pred_abs_globally_std = pred_abs_globally_std * sensor_mask.unsqueeze(-1)

        fail_logits = self.fail_head(moe_fail_output)
        moe_r_expanded = moe_rca_output.unsqueeze(1).expand(-1, self.model_max_sensors, -1)
        rca_head_input = torch.cat([combined_sensor_features_last_step, moe_r_expanded], dim=-1)
        rca_logits = self.rca_head(rca_head_input).squeeze(-1)

        # During eval, aux_terms are not returned by this model's forward
        return pred_abs_globally_std, fail_logits, rca_logits


# --- Test Script Logic ---
def test_model():
    print(f"--- Test Script for Foundational Multi-Task MoE Model (V5) ---")
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_LOAD_PATH}")
    print(f"Loading preprocessor from: {PREPROCESSOR_LOAD_PATH}")

    # 1. Load Preprocessor Config (Global Stats, etc.)
    try:
        preprocessor_data = np.load(PREPROCESSOR_LOAD_PATH, allow_pickle=True)
        global_means = preprocessor_data['global_means']
        global_stds = preprocessor_data['global_stds']
        canonical_sensor_names = list(preprocessor_data['canonical_sensor_names'])
        model_max_sensors_dim = int(preprocessor_data['model_max_sensors_dim'])
        loaded_seq_len = int(preprocessor_data['seq_len'])
        loaded_pred_horizons = list(preprocessor_data['pred_horizons'])
        # Load other task horizons if needed for consistency checks or dataset init
        loaded_fail_horizons = list(
            preprocessor_data.get('fail_horizons', FAIL_HORIZONS))  # Default if not in older pkl
        loaded_rca_lookahead = int(preprocessor_data.get('rca_failure_lookahead', RCA_FAILURE_LOOKAHEAD))

        # Verify consistency (optional, but good practice)
        if loaded_seq_len != SEQ_LEN: print(f"Warning: SEQ_LEN mismatch! Loaded: {loaded_seq_len}, Script: {SEQ_LEN}")
        if loaded_pred_horizons != PRED_HORIZONS: print(
            f"Warning: PRED_HORIZONS mismatch! Loaded: {loaded_pred_horizons}, Script: {PRED_HORIZONS}")

    except FileNotFoundError:
        print(f"ERROR: Preprocessor file not found at {PREPROCESSOR_LOAD_PATH}. Exiting."); return
    except KeyError as e:
        print(f"ERROR: Missing key {e} in preprocessor file. Exiting."); return

    print(
        f"Preprocessor loaded: model_max_sensors_dim={model_max_sensors_dim}, {len(canonical_sensor_names)} canonical sensors.")

    # 2. Initialize Dataset and DataLoader
    if not (os.path.exists(VALID_DIR) and os.path.isdir(VALID_DIR)): print(
        f"ERROR: VALID_DIR '{VALID_DIR}' missing."); return

    test_dataset = TestMultivariateTimeSeriesDataset(
        data_dir=VALID_DIR, seq_len=SEQ_LEN,
        pred_horizons=PRED_HORIZONS, fail_horizons=loaded_fail_horizons, rca_failure_lookahead=loaded_rca_lookahead,
        model_max_sensors_dim=model_max_sensors_dim, global_means=global_means,
        global_stds=global_stds, canonical_sensor_names=canonical_sensor_names
    )
    if len(test_dataset) == 0: print(f"No data in {VALID_DIR} or no valid windows. Exiting."); return
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Initialize Model
    model = FoundationalTimeSeriesModel(
        model_max_sensors=model_max_sensors_dim, seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM, tcn_levels=TCN_LEVELS,
        tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        moe_global_input_dim=MOE_GLOBAL_INPUT_DIM, num_experts_per_task=NUM_EXPERTS_PER_TASK,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        expert_dropout_rate=EXPERT_DROPOUT_RATE,  # Inactive in eval
        pred_horizons_len=len(PRED_HORIZONS), fail_horizons_len=len(loaded_fail_horizons)
    ).to(DEVICE)

    # 4. Load Trained Model Weights
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {MODEL_LOAD_PATH}. Exiting."); return
    except Exception as e:
        print(f"ERROR loading model state: {e}. Check architecture match. Exiting."); return

    model.eval()
    print("Model loaded and in evaluation mode.")

    # 5. Perform Inference and Print Results
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= MAX_BATCHES_TO_TEST:
                print(f"\nReached MAX_BATCHES_TO_TEST ({MAX_BATCHES_TO_TEST}). Stopping.");
                break
            print(f"\n--- Batch {batch_idx + 1}/{min(MAX_BATCHES_TO_TEST, len(test_loader))} ---")

            input_globally_std = batch["input_features"].to(DEVICE)
            sensor_m = batch["sensor_mask"].to(DEVICE)
            last_k_std_gt = batch["last_known_values_globally_std"].to(DEVICE)
            delta_tgt_std_gt = batch["pred_delta_targets_globally_std"].to(DEVICE)
            fail_tgt_gt = batch["fail_targets"].to(DEVICE)
            rca_tgt_gt = batch["rca_targets"].to(DEVICE)

            # Model forward pass
            pred_abs_globally_std, fail_logits, rca_logits = model(input_globally_std, sensor_m)

            # Calculate GT absolute values in standardized space for comparison
            actual_abs_targets_std = last_k_std_gt.unsqueeze(-1) + delta_tgt_std_gt

            num_samples_to_print = min(SAMPLES_PER_BATCH_TO_PRINT, input_globally_std.size(0))
            for i in range(num_samples_to_print):
                sample_fp = batch["filepath"][i];
                sample_win_start = batch["window_start_idx"][i].item()
                print(f"\n  Sample {i + 1} (File: {os.path.basename(sample_fp)}, WinStart: {sample_win_start})")
                active_sensor_indices = torch.where(sensor_m[i] == 1.0)[0].cpu().tolist()
                if not active_sensor_indices: print("    No active sensors in this sample."); continue

                print("    Forecast (Globally Standardized Space):")
                for h_idx, horizon in enumerate(PRED_HORIZONS):
                    print(f"      H={horizon}:")
                    for s_local_idx, s_global_model_idx in enumerate(active_sensor_indices[:min(3,
                                                                                                len(active_sensor_indices))]):  # Print for first few active sensors
                        gt_val = actual_abs_targets_std[i, s_global_model_idx, h_idx].item()
                        pred_val = pred_abs_globally_std[i, s_global_model_idx, h_idx].item()
                        # Find original canonical name for this model_idx if it's within num_globally_normed_features
                        s_name = canonical_sensor_names[s_global_model_idx] if s_global_model_idx < len(
                            canonical_sensor_names) else f"Sensor Pad {s_global_model_idx + 1}"
                        print(
                            f"        {s_name} (idx {s_global_model_idx}): GT Abs_Std={gt_val:.3f}, Pred Abs_Std={pred_val:.3f}")

                print("    Failure Prediction:")
                pred_fail_probs = torch.sigmoid(fail_logits[i]).cpu().tolist()
                gt_fail_status = fail_tgt_gt[i].cpu().tolist()
                for h_idx, horizon in enumerate(loaded_fail_horizons):
                    print(
                        f"      Next {horizon} steps: GT Fail={gt_fail_status[h_idx]:.0f}, Pred Prob={pred_fail_probs[h_idx]:.3f}")

                print("    RCA Prediction (Sigmoid Scores):")
                # Check condition for RCA, e.g., if failure predicted for RCA_FAILURE_LOOKAHEAD
                idx_rca_horizon = loaded_fail_horizons.index(loaded_rca_lookahead)  # Find index of the specific horizon
                rca_eval_condition = pred_fail_probs[idx_rca_horizon] > 0.5 or gt_fail_status[idx_rca_horizon] == 1.0

                if rca_eval_condition:
                    pred_rca_scores = torch.sigmoid(rca_logits[i]).cpu().tolist()  # [model_max_sensors]
                    print(f"      (RCA relevant for {loaded_rca_lookahead}-step horizon):")
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
                        f"      (RCA not actively evaluated for {loaded_rca_lookahead}-step horizon based on this sample's failure pred/GT)")
    print("\n--- Testing Script Finished ---")


if __name__ == '__main__':
    test_model()
