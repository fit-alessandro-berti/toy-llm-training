import os
import glob
import random
import math  # For positional encoding
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- Configuration (MUST MATCH V6 TRAINING SCRIPT) ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")

# Model & Training Parameters (from V6 training script)
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]
FAIL_HORIZONS = [3, 5, 10]
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0]
MAX_SENSORS_CAP = 20

# Architectural Params (from V6 training script)
SENSOR_INPUT_DIM = 1
SENSOR_TCN_PROJ_DIM = 32
SENSOR_TCN_OUT_DIM = 32
TCN_LEVELS = 4
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1  # Not active in eval mode, but part of model def

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

MOE_GLOBAL_INPUT_DIM = TRANSFORMER_D_MODEL
NUM_EXPERTS_PER_TASK = 8
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64
EXPERT_DROPOUT_RATE = 0.1  # Not active in eval mode

# Testing Params
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCHES_TO_TEST = 5
SAMPLES_PER_BATCH_TO_PRINT = 2

MODEL_LOAD_PATH = "foundation_timeseries_model_v6.pth"
PREPROCESSOR_LOAD_PATH = "preprocessor_config_v6.npz"


# --- RevIN Layer (from V6 training script) ---
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def forward(self, x, mode):  # x: [Batch, NumSensors, SeqLen]
        if mode == 'norm':
            self._get_statistics(x)
            x_norm = (x - self.mean) / self.stdev
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm
        elif mode == 'denorm':
            if not hasattr(self, 'mean') or not hasattr(self, 'stdev'):
                return x
            x_denorm = x
            if self.affine:
                safe_affine_weight = self.affine_weight + self.eps
                x_denorm = (x_denorm - self.affine_bias) / safe_affine_weight
            x_denorm = x_denorm * self.stdev + self.mean
            return x_denorm
        else:
            raise NotImplementedError(f"RevIN mode '{mode}' not implemented.")

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=-1, keepdim=True)
        self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps)


# --- Helper: Positional Encoding (from V6 training script) ---
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


# --- TCN Residual Block (from V6 training script) ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        try:
            self.conv1 = nn.utils.parametrizations.weight_norm(self.conv1)
        except AttributeError:
            self.conv1 = nn.utils.weight_norm(self.conv1)
        self.relu1 = nn.ReLU();
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.padding,
                               dilation=dilation)
        try:
            self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)
        except AttributeError:
            self.conv2 = nn.utils.weight_norm(self.conv2)
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


# --- Per-Sensor TCN Encoder (from V6 training script) ---
class PerSensorEncoderTCN(nn.Module):
    def __init__(self, input_dim, proj_dim, tcn_out_dim, seq_len, num_levels, kernel_size, dropout):
        super(PerSensorEncoderTCN, self).__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=seq_len)
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
        x = self.tcn_network(x)
        x = x.permute(0, 2, 1);
        x = self.final_norm(x)
        return x


# --- Data Handling (Replicated from V6 training script) ---
def get_max_sensors_from_files(file_paths):
    max_s = 0
    for fp in file_paths:
        try:
            df_peek = pd.read_csv(fp, nrows=1); max_s = max(max_s, df_peek.shape[1] - 1)
        except Exception:
            continue
    return min(max_s, MAX_SENSORS_CAP)


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizons, fail_horizons, rca_failure_lookahead, max_sensors_global):
        self.data_dir = data_dir;
        self.seq_len = seq_len;
        self.pred_horizons = pred_horizons;
        self.fail_horizons = fail_horizons
        self.rca_failure_lookahead = rca_failure_lookahead;
        self.max_sensors_global = max_sensors_global
        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"));
        self.data_cache = [];
        self.window_indices = []
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(f"Loading data from {self.data_dir}...")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping file {fp} due to loading error: {e}"); continue
            sensor_cols = [c for c in df.columns if c.startswith("Sensor")]
            if not sensor_cols: continue
            if len(sensor_cols) > self.max_sensors_global: sensor_cols = sensor_cols[:self.max_sensors_global]
            features = df[sensor_cols].values.astype(np.float32);
            failure_flags = df["CURRENT_FAILURE"].values.astype(np.int64)
            num_actual_sensors = features.shape[1]
            scalers = [StandardScaler() for _ in range(num_actual_sensors)]
            for i in range(num_actual_sensors):
                if features.shape[0] > 0: features[:, i] = scalers[i].fit_transform(
                    features[:, i].reshape(-1, 1)).flatten()
            self.data_cache.append({
                "features": features, "failure_flags": failure_flags,
                "num_actual_sensors": num_actual_sensors, "filepath": fp
            })
            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons), self.rca_failure_lookahead)
            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item = self.data_cache[file_idx];
        features_full, flags_full, n_actual = item["features"], item["failure_flags"], item["num_actual_sensors"]
        filepath = item["filepath"]
        input_orig = features_full[window_start_idx: window_start_idx + self.seq_len]
        last_known = np.zeros(self.max_sensors_global, dtype=np.float32)
        if input_orig.shape[0] > 0: last_known[:n_actual] = input_orig[-1, :n_actual]
        padded_input = np.zeros((self.seq_len, self.max_sensors_global), dtype=np.float32);
        padded_input[:, :n_actual] = input_orig
        mask = np.zeros(self.max_sensors_global, dtype=np.float32);
        mask[:n_actual] = 1.0
        delta_targets = np.zeros((self.max_sensors_global, len(self.pred_horizons)), dtype=np.float32)
        for i, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < len(features_full): delta_targets[:n_actual, i] = features_full[target_idx,
                                                                              :n_actual] - last_known[:n_actual]
        fail_t = np.zeros(len(self.fail_horizons), dtype=np.float32)
        for i, n in enumerate(self.fail_horizons):
            start, end = window_start_idx + self.seq_len, window_start_idx + self.seq_len + n
            if end <= len(flags_full) and np.any(flags_full[start:end]): fail_t[i] = 1.0
        rca_t = np.zeros(self.max_sensors_global, dtype=np.float32)
        start_r, end_r = window_start_idx + self.seq_len, window_start_idx + self.seq_len + self.rca_failure_lookahead
        if end_r <= len(flags_full) and np.any(flags_full[start_r:end_r]) and n_actual > 0:
            sub_feat = features_full[start_r:end_r, :n_actual];
            in_feat = features_full[window_start_idx:window_start_idx + self.seq_len, :n_actual]
            if sub_feat.shape[0] > 0 and in_feat.shape[0] > 0:
                means, stds = np.mean(in_feat, axis=0), np.std(in_feat, axis=0);
                stds[stds < 1e-6] = 1e-6
                for s in range(n_actual):
                    if np.any(np.abs(sub_feat[:, s] - means[s]) > 3 * stds[s]): rca_t[s] = 1.0
        return {"input_features": torch.from_numpy(padded_input), "sensor_mask": torch.from_numpy(mask),
                "last_known_values": torch.from_numpy(last_known),
                "pred_delta_targets": torch.from_numpy(delta_targets),
                "fail_targets": torch.from_numpy(fail_t), "rca_targets": torch.from_numpy(rca_t),
                "filepath": filepath, "window_start_idx": window_start_idx}


# --- Model Architecture (V6 with TCN, replicated from V6 training script) ---
class InterSensorTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, max_sensors):
        super().__init__();
        self.pos_encoder_inter_sensor = nn.Parameter(torch.zeros(1, max_sensors, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True,
                                                   dim_feedforward=embed_dim * 2, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers);
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask):
        x = x + self.pos_encoder_inter_sensor[:, :x.size(1), :];
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_norm(x)


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


class FoundationalTimeSeriesModelV6(nn.Module):  # Renamed from V5 to V6 to match training script context
    def __init__(self, max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim, tcn_levels, tcn_kernel_size, tcn_dropout,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 moe_global_input_dim, num_experts_per_task, moe_hidden_dim_expert, moe_output_dim, expert_dropout_rate,
                 pred_horizons_len, fail_horizons_len, revin_affine=True):
        super().__init__()
        self.max_sensors = max_sensors;
        self.seq_len = seq_len;
        self.num_experts_per_task = num_experts_per_task
        self.moe_output_dim = moe_output_dim
        self.revin_layer = RevIN(num_features=max_sensors, affine=revin_affine)

        self.per_sensor_encoder = PerSensorEncoderTCN(sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                                                      seq_len,
                                                      tcn_levels, tcn_kernel_size, tcn_dropout)

        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_tcn_out_dim,
                                                        transformer_d_model) if sensor_tcn_out_dim != transformer_d_model else nn.Identity()
        self.inter_sensor_transformer = InterSensorTransformer(transformer_d_model, transformer_nhead,
                                                               transformer_nlayers, max_sensors)

        self.experts_forecast = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_forecast = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.experts_fail = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_fail = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.experts_rca = nn.ModuleList(
            [Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts_per_task)])
        self.gating_rca = GatingNetwork(moe_global_input_dim, num_experts_per_task)
        self.expert_dropout = nn.Dropout(expert_dropout_rate)  # Dropout is off during model.eval()

        final_combined_feat_dim_last_step = sensor_tcn_out_dim + transformer_d_model
        self.pred_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, 1)

    def _apply_moe_switch(self, global_moe_input, gating_network, expert_pool):
        gating_logits = gating_network(global_moe_input)
        # router_probs = torch.softmax(gating_logits, dim=-1) # Not needed for inference if aux loss not computed
        chosen_expert_indices = torch.argmax(gating_logits, dim=-1)
        one_hot_selection = F.one_hot(chosen_expert_indices, num_classes=len(expert_pool)).float()
        all_expert_outputs = torch.stack([expert(global_moe_input) for expert in expert_pool], dim=1)
        moe_task_output = torch.sum(all_expert_outputs * one_hot_selection.unsqueeze(-1), dim=1)
        # Expert dropout is automatically handled by model.eval()
        if self.training:  # This check ensures dropout is only applied during training
            moe_task_output = self.expert_dropout(moe_task_output)
        # For inference, fi and Pi are not needed for the main output
        return moe_task_output  # Removed fi, Pi from return for pure inference

    def forward(self, x_features_orig_scale, sensor_mask):
        batch_size, seq_len, _ = x_features_orig_scale.shape
        x_revin_input = x_features_orig_scale.permute(0, 2, 1) * sensor_mask.unsqueeze(-1);
        x_norm_revin = self.revin_layer(x_revin_input, mode='norm')
        last_val_norm_for_delta_recon = x_norm_revin[:, :, -1]
        x_norm_for_encoder_input = x_norm_revin.permute(0, 2, 1).unsqueeze(-1)
        x_reshaped_for_encoder = x_norm_for_encoder_input.reshape(batch_size * self.max_sensors, seq_len,
                                                                  SENSOR_INPUT_DIM)

        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, self.max_sensors, seq_len,
                                                                         SENSOR_TCN_OUT_DIM)
        sensor_temporal_features = sensor_temporal_features * sensor_mask.view(batch_size, self.max_sensors, 1, 1)

        pooled_sensor_features = torch.mean(sensor_temporal_features, dim=2)
        projected_for_inter_sensor = self.pooled_to_transformer_dim_proj(pooled_sensor_features)
        transformer_padding_mask = (sensor_mask == 0)
        cross_sensor_context = self.inter_sensor_transformer(projected_for_inter_sensor, transformer_padding_mask)
        cross_sensor_context = cross_sensor_context * sensor_mask.unsqueeze(-1)
        expanded_cross_sensor_context = cross_sensor_context.unsqueeze(2).expand(-1, -1, seq_len, TRANSFORMER_D_MODEL)
        final_combined_features = torch.cat([sensor_temporal_features, expanded_cross_sensor_context], dim=-1)

        global_moe_input_sum = (cross_sensor_context * sensor_mask.unsqueeze(-1)).sum(dim=1)
        active_sensors_per_batch = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_moe_input = global_moe_input_sum / active_sensors_per_batch

        # In eval mode, fi and Pi from _apply_moe_switch are not needed for aux loss
        moe_forecast_output = self._apply_moe_switch(global_moe_input, self.gating_forecast, self.experts_forecast)[0]
        moe_fail_output = self._apply_moe_switch(global_moe_input, self.gating_fail, self.experts_fail)[0]
        moe_rca_output = self._apply_moe_switch(global_moe_input, self.gating_rca, self.experts_rca)[0]

        final_features_last_step = final_combined_features[:, :, -1, :]
        moe_f_exp = moe_forecast_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        pred_head_in = torch.cat([final_features_last_step, moe_f_exp], dim=-1)
        pred_delta_norm = self.pred_head(pred_head_in)
        pred_abs_norm = last_val_norm_for_delta_recon.unsqueeze(-1) + pred_delta_norm
        pred_abs_denorm = self.revin_layer(pred_abs_norm, mode='denorm') * sensor_mask.unsqueeze(-1)

        fail_logits = self.fail_head(moe_fail_output)
        moe_r_exp = moe_rca_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        rca_head_in = torch.cat([final_features_last_step, moe_r_exp], dim=-1)
        rca_logits = self.rca_head(rca_head_in).squeeze(-1)

        # For inference, only return main outputs
        return pred_abs_denorm, fail_logits, rca_logits


# --- Test Script Logic ---
def test_model():
    print(f"Using device: {DEVICE}")

    try:
        preprocessor_config = np.load(PREPROCESSOR_LOAD_PATH)
        max_sensors_overall = int(preprocessor_config['max_sensors_overall'])
    except FileNotFoundError:
        print(f"Error: Preprocessor config file not found at {PREPROCESSOR_LOAD_PATH}. Exiting.")
        return
    except KeyError:
        print(f"Error: 'max_sensors_overall' not found in preprocessor config. Exiting.")
        return

    print(f"Loaded max_sensors_overall from preprocessor: {max_sensors_overall}")

    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, max_sensors_overall)
    if len(valid_dataset) == 0:
        print(f"No data found in {VALID_DIR} or no valid windows created. Exiting.")
        return
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FoundationalTimeSeriesModelV6(
        max_sensors=max_sensors_overall, seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL,
        transformer_nhead=TRANSFORMER_NHEAD, transformer_nlayers=TRANSFORMER_NLAYERS,
        moe_global_input_dim=MOE_GLOBAL_INPUT_DIM, num_experts_per_task=NUM_EXPERTS_PER_TASK,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        expert_dropout_rate=EXPERT_DROPOUT_RATE,
        pred_horizons_len=len(PRED_HORIZONS), fail_horizons_len=len(FAIL_HORIZONS)
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_LOAD_PATH}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading model state: {e}. Ensure model architecture and saved keys match. Exiting.")
        return

    model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            if batch_idx >= MAX_BATCHES_TO_TEST:
                print(f"\nReached MAX_BATCHES_TO_TEST ({MAX_BATCHES_TO_TEST}). Stopping inference.")
                break

            print(f"\n--- Processing Batch {batch_idx + 1} ---")
            input_features_orig_scale = batch["input_features"].to(DEVICE)
            sensor_mask = batch["sensor_mask"].to(DEVICE)
            last_known_values_orig_scale = batch["last_known_values"].to(DEVICE)
            pred_delta_targets_orig_scale = batch["pred_delta_targets"].to(DEVICE)
            fail_targets_gt = batch["fail_targets"].to(DEVICE)
            rca_targets_gt = batch["rca_targets"].to(DEVICE)

            batch_s = input_features_orig_scale.size(0)

            # Model's forward pass now returns only main outputs during eval
            pred_absolute_final_denorm, fail_output_logits, rca_output_logits = model(input_features_orig_scale,
                                                                                      sensor_mask)

            actual_pred_abs_targets = last_known_values_orig_scale.unsqueeze(-1) + pred_delta_targets_orig_scale

            num_samples_this_batch_to_print = min(SAMPLES_PER_BATCH_TO_PRINT, batch_s)
            for i in range(num_samples_this_batch_to_print):
                sample_filepath = batch["filepath"][i]
                sample_window_start = batch["window_start_idx"][i].item()

                print(
                    f"\n  --- Sample {i + 1} from Batch {batch_idx + 1} (File: {os.path.basename(sample_filepath)}, Window Start: {sample_window_start}) ---")
                active_sensor_indices = torch.where(sensor_mask[i] == 1.0)[0].cpu().tolist()
                num_active_sensors_sample = len(active_sensor_indices)

                if num_active_sensors_sample == 0:
                    print("    No active sensors in this sample.")
                    continue

                print("    Forecasting Task (Absolute Values):")
                for h_idx, horizon_steps in enumerate(PRED_HORIZONS):
                    print(f"      Horizon: +{horizon_steps} steps")
                    sensors_to_print_forecast = active_sensor_indices[:min(2, num_active_sensors_sample)]
                    if not sensors_to_print_forecast: print("        (No active sensors to show forecast for)")
                    for s_global_idx in sensors_to_print_forecast:
                        actual_absolute_gt = actual_pred_abs_targets[i, s_global_idx, h_idx].item()
                        predicted_absolute = pred_absolute_final_denorm[i, s_global_idx, h_idx].item()
                        print(
                            f"        Sensor {s_global_idx + 1}: Actual Abs={actual_absolute_gt:.4f}, Predicted Abs={predicted_absolute:.4f}")

                print("    Failure Prediction Task:")
                predicted_fail_probs = torch.sigmoid(fail_output_logits[i]).cpu().tolist()
                actual_fail_status = fail_targets_gt[i].cpu().tolist()
                for h_idx, horizon_steps in enumerate(FAIL_HORIZONS):
                    print(
                        f"      Horizon: within next {horizon_steps} steps - Actual: {actual_fail_status[h_idx]:.0f}, Predicted Prob: {predicted_fail_probs[h_idx]:.4f}")

                print("    Root Cause Analysis Task:")
                idx_rca_horizon = FAIL_HORIZONS.index(RCA_FAILURE_LOOKAHEAD)
                rca_condition_met = predicted_fail_probs[idx_rca_horizon] > 0.6 or actual_fail_status[
                    idx_rca_horizon] == 1.0

                if rca_condition_met:
                    rca_scores_sample_all_sensors = torch.sigmoid(rca_output_logits[i]).cpu().tolist()
                    active_rca_scores = []
                    for s_global_idx in active_sensor_indices:
                        active_rca_scores.append((s_global_idx + 1, rca_scores_sample_all_sensors[s_global_idx]))
                    active_rca_scores.sort(key=lambda x: x[1], reverse=True)
                    print(f"      (RCA condition met for {RCA_FAILURE_LOOKAHEAD}-step horizon)")
                    print(f"      Top contributing sensors (Sensor #, RCA Score):")
                    if not active_rca_scores: print("        (No active sensors for RCA)")
                    for s_id, score in active_rca_scores[:min(5, len(active_rca_scores))]:
                        actual_rca_target_for_sensor = rca_targets_gt[i, s_id - 1].item()
                        print(
                            f"        Sensor {s_id}: Score={score:.4f} (Actual RCA Target: {actual_rca_target_for_sensor:.0f})")
                else:
                    print(f"      (RCA condition not met for this sample for {RCA_FAILURE_LOOKAHEAD}-step horizon)")

    print("\n--- Testing complete ---")


if __name__ == '__main__':
    test_model()