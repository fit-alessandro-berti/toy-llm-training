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

# --- Configuration (MUST MATCH V3 TRAINING SCRIPT) ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # Test on VALIDATION data

# Model & Training Parameters (from V3 training script)
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]
FAIL_HORIZONS = [3, 5, 10]
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0]
MAX_SENSORS_CAP = 20

# Architectural Params (from V3 training script)
SENSOR_INPUT_DIM = 1
SENSOR_CNN_PROJ_DIM = 16
SENSOR_CNN_OUT_DIM = 32
CNN_KERNEL_SIZE = 3
CNN_LAYERS = 3
CNN_DILATION_BASE = 2

TRANSFORMER_D_MODEL = 64  # Core operational dimension of InterSensorTransformer
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

MOE_GLOBAL_INPUT_DIM = TRANSFORMER_D_MODEL
MOE_HIDDEN_DIM_EXPERT = 128
NUM_EXPERTS = 4
MOE_OUTPUT_DIM = 64
MOE_TOP_K = 2

# Testing Params
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCHES_TO_TEST = 5
SAMPLES_PER_BATCH_TO_PRINT = 2

MODEL_LOAD_PATH = "foundation_timeseries_model_v3.pth"  # Updated path
PREPROCESSOR_LOAD_PATH = "preprocessor_config_v3.npz"  # Updated path


# --- Helper: Positional Encoding (from V3 training script) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [Batch, SeqLen, Dim]
        return x + self.pe[:, :x.size(1), :]


# --- Helper: CausalConv1D Block (from V3 training script) ---
class CausalConv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):  # x: [Batch, Channels, SeqLen]
        x = self.conv1(x)
        x = x[:, :, :-self.padding] if self.padding > 0 else x
        x = self.relu(x)
        return x


# --- Data Handling (Replicated from V3 training script) ---
def get_max_sensors_from_files(file_paths):
    max_s = 0
    for fp in file_paths:
        try:
            df_peek = pd.read_csv(fp, nrows=1)
            num_cols = df_peek.shape[1]
            max_s = max(max_s, num_cols - 1)
        except Exception:
            continue
    return min(max_s, MAX_SENSORS_CAP)


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_horizons, fail_horizons, rca_failure_lookahead, max_sensors_global):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons
        self.fail_horizons = fail_horizons
        self.rca_failure_lookahead = rca_failure_lookahead
        self.max_sensors_global = max_sensors_global

        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
        self.data_cache = []
        self.window_indices = []
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        print(f"Loading data from {self.data_dir}...")
        for file_idx, fp in enumerate(self.file_paths):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                print(f"Warning: Skipping file {fp} due to loading error: {e}")
                continue
            sensor_cols = [col for col in df.columns if col.startswith("Sensor")]
            if not sensor_cols: continue
            if len(sensor_cols) > self.max_sensors_global:
                sensor_cols = sensor_cols[:self.max_sensors_global]

            features = df[sensor_cols].values.astype(np.float32)
            failure_flags = df["CURRENT_FAILURE"].values.astype(np.int64)
            num_actual_sensors = features.shape[1]

            scalers = [StandardScaler() for _ in range(num_actual_sensors)]
            for i in range(num_actual_sensors):
                if features.shape[0] > 0:
                    features[:, i] = scalers[i].fit_transform(features[:, i].reshape(-1, 1)).flatten()
            self.data_cache.append({
                "features": features,
                "failure_flags": failure_flags,
                "num_actual_sensors": num_actual_sensors,
                "filepath": fp
            })
            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons), self.rca_failure_lookahead)
            for i in range(len(df) - self.seq_len - max_lookahead + 1):
                self.window_indices.append((file_idx, i))
        print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        cached_item = self.data_cache[file_idx]
        features_full, failure_flags_full, num_actual_sensors = cached_item["features"], cached_item["failure_flags"], \
        cached_item["num_actual_sensors"]
        filepath = cached_item["filepath"]

        input_seq_features_orig = features_full[window_start_idx: window_start_idx + self.seq_len]

        last_known_values = np.zeros(self.max_sensors_global, dtype=np.float32)
        if input_seq_features_orig.shape[0] > 0:
            last_known_values[:num_actual_sensors] = input_seq_features_orig[-1, :num_actual_sensors]

        padded_input_seq_features = np.zeros((self.seq_len, self.max_sensors_global), dtype=np.float32)
        padded_input_seq_features[:, :num_actual_sensors] = input_seq_features_orig

        sensor_mask = np.zeros(self.max_sensors_global, dtype=np.float32)
        sensor_mask[:num_actual_sensors] = 1.0

        pred_delta_targets = np.zeros((self.max_sensors_global, len(self.pred_horizons)), dtype=np.float32)
        for i, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < len(features_full):
                future_values = features_full[target_idx, :num_actual_sensors]
                pred_delta_targets[:num_actual_sensors, i] = future_values - last_known_values[:num_actual_sensors]

        fail_targets = np.zeros(len(self.fail_horizons), dtype=np.float32)
        for i, n in enumerate(self.fail_horizons):
            fail_window_start = window_start_idx + self.seq_len
            fail_window_end = fail_window_start + n
            if fail_window_end <= len(failure_flags_full):
                if np.any(failure_flags_full[fail_window_start:fail_window_end]):
                    fail_targets[i] = 1.0

        rca_targets = np.zeros(self.max_sensors_global, dtype=np.float32)
        rca_fail_window_start = window_start_idx + self.seq_len
        rca_fail_window_end = rca_fail_window_start + self.rca_failure_lookahead
        is_imminent_failure = False
        if rca_fail_window_end <= len(failure_flags_full) and np.any(
                failure_flags_full[rca_fail_window_start:rca_fail_window_end]):
            is_imminent_failure = True

        if is_imminent_failure and num_actual_sensors > 0:
            failure_sub_window_features = features_full[rca_fail_window_start:rca_fail_window_end, :num_actual_sensors]
            if failure_sub_window_features.shape[0] > 0:
                current_input_window_features = features_full[window_start_idx:window_start_idx + self.seq_len,
                                                :num_actual_sensors]
                if current_input_window_features.shape[0] > 0:
                    input_means = np.mean(current_input_window_features, axis=0)
                    input_stds = np.std(current_input_window_features, axis=0)
                    input_stds[input_stds < 1e-6] = 1e-6
                    for s_idx in range(num_actual_sensors):
                        if np.any(np.abs(failure_sub_window_features[:, s_idx] - input_means[s_idx]) > 3 * input_stds[
                            s_idx]):
                            rca_targets[s_idx] = 1.0

        return {
            "input_features": torch.from_numpy(padded_input_seq_features),
            "sensor_mask": torch.from_numpy(sensor_mask),
            "last_known_values": torch.from_numpy(last_known_values),
            "pred_delta_targets": torch.from_numpy(pred_delta_targets),
            "fail_targets": torch.from_numpy(fail_targets),
            "rca_targets": torch.from_numpy(rca_targets),
            "filepath": filepath,
            "window_start_idx": window_start_idx
        }


# --- Model Architecture (Replicated from V3 training script) ---
class PerSensorEncoderCNN(nn.Module):
    def __init__(self, input_dim, proj_dim, cnn_out_dim, seq_len, num_layers, kernel_size, dilation_base):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=seq_len)

        self.manual_cnn_layers = nn.ModuleList()
        current_dim_manual = proj_dim
        for i in range(num_layers):
            out_d = cnn_out_dim if i == num_layers - 1 else proj_dim
            self.manual_cnn_layers.append(CausalConv1DBlock(current_dim_manual, out_d, kernel_size, dilation_base ** i))
            current_dim_manual = out_d
        self.final_norm = nn.LayerNorm(cnn_out_dim)

    def forward(self, x):  # x: [Batch*MaxSensors, SeqLen, InputDim=1]
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        x = x.permute(0, 2, 1)
        for layer in self.manual_cnn_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)
        x = self.final_norm(x)
        return x


class InterSensorTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, max_sensors):  # embed_dim is d_model
        super().__init__()
        self.pos_encoder_inter_sensor = nn.Parameter(torch.zeros(1, max_sensors, embed_dim))
        # norm_first=True is good for stability with higher LRs
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True,
                                                   dim_feedforward=embed_dim * 2, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask):
        x = x + self.pos_encoder_inter_sensor[:, :x.size(1), :]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.output_norm(x)
        return x


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


class FoundationalTimeSeriesModelV2(nn.Module):  # Keeping class name for consistency with saved state_dict keys
    def __init__(self, max_sensors, seq_len,
                 sensor_input_dim, sensor_cnn_proj_dim, sensor_cnn_out_dim, cnn_layers, cnn_kernel_size,
                 cnn_dilation_base,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 moe_global_input_dim, moe_hidden_dim_expert, num_experts, moe_output_dim, moe_top_k,
                 pred_horizons_len, fail_horizons_len):
        super().__init__()
        self.max_sensors = max_sensors
        self.seq_len = seq_len
        self.moe_top_k = moe_top_k
        self.moe_output_dim = moe_output_dim

        self.per_sensor_encoder = PerSensorEncoderCNN(
            sensor_input_dim, sensor_cnn_proj_dim, sensor_cnn_out_dim, seq_len,
            cnn_layers, cnn_kernel_size, cnn_dilation_base
        )

        if sensor_cnn_out_dim != transformer_d_model:
            self.pooled_to_transformer_dim_proj = nn.Linear(sensor_cnn_out_dim, transformer_d_model)
        else:
            self.pooled_to_transformer_dim_proj = nn.Identity()

        self.inter_sensor_transformer = InterSensorTransformer(
            embed_dim=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_nlayers,
            max_sensors=max_sensors
        )

        self.experts = nn.ModuleList([
            Expert(moe_global_input_dim, moe_hidden_dim_expert, moe_output_dim) for _ in range(num_experts)
        ])
        self.gating_forecast = GatingNetwork(moe_global_input_dim, num_experts)
        self.gating_fail = GatingNetwork(moe_global_input_dim, num_experts)
        self.gating_rca = GatingNetwork(moe_global_input_dim, num_experts)

        final_combined_feat_dim_last_step = sensor_cnn_out_dim + transformer_d_model

        self.pred_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, 1)

    def _apply_moe(self, global_moe_input, gating_network, expert_outputs_all):
        gating_logits = gating_network(global_moe_input)
        top_k_weights, top_k_indices = torch.topk(gating_logits, self.moe_top_k, dim=-1)
        top_k_gates = torch.softmax(top_k_weights, dim=-1)

        batch_idx_for_gather = torch.arange(expert_outputs_all.size(0), device=expert_outputs_all.device).unsqueeze(
            -1).expand_as(top_k_indices)
        chosen_expert_outputs = expert_outputs_all[batch_idx_for_gather, top_k_indices]
        moe_task_output = torch.sum(chosen_expert_outputs * top_k_gates.unsqueeze(-1), dim=1)
        return moe_task_output

    def forward(self, x_features, sensor_mask):
        batch_size, seq_len, _ = x_features.shape

        x_permuted = x_features.permute(0, 2, 1).unsqueeze(-1)
        x_reshaped = x_permuted.reshape(batch_size * self.max_sensors, seq_len, SENSOR_INPUT_DIM)

        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped)
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, self.max_sensors, seq_len,
                                                                         SENSOR_CNN_OUT_DIM)
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

        expert_outputs_all = torch.stack([exp(global_moe_input) for exp in self.experts], dim=1)

        moe_forecast_output = self._apply_moe(global_moe_input, self.gating_forecast, expert_outputs_all)
        moe_fail_output = self._apply_moe(global_moe_input, self.gating_fail, expert_outputs_all)
        moe_rca_output = self._apply_moe(global_moe_input, self.gating_rca, expert_outputs_all)

        final_features_last_step = final_combined_features[:, :, -1, :]

        moe_forecast_expanded = moe_forecast_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        pred_head_input = torch.cat([final_features_last_step, moe_forecast_expanded], dim=-1)
        pred_delta_output = self.pred_head(pred_head_input)

        fail_output_logits = self.fail_head(moe_fail_output)

        moe_rca_expanded = moe_rca_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        rca_head_input = torch.cat([final_features_last_step, moe_rca_expanded], dim=-1)
        rca_output_logits = self.rca_head(rca_head_input).squeeze(-1)

        return pred_delta_output, fail_output_logits, rca_output_logits


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

    model = FoundationalTimeSeriesModelV2(  # Ensure this is the V3 structure
        max_sensors=max_sensors_overall, seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_cnn_proj_dim=SENSOR_CNN_PROJ_DIM,
        sensor_cnn_out_dim=SENSOR_CNN_OUT_DIM,
        cnn_layers=CNN_LAYERS, cnn_kernel_size=CNN_KERNEL_SIZE, cnn_dilation_base=CNN_DILATION_BASE,
        transformer_d_model=TRANSFORMER_D_MODEL,
        transformer_nhead=TRANSFORMER_NHEAD, transformer_nlayers=TRANSFORMER_NLAYERS,
        moe_global_input_dim=MOE_GLOBAL_INPUT_DIM, moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT,
        num_experts=NUM_EXPERTS, moe_output_dim=MOE_OUTPUT_DIM, moe_top_k=MOE_TOP_K,
        pred_horizons_len=len(PRED_HORIZONS), fail_horizons_len=len(FAIL_HORIZONS)
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_LOAD_PATH}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading model state: {e}. Ensure model architecture matches saved state. Exiting.")
        return

    model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            if batch_idx >= MAX_BATCHES_TO_TEST:
                print(f"\nReached MAX_BATCHES_TO_TEST ({MAX_BATCHES_TO_TEST}). Stopping inference.")
                break

            print(f"\n--- Processing Batch {batch_idx + 1} ---")
            input_features = batch["input_features"].to(DEVICE)
            sensor_mask = batch["sensor_mask"].to(DEVICE)
            last_known_values_batch = batch["last_known_values"].to(DEVICE)
            pred_delta_targets_gt = batch["pred_delta_targets"].to(DEVICE)
            fail_targets_gt = batch["fail_targets"].to(DEVICE)
            rca_targets_gt = batch["rca_targets"].to(DEVICE)

            batch_s = input_features.size(0)

            pred_delta_output, fail_output_logits, rca_output_logits = model(input_features, sensor_mask)

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

                last_known_values_sample = last_known_values_batch[i]

                print("    Forecasting Task (Absolute Values):")
                for h_idx, horizon_steps in enumerate(PRED_HORIZONS):
                    print(f"      Horizon: +{horizon_steps} steps")
                    sensors_to_print_forecast = active_sensor_indices[:min(2, num_active_sensors_sample)]
                    if not sensors_to_print_forecast: print("        (No active sensors to show forecast for)")
                    for s_global_idx in sensors_to_print_forecast:
                        actual_delta_gt = pred_delta_targets_gt[i, s_global_idx, h_idx].item()
                        actual_absolute_gt = last_known_values_sample[s_global_idx].item() + actual_delta_gt

                        predicted_delta = pred_delta_output[i, s_global_idx, h_idx].item()
                        predicted_absolute = last_known_values_sample[s_global_idx].item() + predicted_delta

                        print(
                            f"        Sensor {s_global_idx + 1}: Actual Abs={actual_absolute_gt:.4f}, Predicted Abs={predicted_absolute:.4f} (Pred Δ={predicted_delta:.4f})")

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
