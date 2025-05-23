import os
import glob
import random
import math  # For positional encoding and scheduler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast  # For Mixed Precision

# --- Configuration ---
# Data paths
BASE_DATA_DIR = "../../data/time_series/1"
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAINING")
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")

# Model & Training Parameters
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]
FAIL_HORIZONS = [3, 5, 10]
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0]
MAX_SENSORS_CAP = 20

# Architectural Params
SENSOR_INPUT_DIM = 1
SENSOR_CNN_PROJ_DIM = 16
SENSOR_CNN_OUT_DIM = 32
CNN_KERNEL_SIZE = 3
CNN_LAYERS = 3
CNN_DILATION_BASE = 2

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

# MoE V5 Specific Params
MOE_GLOBAL_INPUT_DIM = TRANSFORMER_D_MODEL
NUM_EXPERTS_PER_TASK = 8  # Example: use 8 or 16 or 32 as suggested by user
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64
# MOE_TOP_K is effectively 1 for Switch Router
EXPERT_DROPOUT_RATE = 0.1
AUX_LOSS_COEFF = 0.01  # Coefficient for the load balancing loss

# Training Params
BATCH_SIZE = 32
EPOCHS = 3  # Might need more or less depending on convergence
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == 'cuda'

MODEL_SAVE_PATH = "foundation_timeseries_model_v5.pth"
PREPROCESSOR_SAVE_PATH = "preprocessor_config_v5.npz"

HUBER_DELTA = 1.0
FOCAL_ALPHA_PARAM = 0.25
FOCAL_GAMMA = 2.0

JITTER_STRENGTH_RATIO = 0.03
MAG_WARP_STRENGTH_RATIO = 0.05
MAG_WARP_KNOTS = 4
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.4


# --- RevIN Layer ---
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
                return x  # Safety for uninitialized stats

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


# --- Helper: Positional Encoding ---
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

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# --- Helper: CausalConv1D Block ---
class CausalConv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-self.padding] if self.padding > 0 else x
        x = self.relu(x)
        return x


# --- Helper: Focal Loss ---
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
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# --- Data Handling ---
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
            self.data_cache.append(
                {"features": features, "failure_flags": failure_flags, "num_actual_sensors": num_actual_sensors})
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
        input_seq_features_orig = features_full[window_start_idx: window_start_idx + self.seq_len]
        last_known_values = np.zeros(self.max_sensors_global, dtype=np.float32)
        if input_seq_features_orig.shape[0] > 0: last_known_values[:num_actual_sensors] = input_seq_features_orig[-1,
                                                                                          :num_actual_sensors]
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
            if fail_window_end <= len(failure_flags_full) and np.any(
                    failure_flags_full[fail_window_start:fail_window_end]):
                fail_targets[i] = 1.0
        rca_targets = np.zeros(self.max_sensors_global, dtype=np.float32)
        rca_fail_window_start = window_start_idx + self.seq_len
        rca_fail_window_end = rca_fail_window_start + self.rca_failure_lookahead
        if rca_fail_window_end <= len(failure_flags_full) and np.any(
                failure_flags_full[rca_fail_window_start:rca_fail_window_end]) and num_actual_sensors > 0:
            failure_sub_window_features = features_full[rca_fail_window_start:rca_fail_window_end, :num_actual_sensors]
            if failure_sub_window_features.shape[0] > 0:
                current_input_window_features = features_full[window_start_idx:window_start_idx + self.seq_len,
                                                :num_actual_sensors]
                if current_input_window_features.shape[0] > 0:
                    input_means, input_stds = np.mean(current_input_window_features, axis=0), np.std(
                        current_input_window_features, axis=0)
                    input_stds[input_stds < 1e-6] = 1e-6
                    for s_idx in range(num_actual_sensors):
                        if np.any(np.abs(failure_sub_window_features[:, s_idx] - input_means[s_idx]) > 3 * input_stds[
                            s_idx]):
                            rca_targets[s_idx] = 1.0
        return {"input_features": torch.from_numpy(padded_input_seq_features),
                "sensor_mask": torch.from_numpy(sensor_mask),
                "last_known_values": torch.from_numpy(last_known_values),
                "pred_delta_targets": torch.from_numpy(pred_delta_targets),
                "fail_targets": torch.from_numpy(fail_targets), "rca_targets": torch.from_numpy(rca_targets)}


# --- Model Architecture (V5) ---
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

    def forward(self, x):
        x = self.input_proj(x);
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)
        for layer in self.manual_cnn_layers: x = layer(x)
        x = x.permute(0, 2, 1);
        x = self.final_norm(x)
        return x


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

    def forward(self, x): return self.fc(x)  # Return logits


class FoundationalTimeSeriesModelV5(nn.Module):
    def __init__(self, max_sensors, seq_len,
                 sensor_input_dim, sensor_cnn_proj_dim, sensor_cnn_out_dim, cnn_layers, cnn_kernel_size,
                 cnn_dilation_base,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 moe_global_input_dim, num_experts_per_task, moe_hidden_dim_expert, moe_output_dim, expert_dropout_rate,
                 pred_horizons_len, fail_horizons_len, revin_affine=True):
        super().__init__()
        self.max_sensors = max_sensors
        self.seq_len = seq_len
        self.num_experts_per_task = num_experts_per_task
        self.moe_output_dim = moe_output_dim
        self.revin_layer = RevIN(num_features=max_sensors, affine=revin_affine)
        self.per_sensor_encoder = PerSensorEncoderCNN(sensor_input_dim, sensor_cnn_proj_dim, sensor_cnn_out_dim,
                                                      seq_len, cnn_layers, cnn_kernel_size, cnn_dilation_base)
        self.pooled_to_transformer_dim_proj = nn.Linear(sensor_cnn_out_dim,
                                                        transformer_d_model) if sensor_cnn_out_dim != transformer_d_model else nn.Identity()
        self.inter_sensor_transformer = InterSensorTransformer(transformer_d_model, transformer_nhead,
                                                               transformer_nlayers, max_sensors)

        # Task-specific MoE components
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

        final_combined_feat_dim_last_step = sensor_cnn_out_dim + transformer_d_model
        self.pred_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, 1)

    def _apply_moe_switch(self, global_moe_input, gating_network, expert_pool):
        batch_size = global_moe_input.size(0)
        gating_logits = gating_network(global_moe_input)  # [B, NumExpertsPerTask]
        router_probs = torch.softmax(gating_logits, dim=-1)  # For aux loss: [B, NumExpertsPerTask]

        chosen_expert_indices = torch.argmax(gating_logits, dim=-1)  # [B], k=1 switch routing
        one_hot_selection = F.one_hot(chosen_expert_indices,
                                      num_classes=len(expert_pool)).float()  # [B, NumExpertsPerTask]

        # Compute all expert outputs (for simplicity in PyTorch without custom kernels)
        # This is not computationally efficient like a true Switch Transformer's sparse dispatch
        all_expert_outputs = torch.stack([expert(global_moe_input) for expert in expert_pool],
                                         dim=1)  # [B, NumExp, MoeOutDim]

        # Select the output of the chosen expert
        moe_task_output = torch.sum(all_expert_outputs * one_hot_selection.unsqueeze(-1), dim=1)  # [B, MoeOutDim]

        if self.training:
            moe_task_output = self.expert_dropout(moe_task_output)

        # For auxiliary load balancing loss
        fi = one_hot_selection.mean(dim=0)  # Fraction of batch dispatched to each expert [NumExpertsPerTask]
        Pi = router_probs.mean(dim=0)  # Average router probability for each expert [NumExpertsPerTask]

        return moe_task_output, fi, Pi

    def forward(self, x_features_orig_scale, sensor_mask):
        batch_size, seq_len, _ = x_features_orig_scale.shape
        x_revin_input = x_features_orig_scale.permute(0, 2, 1) * sensor_mask.unsqueeze(-1)
        x_norm_revin = self.revin_layer(x_revin_input, mode='norm')
        last_val_norm_for_delta_recon = x_norm_revin[:, :, -1]
        x_norm_for_encoder_input = x_norm_revin.permute(0, 2, 1).unsqueeze(-1)
        x_reshaped_for_encoder = x_norm_for_encoder_input.reshape(batch_size * self.max_sensors, seq_len,
                                                                  SENSOR_INPUT_DIM)

        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
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

        # Task-specific MoE processing
        moe_forecast_output, fi_forecast, Pi_forecast = self._apply_moe_switch(global_moe_input, self.gating_forecast,
                                                                               self.experts_forecast)
        moe_fail_output, fi_fail, Pi_fail = self._apply_moe_switch(global_moe_input, self.gating_fail,
                                                                   self.experts_fail)
        moe_rca_output, fi_rca, Pi_rca = self._apply_moe_switch(global_moe_input, self.gating_rca, self.experts_rca)

        aux_loss_terms = {
            "forecast": (fi_forecast, Pi_forecast),
            "fail": (fi_fail, Pi_fail),
            "rca": (fi_rca, Pi_rca)
        }

        final_features_last_step = final_combined_features[:, :, -1, :]
        moe_forecast_expanded = moe_forecast_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        pred_head_input = torch.cat([final_features_last_step, moe_forecast_expanded], dim=-1)
        pred_delta_output_norm = self.pred_head(pred_head_input)

        pred_absolute_norm = last_val_norm_for_delta_recon.unsqueeze(-1) + pred_delta_output_norm
        pred_absolute_final_denorm = self.revin_layer(pred_absolute_norm, mode='denorm')
        pred_absolute_final_denorm = pred_absolute_final_denorm * sensor_mask.unsqueeze(-1)

        fail_output_logits = self.fail_head(moe_fail_output)
        moe_rca_expanded = moe_rca_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        rca_head_input = torch.cat([final_features_last_step, moe_rca_expanded], dim=-1)
        rca_output_logits = self.rca_head(rca_head_input).squeeze(-1)

        return pred_absolute_final_denorm, fail_output_logits, rca_output_logits, aux_loss_terms


# --- Augmentation Functions ---
def augment_jitter(x_batch, sensor_mask, strength_ratio=0.03):
    if strength_ratio == 0: return x_batch
    x_aug = x_batch.clone()
    noise = torch.normal(mean=0., std=strength_ratio, size=x_batch.shape,
                         device=x_batch.device)  # Simplified: fixed std relative to expected normalized data range
    x_aug = x_aug + noise
    return x_aug * sensor_mask.unsqueeze(1)


def augment_magnitude_warp(x_batch, sensor_mask, strength_ratio=0.05, num_knots=4):
    if strength_ratio == 0 or num_knots <= 1: return x_batch
    x_aug = x_batch.clone()
    B, SL, MS = x_batch.shape
    for i in range(B):
        if sensor_mask[i].sum() == 0: continue
        t_knots = torch.linspace(0, SL - 1, num_knots, device=x_batch.device)
        y_knots = torch.normal(mean=1.0, std=strength_ratio, size=(num_knots,), device=x_batch.device)
        x_coords = torch.arange(SL, device=x_batch.device).float()
        try:  # np.interp might be slow, but robust. For pure torch, consider a simpler scheme or library
            warp_curve_np = np.interp(x_coords.cpu().numpy(), t_knots.cpu().numpy(), y_knots.cpu().numpy())
            warp_curve = torch.from_numpy(warp_curve_np).float().to(x_batch.device).unsqueeze(-1)
            active_sensor_indices_sample_i = sensor_mask[i] == 1.0
            x_aug[i, :, active_sensor_indices_sample_i] *= warp_curve
        except Exception as e:
            # print(f"Magnitude warping interp failed: {e}") # Potentially too verbose
            pass  # Skip warping for this sample if interp fails
    return x_aug


# --- Training Loop ---
def train_model():
    print(f"Using device: {DEVICE}, AMP enabled: {AMP_ENABLED}")
    all_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv")) + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_files: print("No CSV files found. Exiting."); return
    max_sensors_overall = get_max_sensors_from_files(all_files)
    if max_sensors_overall == 0: print("Could not determine max_sensors_overall. Exiting."); return
    print(f"Max sensors overall (capped at {MAX_SENSORS_CAP}): {max_sensors_overall}")
    np.savez(PREPROCESSOR_SAVE_PATH, max_sensors_overall=max_sensors_overall)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, max_sensors_overall)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, max_sensors_overall)
    if len(train_dataset) == 0: print("Training dataset empty. Exiting."); return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=AMP_ENABLED)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=AMP_ENABLED) if len(valid_dataset) > 0 else None

    model = FoundationalTimeSeriesModelV5(  # Using V5
        max_sensors=max_sensors_overall, seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_cnn_proj_dim=SENSOR_CNN_PROJ_DIM,
        sensor_cnn_out_dim=SENSOR_CNN_OUT_DIM,
        cnn_layers=CNN_LAYERS, cnn_kernel_size=CNN_KERNEL_SIZE, cnn_dilation_base=CNN_DILATION_BASE,
        transformer_d_model=TRANSFORMER_D_MODEL,
        transformer_nhead=TRANSFORMER_NHEAD, transformer_nlayers=TRANSFORMER_NLAYERS,
        moe_global_input_dim=MOE_GLOBAL_INPUT_DIM, num_experts_per_task=NUM_EXPERTS_PER_TASK,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        expert_dropout_rate=EXPERT_DROPOUT_RATE,  # moe_top_k is implicitly 1
        pred_horizons_len=len(PRED_HORIZONS), fail_horizons_len=len(FAIL_HORIZONS)
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_steps)

    def lr_lambda_combined(current_step):
        if warmup_steps > 0 and current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_combined)
    scaler = GradScaler(enabled=AMP_ENABLED)
    huber_loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')
    focal_loss_elementwise = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA, reduction='none')
    focal_loss_mean = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA, reduction='mean')
    w_pred, w_fail, w_rca = 1.0, 1.0, 0.5

    for epoch in range(EPOCHS):
        model.train()
        total_loss_train, total_pred_loss_train, total_fail_loss_train, total_rca_loss_train, total_aux_loss_train = 0, 0, 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            input_features_orig = batch["input_features"].to(DEVICE)
            sensor_mask = batch["sensor_mask"].to(DEVICE)
            last_known_values_orig = batch["last_known_values"].to(DEVICE)
            pred_delta_targets_orig = batch["pred_delta_targets"].to(DEVICE)
            fail_targets_orig = batch["fail_targets"].to(DEVICE)
            rca_targets_orig = batch["rca_targets"].to(DEVICE)

            current_input_features = input_features_orig
            if JITTER_STRENGTH_RATIO > 0: current_input_features = augment_jitter(current_input_features, sensor_mask,
                                                                                  JITTER_STRENGTH_RATIO)
            if MAG_WARP_STRENGTH_RATIO > 0: current_input_features = augment_magnitude_warp(current_input_features,
                                                                                            sensor_mask,
                                                                                            MAG_WARP_STRENGTH_RATIO,
                                                                                            MAG_WARP_KNOTS)

            actual_pred_abs_targets = last_known_values_orig.unsqueeze(-1) + pred_delta_targets_orig

            if random.random() < MIXUP_PROB:
                lambda_mix = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                perm_indices = torch.randperm(current_input_features.size(0), device=DEVICE)
                mixed_input_features = lambda_mix * current_input_features + (1 - lambda_mix) * current_input_features[
                    perm_indices]
                mixed_pred_abs_targets = lambda_mix * actual_pred_abs_targets + (1 - lambda_mix) * \
                                         actual_pred_abs_targets[perm_indices]
                mixed_fail_targets = lambda_mix * fail_targets_orig + (1 - lambda_mix) * fail_targets_orig[perm_indices]
                mixed_rca_targets = lambda_mix * rca_targets_orig + (1 - lambda_mix) * rca_targets_orig[perm_indices]
                loss_input, loss_pred_t, loss_fail_t, loss_rca_t = mixed_input_features, mixed_pred_abs_targets, mixed_fail_targets, mixed_rca_targets
            else:
                loss_input, loss_pred_t, loss_fail_t, loss_rca_t = current_input_features, actual_pred_abs_targets, fail_targets_orig, rca_targets_orig

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=AMP_ENABLED):
                pred_abs_final_denorm, fail_output_logits, rca_output_logits, aux_loss_terms = model(loss_input,
                                                                                                     sensor_mask)
                loss_pred_unmasked = huber_loss_fn(pred_abs_final_denorm, loss_pred_t)
                active_pred_elements = sensor_mask.sum().clamp(min=1) * len(PRED_HORIZONS)
                loss_pred = (loss_pred_unmasked * sensor_mask.unsqueeze(-1)).sum() / active_pred_elements.clamp(
                    min=1e-9)
                loss_fail = focal_loss_mean(fail_output_logits, loss_fail_t)
                loss_rca_unmasked = focal_loss_elementwise(rca_output_logits, loss_rca_t)
                loss_rca = (loss_rca_unmasked * sensor_mask).sum() / sensor_mask.sum().clamp(min=1e-9)

                # Auxiliary MoE Load Balancing Loss
                aux_loss_total_batch = 0
                for task_name, (fi, Pi) in aux_loss_terms.items():
                    # Sum over experts: sum(fi * Pi)
                    # Number of experts for this task is fi.shape[0] (or Pi.shape[0])
                    num_experts_this_task = fi.size(0)
                    aux_loss_task = num_experts_this_task * torch.sum(
                        fi * Pi)  # As per Switch Transformer paper (N * sum(fi*Pi))
                    aux_loss_total_batch += aux_loss_task
                aux_loss_total_batch *= AUX_LOSS_COEFF

                combined_loss = w_pred * loss_pred + w_fail * loss_fail + w_rca * loss_rca + aux_loss_total_batch

            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(
                    f"NaN/Inf L. Skip. P:{loss_pred.item():.2f} F:{loss_fail.item():.2f} R:{loss_rca.item():.2f} Aux:{aux_loss_total_batch.item():.2f}")
                scheduler.step();
                continue
            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer);
            scaler.update();
            scheduler.step()

            total_loss_train += combined_loss.item()
            total_pred_loss_train += loss_pred.item();
            total_fail_loss_train += loss_fail.item();
            total_rca_loss_train += loss_rca.item()
            total_aux_loss_train += aux_loss_total_batch.item()

            if batch_idx > 0 and batch_idx % (len(train_loader) // 5 if len(train_loader) > 5 else 1) == 0:
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                print(
                    f"E{epoch + 1} B{batch_idx}/{len(train_loader)} LR:{optimizer.param_groups[0]['lr']:.1e} GNs:{grad_norm:.1f} L:{combined_loss.item():.2f}(P:{loss_pred.item():.2f} F:{loss_fail.item():.2f} R:{loss_rca.item():.2f} Aux:{aux_loss_total_batch.item():.2f})")

        num_b = max(1, len(train_loader))
        print(
            f"E{epoch + 1} AvgTrL:{total_loss_train / num_b:.2f} (P:{total_pred_loss_train / num_b:.2f} F:{total_fail_loss_train / num_b:.2f} R:{total_rca_loss_train / num_b:.2f} Aux:{total_aux_loss_train / num_b:.2f})")

        if valid_loader and len(valid_loader) > 0:
            model.eval()
            total_loss_val, total_pred_loss_val, total_fail_loss_val, total_rca_loss_val, total_aux_loss_val = 0, 0, 0, 0, 0
            with torch.no_grad():
                for batch_val in valid_loader:
                    input_features_val, sensor_mask_val = batch_val["input_features"].to(DEVICE), batch_val[
                        "sensor_mask"].to(DEVICE)
                    last_known_values_val, pred_delta_targets_val = batch_val["last_known_values"].to(DEVICE), \
                    batch_val["pred_delta_targets"].to(DEVICE)
                    fail_targets_val, rca_targets_val = batch_val["fail_targets"].to(DEVICE), batch_val[
                        "rca_targets"].to(DEVICE)
                    actual_pred_abs_targets_val = last_known_values_val.unsqueeze(-1) + pred_delta_targets_val
                    with autocast(enabled=AMP_ENABLED):
                        pred_abs_denorm_val, fail_logits_val, rca_logits_val, aux_val = model(input_features_val,
                                                                                              sensor_mask_val)
                        active_pred_elements_val = sensor_mask_val.sum().clamp(min=1) * len(PRED_HORIZONS)
                        loss_p_val = (huber_loss_fn(pred_abs_denorm_val,
                                                    actual_pred_abs_targets_val) * sensor_mask_val.unsqueeze(
                            -1)).sum() / active_pred_elements_val.clamp(min=1e-9)
                        loss_f_val = focal_loss_mean(fail_logits_val, fail_targets_val)
                        loss_r_val = (focal_loss_elementwise(rca_logits_val,
                                                             rca_targets_val) * sensor_mask_val).sum() / sensor_mask_val.sum().clamp(
                            min=1e-9)
                        aux_l_val_b = sum(
                            p[0].size(0) * torch.sum(p[0] * p[1]) for p in aux_val.values()) * AUX_LOSS_COEFF
                        comb_l_val = w_pred * loss_p_val + w_fail * loss_f_val + w_rca * loss_r_val + aux_l_val_b
                    if not (torch.isnan(comb_l_val) or torch.isinf(comb_l_val)):
                        total_loss_val += comb_l_val.item();
                        total_pred_loss_val += loss_p_val.item();
                        total_fail_loss_val += loss_f_val.item();
                        total_rca_loss_val += loss_r_val.item();
                        total_aux_loss_val += aux_l_val_b.item()
            num_b_v = max(1, len(valid_loader))
            print(
                f"E{epoch + 1} AvgValL:{total_loss_val / num_b_v:.2f} (P:{total_pred_loss_val / num_b_v:.2f} F:{total_fail_loss_val / num_b_v:.2f} R:{total_rca_loss_val / num_b_v:.2f} Aux:{total_aux_loss_val / num_b_v:.2f})")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Preprocessor config saved to {PREPROCESSOR_SAVE_PATH}")


if __name__ == '__main__':
    train_model()