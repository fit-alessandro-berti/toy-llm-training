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

MOE_GLOBAL_INPUT_DIM = TRANSFORMER_D_MODEL
MOE_HIDDEN_DIM_EXPERT = 128
NUM_EXPERTS = 4
MOE_OUTPUT_DIM = 64
MOE_TOP_K = 2

# Training Params
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == 'cuda'

MODEL_SAVE_PATH = "foundation_timeseries_model_v4.pth"  # New version
PREPROCESSOR_SAVE_PATH = "preprocessor_config_v4.npz"  # New version

HUBER_DELTA = 1.0
FOCAL_ALPHA_PARAM = 0.25  # Alpha for Focal Loss (weight of positive class)
FOCAL_GAMMA = 2.0

# Augmentation Params
JITTER_STRENGTH_RATIO = 0.03  # Std of jitter noise as a ratio of batch data std
MAG_WARP_STRENGTH_RATIO = 0.05  # Std of warping curve knots relative to 1
MAG_WARP_KNOTS = 4
MIXUP_PROB = 0.5  # Probability of applying mixup
MIXUP_ALPHA = 0.4  # Alpha for Beta distribution in Mixup


# --- RevIN Layer ---
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        num_features: Represents the number of time series to normalize independently (e.g., MaxSensors)
        Assumes input x: [Batch, NumSensors(num_features), SeqLen]
        """
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
                # This might happen if a batch during eval had all-zero inputs after masking,
                # leading to NaN stats that were perhaps reset.
                # Or if called on uninitialized module. For safety, just return input.
                # A more robust solution might involve checking for stored stats more carefully.
                print("Warning: RevIN denorm called without valid stats, returning input as is.")
                return x

            x_denorm = x
            if self.affine:
                # Ensure affine_weight is not too close to zero before division
                safe_affine_weight = self.affine_weight + self.eps
                x_denorm = (x_denorm - self.affine_bias) / safe_affine_weight
            x_denorm = x_denorm * self.stdev + self.mean
            return x_denorm
        else:
            raise NotImplementedError(f"RevIN mode '{mode}' not implemented.")

    def _get_statistics(self, x):
        # x: [Batch, NumSensors, SeqLen]
        # Calculate mean and stdev per sensor series (over SeqLen)
        self.mean = torch.mean(x, dim=-1, keepdim=True)  # [B, NumSensors, 1]
        self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps)  # [B, NumSensors, 1]


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

    def forward(self, x):  # x: [Batch, SeqLen, Dim]
        return x + self.pe[:, :x.size(1), :]


# --- Helper: CausalConv1D Block ---
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


# --- Helper: Focal Loss (updated for soft targets from Mixup) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha_param = alpha  # Weight for the positive class contribution
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):  # targets can be soft (e.g. from mixup)
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)  # This is p_t in the paper (probability of ground truth class)

        # Calculate alpha_t for soft targets: alpha if target is 1, (1-alpha) if target is 0
        # For soft targets t_soft in [0,1], alpha_t = t_soft * alpha + (1-t_soft) * (1-alpha)
        # This means we weight the "positive" contribution by alpha, and "negative" by (1-alpha)
        alpha_t = targets * self.alpha_param + (1.0 - targets) * (1.0 - self.alpha_param)

        focal_loss_unweighted = (1 - pt) ** self.gamma * bce_loss
        loss = alpha_t * focal_loss_unweighted

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
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

            # NOTE: Per-file scaling is kept here. RevIN will do instance normalization on top of this.
            # Alternatively, remove this scaling if RevIN is expected to handle raw-like data.
            # For now, keeping it as it might help stabilize inputs to RevIN.
            scalers = [StandardScaler() for _ in range(num_actual_sensors)]
            for i in range(num_actual_sensors):
                if features.shape[0] > 0:  # Check if features is not empty
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
            "input_features": torch.from_numpy(padded_input_seq_features),  # Original scale
            "sensor_mask": torch.from_numpy(sensor_mask),
            "last_known_values": torch.from_numpy(last_known_values),  # Original scale
            "pred_delta_targets": torch.from_numpy(pred_delta_targets),  # Original scale deltas
            "fail_targets": torch.from_numpy(fail_targets),
            "rca_targets": torch.from_numpy(rca_targets)
        }


# --- Model Architecture ---
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

    def forward(self,
                x):  # x: [Batch*MaxSensors, SeqLen, InputDim (after RevIN, this is effectively 1 if RevIN outputs 1 feature per sensor time series)]
        # Or, if RevIN is on [B, MS, SL] and then this gets [B*MS, SL, SENSOR_INPUT_DIM=1 (orig scale)]
        # Let's assume this encoder gets the SENSOR_INPUT_DIM (e.g. 1) as input feature dim for each sensor series.
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        x = x.permute(0, 2, 1)
        for layer in self.manual_cnn_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)
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

    def forward(self, x): return self.fc(x)


class FoundationalTimeSeriesModelV4(nn.Module):  # Renamed to V4 for clarity
    def __init__(self, max_sensors, seq_len,
                 sensor_input_dim, sensor_cnn_proj_dim, sensor_cnn_out_dim, cnn_layers, cnn_kernel_size,
                 cnn_dilation_base,
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 moe_global_input_dim, moe_hidden_dim_expert, num_experts, moe_output_dim, moe_top_k,
                 pred_horizons_len, fail_horizons_len, revin_affine=True):
        super().__init__()
        self.max_sensors = max_sensors
        self.seq_len = seq_len
        self.moe_top_k = moe_top_k
        self.moe_output_dim = moe_output_dim

        self.revin_layer = RevIN(num_features=max_sensors, affine=revin_affine)  # num_features is MaxSensors

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

    def forward(self, x_features_orig_scale, sensor_mask, last_known_values_orig_scale_for_pred_target=None):
        # x_features_orig_scale: [B, SeqLen, MaxSensors] (original or dataset-scaled)
        # sensor_mask: [B, MaxSensors]
        batch_size, seq_len, _ = x_features_orig_scale.shape

        # 1. RevIN Normalization
        # Input to RevIN: [B, MaxSensors, SeqLen]
        x_revin_input = x_features_orig_scale.permute(0, 2, 1) * sensor_mask.unsqueeze(-1)  # Mask before RevIN stats
        x_norm_revin = self.revin_layer(x_revin_input, mode='norm')  # Output: [B, MaxSensors, SeqLen]

        # Extract last normalized value for delta prediction reconstruction
        # This is the last value of the *normalized input sequence*
        last_val_norm_for_delta_recon = x_norm_revin[:, :, -1]  # [B, MaxSensors]

        # Prepare for PerSensorEncoder: [B*MaxSensors, SeqLen, SENSOR_INPUT_DIM=1]
        # x_norm_revin needs to be [B, SeqLen, MaxSensors] then reshaped
        x_norm_for_encoder_input = x_norm_revin.permute(0, 2, 1).unsqueeze(-1)  # [B, SeqLen, MaxSensors, 1]
        x_reshaped_for_encoder = x_norm_for_encoder_input.reshape(batch_size * self.max_sensors, seq_len,
                                                                  SENSOR_INPUT_DIM)

        # 2. Per-Sensor Encoding on normalized data
        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, self.max_sensors, seq_len,
                                                                         SENSOR_CNN_OUT_DIM)
        sensor_temporal_features = sensor_temporal_features * sensor_mask.view(batch_size, self.max_sensors, 1, 1)

        # ... (rest of the forward pass is same as V2/V3, operating on these normalized features) ...
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
        pred_delta_output_norm = self.pred_head(pred_head_input)  # These are deltas in normalized space

        # Denormalize prediction output using RevIN
        # pred_delta_output_norm: [B, MaxSensors, NumPredHorizons]
        # last_val_norm_for_delta_recon: [B, MaxSensors]
        pred_absolute_norm = last_val_norm_for_delta_recon.unsqueeze(-1) + pred_delta_output_norm

        # For denorm, RevIN expects [B, MaxSensors, SomeLengthDim]
        # pred_absolute_norm is [B, MaxSensors, NumPredHorizons], which fits.
        pred_absolute_final_denorm = self.revin_layer(pred_absolute_norm, mode='denorm')
        pred_absolute_final_denorm = pred_absolute_final_denorm * sensor_mask.unsqueeze(
            -1)  # Mask out padded sensors' predictions

        fail_output_logits = self.fail_head(moe_fail_output)

        moe_rca_expanded = moe_rca_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        rca_head_input = torch.cat([final_features_last_step, moe_rca_expanded], dim=-1)
        rca_output_logits = self.rca_head(rca_head_input).squeeze(-1)

        return pred_absolute_final_denorm, fail_output_logits, rca_output_logits


# --- Augmentation Functions ---
def augment_jitter(x_batch, sensor_mask, strength_ratio=0.03):
    # x_batch: [B, SeqLen, MaxSensors]
    if strength_ratio == 0: return x_batch
    x_aug = x_batch.clone()
    B, SL, MS = x_batch.shape

    # Calculate std per active sensor series for scaling noise
    # This ensures noise is proportional to sensor's own variance
    # Active sensor data: x_batch[b, :, s_idx] where sensor_mask[b, s_idx] == 1
    # For simplicity, compute std across whole batch for active parts, or use a fixed small noise if this is complex

    # Simplified Jitter: Add Gaussian noise scaled by a fixed ratio of overall data magnitude or just a small absolute value
    # To prevent adding noise to padded sensors, ensure multiplication by mask at the end
    noise = torch.normal(mean=0., std=1.0, size=x_batch.shape, device=x_batch.device)

    # Calculate std for each sensor series in the batch
    # We need to do this carefully due to padding.
    # Only consider active parts for std calculation.
    masked_x = x_batch * sensor_mask.unsqueeze(1)  # Zero out padded sensors before std
    # Calculate std over SeqLen for each sensor for each batch item
    # Keepdims to maintain B, 1, MS for broadcasting
    stds = torch.std(masked_x, dim=1, keepdim=True, unbiased=False)  # [B, 1, MS]
    # For sensors that are entirely padding (std=0), or constant, avoid NaN/Inf by adding eps or replacing 0 std
    stds = stds + 1e-6

    scaled_noise = noise * (strength_ratio * stds)
    x_aug = x_aug + scaled_noise
    return x_aug * sensor_mask.unsqueeze(1)  # Ensure only active sensors are affected


def augment_magnitude_warp(x_batch, sensor_mask, strength_ratio=0.05, num_knots=4):
    # x_batch: [B, SeqLen, MaxSensors]
    if strength_ratio == 0 or num_knots == 0: return x_batch
    x_aug = x_batch.clone()
    B, SL, MS = x_batch.shape

    for i in range(B):
        if sensor_mask[i].sum() == 0: continue  # Skip if no active sensors for this sample

        t_knots = torch.linspace(0, SL - 1, num_knots, device=x_batch.device)
        # Knots centered around 1.0
        y_knots = torch.normal(mean=1.0, std=strength_ratio, size=(num_knots,), device=x_batch.device)

        x_coords = torch.arange(SL, device=x_batch.device).float()

        # Linear interpolation of knots to create a warp curve of length SL
        # This part is a bit tricky to do efficiently in pure PyTorch for multiple batch items
        # without explicit loops or more advanced tensor ops if num_knots is small.
        # For simplicity, using a PyTorch-based linear interpolation if possible, or np.interp per sample.
        # This implementation will do it per sample (can be slow if B is large)
        warp_curve_np = np.interp(x_coords.cpu().numpy(), t_knots.cpu().numpy(), y_knots.cpu().numpy())
        warp_curve = torch.from_numpy(warp_curve_np).float().to(x_batch.device).unsqueeze(-1)  # [SL, 1]

        # Apply to active sensors of sample i
        active_sensor_indices_sample_i = sensor_mask[i] == 1.0  # [MS]
        x_aug[i, :, active_sensor_indices_sample_i] *= warp_curve  # Broadcast SL along active sensors

    return x_aug


# --- Training Loop ---
def train_model():
    print(f"Using device: {DEVICE}")
    print(f"Automatic Mixed Precision (AMP) enabled: {AMP_ENABLED}")

    all_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv")) + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_files: print("No CSV files found. Exiting."); return

    max_sensors_overall = get_max_sensors_from_files(all_files)
    if max_sensors_overall == 0: print("Could not determine max_sensors_overall. Exiting."); return
    print(f"Determined max_sensors_overall (capped at {MAX_SENSORS_CAP}): {max_sensors_overall}")
    np.savez(PREPROCESSOR_SAVE_PATH, max_sensors_overall=max_sensors_overall)

    train_dataset = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, max_sensors_overall)
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, max_sensors_overall)
    if len(train_dataset) == 0: print("Training dataset empty. Exiting."); return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              pin_memory=True if DEVICE.type == 'cuda' else False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=True if DEVICE.type == 'cuda' else False) if len(valid_dataset) > 0 else None

    model = FoundationalTimeSeriesModelV4(  # Using new model name
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

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_steps)

    def lr_lambda_combined(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_combined)
    scaler = GradScaler(enabled=AMP_ENABLED)

    huber_loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')
    focal_loss_fn_elementwise = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA, reduction='none')
    focal_loss_fn_mean = FocalLoss(alpha=FOCAL_ALPHA_PARAM, gamma=FOCAL_GAMMA, reduction='mean')

    w_pred, w_fail, w_rca = 1.0, 1.0, 0.5

    for epoch in range(EPOCHS):
        model.train()
        total_loss_train, total_pred_loss_train, total_fail_loss_train, total_rca_loss_train = 0, 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            input_features_orig = batch["input_features"].to(DEVICE)  # [B, SL, MS]
            sensor_mask = batch["sensor_mask"].to(DEVICE)  # [B, MS]
            last_known_values_orig = batch["last_known_values"].to(DEVICE)  # [B, MS] (orig scale)
            pred_delta_targets_orig = batch["pred_delta_targets"].to(
                DEVICE)  # [B, MS, NumPredHorizons] (orig scale deltas)
            fail_targets_orig = batch["fail_targets"].to(DEVICE)
            rca_targets_orig = batch["rca_targets"].to(DEVICE)

            # --- Augmentations (only in training) ---
            current_input_features = input_features_orig
            if JITTER_STRENGTH_RATIO > 0:
                current_input_features = augment_jitter(current_input_features, sensor_mask, JITTER_STRENGTH_RATIO)
            if MAG_WARP_STRENGTH_RATIO > 0:
                current_input_features = augment_magnitude_warp(current_input_features, sensor_mask,
                                                                MAG_WARP_STRENGTH_RATIO, MAG_WARP_KNOTS)

            # Prepare targets for loss calculation
            # Prediction target is absolute values in original scale
            actual_pred_abs_targets = last_known_values_orig.unsqueeze(-1) + pred_delta_targets_orig

            # --- Mixup (only in training) ---
            if random.random() < MIXUP_PROB:
                lambda_mix = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)

                # Shuffle batch to get pairs for mixup
                perm_indices = torch.randperm(current_input_features.size(0), device=DEVICE)

                input_features_shuffled = current_input_features[perm_indices]
                actual_pred_abs_targets_shuffled = actual_pred_abs_targets[perm_indices]
                fail_targets_shuffled = fail_targets_orig[perm_indices]
                rca_targets_shuffled = rca_targets_orig[perm_indices]
                # sensor_mask_shuffled = sensor_mask[perm_indices] # if needed for mixed mask

                mixed_input_features = lambda_mix * current_input_features + (1 - lambda_mix) * input_features_shuffled

                # Mix targets
                mixed_pred_abs_targets = lambda_mix * actual_pred_abs_targets + (
                            1 - lambda_mix) * actual_pred_abs_targets_shuffled
                mixed_fail_targets = lambda_mix * fail_targets_orig + (1 - lambda_mix) * fail_targets_shuffled
                mixed_rca_targets = lambda_mix * rca_targets_orig + (1 - lambda_mix) * rca_targets_shuffled
                # Use the original sensor_mask for the mixed input, assuming structure remains.
                # Or, if masks can differ: mixed_sensor_mask = torch.max(sensor_mask, sensor_mask_shuffled)

                loss_input_features = mixed_input_features
                loss_pred_targets = mixed_pred_abs_targets
                loss_fail_targets = mixed_fail_targets
                loss_rca_targets = mixed_rca_targets
            else:
                loss_input_features = current_input_features
                loss_pred_targets = actual_pred_abs_targets
                loss_fail_targets = fail_targets_orig
                loss_rca_targets = rca_targets_orig
            # --- End Mixup ---

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=AMP_ENABLED):
                # Model's forward now takes original scale input, handles RevIN, and returns original scale abs predictions
                pred_abs_final_denorm, fail_output_logits, rca_output_logits = model(loss_input_features, sensor_mask)

                loss_pred_unmasked = huber_loss_fn(pred_abs_final_denorm, loss_pred_targets)
                active_pred_elements = sensor_mask.sum().clamp(min=1) * len(PRED_HORIZONS)
                loss_pred = (loss_pred_unmasked * sensor_mask.unsqueeze(-1)).sum() / active_pred_elements.clamp(
                    min=1e-9)

                loss_fail = focal_loss_fn_mean(fail_output_logits, loss_fail_targets)

                loss_rca_unmasked = focal_loss_fn_elementwise(rca_output_logits, loss_rca_targets)
                loss_rca = (loss_rca_unmasked * sensor_mask).sum() / sensor_mask.sum().clamp(min=1e-9)

                combined_loss = w_pred * loss_pred + w_fail * loss_fail + w_rca * loss_rca

            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(f"NaN/Inf loss: P:{loss_pred.item():.3f} F:{loss_fail.item():.3f} R:{loss_rca.item():.3f}. Skip.")
                scheduler.step()
                continue

            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss_train += combined_loss.item()
            total_pred_loss_train += loss_pred.item()
            total_fail_loss_train += loss_fail.item()
            total_rca_loss_train += loss_rca.item()

            if batch_idx > 0 and batch_idx % (len(train_loader) // 4 if len(train_loader) > 4 else 1) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                grad_norm_val = sum(
                    p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                print(
                    f"E{epoch + 1} B{batch_idx}/{len(train_loader)}, LR:{current_lr:.2e}, GradN:{grad_norm_val:.2f}, TrL:{combined_loss.item():.3f} (P:{loss_pred.item():.3f} F:{loss_fail.item():.3f} R:{loss_rca.item():.3f})")

        num_b_tr = max(1, len(train_loader))
        print(
            f"E{epoch + 1} AvgTrL: {total_loss_train / num_b_tr:.3f} (P:{total_pred_loss_train / num_b_tr:.3f} F:{total_fail_loss_train / num_b_tr:.3f} R:{total_rca_loss_train / num_b_tr:.3f})")

        if valid_loader and len(valid_loader) > 0:
            model.eval()
            total_loss_val, total_pred_loss_val, total_fail_loss_val, total_rca_loss_val = 0, 0, 0, 0
            with torch.no_grad():
                for batch_val in valid_loader:
                    input_features_val = batch_val["input_features"].to(DEVICE)
                    sensor_mask_val = batch_val["sensor_mask"].to(DEVICE)
                    last_known_values_val = batch_val["last_known_values"].to(DEVICE)
                    pred_delta_targets_val = batch_val["pred_delta_targets"].to(DEVICE)
                    fail_targets_val = batch_val["fail_targets"].to(DEVICE)
                    rca_targets_val = batch_val["rca_targets"].to(DEVICE)

                    actual_pred_abs_targets_val = last_known_values_val.unsqueeze(-1) + pred_delta_targets_val

                    with autocast(enabled=AMP_ENABLED):
                        pred_abs_final_denorm_val, fail_output_logits_val, rca_output_logits_val = model(
                            input_features_val, sensor_mask_val)

                        active_pred_elements_val = sensor_mask_val.sum().clamp(min=1) * len(PRED_HORIZONS)
                        loss_pred_val = (huber_loss_fn(pred_abs_final_denorm_val,
                                                       actual_pred_abs_targets_val) * sensor_mask_val.unsqueeze(
                            -1)).sum() / active_pred_elements_val.clamp(min=1e-9)
                        loss_fail_val = focal_loss_fn_mean(fail_output_logits_val, fail_targets_val)
                        loss_rca_val = (focal_loss_fn_elementwise(rca_output_logits_val,
                                                                  rca_targets_val) * sensor_mask_val).sum() / sensor_mask_val.sum().clamp(
                            min=1e-9)

                        combined_loss_val = w_pred * loss_pred_val + w_fail * loss_fail_val + w_rca * loss_rca_val

                    if not (torch.isnan(combined_loss_val) or torch.isinf(combined_loss_val)):
                        total_loss_val += combined_loss_val.item()
                        total_pred_loss_val += loss_pred_val.item()
                        total_fail_loss_val += loss_fail_val.item()
                        total_rca_loss_val += loss_rca_val.item()

            num_b_val = max(1, len(valid_loader))
            print(
                f"E{epoch + 1} AvgValL: {total_loss_val / num_b_val:.3f} (P:{total_pred_loss_val / num_b_val:.3f} F:{total_fail_loss_val / num_b_val:.3f} R:{total_rca_loss_val / num_b_val:.3f})")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Preprocessor config (max_sensors_overall) saved to {PREPROCESSOR_SAVE_PATH}")


if __name__ == '__main__':
    train_model()
