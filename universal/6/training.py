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

# For Weight Normalization, ensure PyTorch version supports parametrizations (1.9+)
# If older, from torch.nn.utils import weight_norm might be needed, but parametrizations is preferred.

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

# I. Per-sensor Temporal Encoder (TCN based)
SENSOR_TCN_PROJ_DIM = 32  # Dimension after initial projection, input to first TCN layer
SENSOR_TCN_OUT_DIM = 32  # Output channels of each TCN block and final TCN encoder output
TCN_LEVELS = 4  # Number of TCN residual blocks (e.g., dilations 1, 2, 4, 8)
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

# II. Inter-sensor Representation
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2

# III. Mixture-of-Experts (MoE)
MOE_GLOBAL_INPUT_DIM = TRANSFORMER_D_MODEL
NUM_EXPERTS_PER_TASK = 8
MOE_HIDDEN_DIM_EXPERT = 128
MOE_OUTPUT_DIM = 64
EXPERT_DROPOUT_RATE = 0.1
AUX_LOSS_COEFF = 0.01

# Training Params
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == 'cuda'

MODEL_SAVE_PATH = "foundation_timeseries_model_v6.pth"  # New version
PREPROCESSOR_SAVE_PATH = "preprocessor_config_v6.npz"  # New version

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
            if not hasattr(self, 'mean') or not hasattr(self, 'stdev'): return x
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
        pe[:, 0::2] = torch.sin(position * div_term);
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1), :]


# --- TCN Residual Block ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation  # For causal convolution by slicing output

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.conv1 = nn.utils.parametrizations.weight_norm(self.conv1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.padding,
                               dilation=dilation)
        self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()

    def forward(self, x):  # x: [Batch, Channels_in, SeqLen]
        # Layer 1
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)

        # Layer 2
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        # Ensure residual has same length as output after causal slicing
        if res.size(2) != out.size(
                2):  # Should only happen if input x was longer than conv output (not typical with this padding)
            # This case needs careful handling if stride > 1 or something unexpected.
            # For stride=1 and causal slice, res length should match out length if downsample doesn't change length.
            # If downsample is Conv1d(kernel_size=1), length is preserved.
            pass  # Assume lengths match with current setup.

        return self.relu_out(out + res)


# --- Per-Sensor TCN Encoder ---
class PerSensorEncoderTCN(nn.Module):
    def __init__(self, input_dim, proj_dim, tcn_out_dim, seq_len, num_levels, kernel_size, dropout):
        super(PerSensorEncoderTCN, self).__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=seq_len)

        tcn_blocks = []
        current_channels = proj_dim
        for i in range(num_levels):
            dilation_size = 2 ** i
            # All TCN blocks output tcn_out_dim
            tcn_blocks.append(TemporalBlock(current_channels, tcn_out_dim, kernel_size, stride=1,
                                            dilation=dilation_size, dropout=dropout))
            current_channels = tcn_out_dim  # Subsequent blocks take tcn_out_dim as input

        self.tcn_network = nn.Sequential(*tcn_blocks)
        self.final_norm = nn.LayerNorm(tcn_out_dim)

    def forward(self, x):  # x: [Batch*MaxSensors, SeqLen, InputDim]
        x = self.input_proj(x)  # -> [B*MS, SeqLen, ProjDim]
        x = self.pos_encoder(x)  # -> [B*MS, SeqLen, ProjDim]

        x = x.permute(0, 2, 1)  # -> [B*MS, ProjDim, SeqLen] for Conv1D
        x = self.tcn_network(x)  # -> [B*MS, TcnOutDim, SeqLen]
        x = x.permute(0, 2, 1)  # -> [B*MS, SeqLen, TcnOutDim]
        x = self.final_norm(x)
        return x


# --- Helper: Focal Loss ---
class FocalLoss(nn.Module):  # Same as V5
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha_param = alpha;
        self.gamma = gamma;
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets);
        pt = torch.exp(-bce_loss)
        alpha_t = targets * self.alpha_param + (1.0 - targets) * (1.0 - self.alpha_param)
        loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# --- Data Handling (same as V5) ---
def get_max_sensors_from_files(file_paths):
    max_s = 0
    for fp in file_paths:
        try:
            df_peek = pd.read_csv(fp, nrows=1); max_s = max(max_s, df_peek.shape[1] - 1)
        except Exception:
            continue
    return min(max_s, MAX_SENSORS_CAP)


class MultivariateTimeSeriesDataset(Dataset):  # Same as V5
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
            self.data_cache.append(
                {"features": features, "failure_flags": failure_flags, "num_actual_sensors": num_actual_sensors})
            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons), self.rca_failure_lookahead)
            for i in range(len(df) - self.seq_len - max_lookahead + 1): self.window_indices.append((file_idx, i))
        print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        item = self.data_cache[file_idx];
        features_full, flags_full, n_actual = item["features"], item["failure_flags"], item["num_actual_sensors"]
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
                "fail_targets": torch.from_numpy(fail_t), "rca_targets": torch.from_numpy(rca_t)}


# --- Model Architecture (V6 with TCN) ---
class InterSensorTransformer(nn.Module):  # Same as V5
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


class Expert(nn.Module):  # Same as V5
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__();
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x): return self.fc(x)


class GatingNetwork(nn.Module):  # Same as V5
    def __init__(self, input_dim, num_experts):
        super().__init__();
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x): return self.fc(x)


class FoundationalTimeSeriesModelV6(nn.Module):
    def __init__(self, max_sensors, seq_len,
                 sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim, tcn_levels, tcn_kernel_size, tcn_dropout,
                 # TCN params
                 transformer_d_model, transformer_nhead, transformer_nlayers,
                 moe_global_input_dim, num_experts_per_task, moe_hidden_dim_expert, moe_output_dim, expert_dropout_rate,
                 pred_horizons_len, fail_horizons_len, revin_affine=True):
        super().__init__()
        self.max_sensors = max_sensors;
        self.seq_len = seq_len;
        self.num_experts_per_task = num_experts_per_task
        self.moe_output_dim = moe_output_dim
        self.revin_layer = RevIN(num_features=max_sensors, affine=revin_affine)

        # Using PerSensorEncoderTCN
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
        self.expert_dropout = nn.Dropout(expert_dropout_rate)

        final_combined_feat_dim_last_step = sensor_tcn_out_dim + transformer_d_model  # Uses TCN output dim
        self.pred_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(final_combined_feat_dim_last_step + moe_output_dim, 1)

    def _apply_moe_switch(self, global_moe_input, gating_network, expert_pool):  # Same as V5
        gating_logits = gating_network(global_moe_input);
        router_probs = torch.softmax(gating_logits, dim=-1)
        chosen_expert_indices = torch.argmax(gating_logits, dim=-1)
        one_hot_selection = F.one_hot(chosen_expert_indices, num_classes=len(expert_pool)).float()
        all_expert_outputs = torch.stack([expert(global_moe_input) for expert in expert_pool], dim=1)
        moe_task_output = torch.sum(all_expert_outputs * one_hot_selection.unsqueeze(-1), dim=1)
        if self.training: moe_task_output = self.expert_dropout(moe_task_output)
        fi = one_hot_selection.mean(dim=0);
        Pi = router_probs.mean(dim=0)
        return moe_task_output, fi, Pi

    def forward(self, x_features_orig_scale, sensor_mask):
        batch_size, seq_len, _ = x_features_orig_scale.shape
        x_revin_input = x_features_orig_scale.permute(0, 2, 1) * sensor_mask.unsqueeze(-1);
        x_norm_revin = self.revin_layer(x_revin_input, mode='norm')
        last_val_norm_for_delta_recon = x_norm_revin[:, :, -1]
        x_norm_for_encoder_input = x_norm_revin.permute(0, 2, 1).unsqueeze(-1)
        x_reshaped_for_encoder = x_norm_for_encoder_input.reshape(batch_size * self.max_sensors, seq_len,
                                                                  SENSOR_INPUT_DIM)

        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)  # Uses TCN now
        sensor_temporal_features = sensor_temporal_features_flat.reshape(batch_size, self.max_sensors, seq_len,
                                                                         SENSOR_TCN_OUT_DIM)  # SENSOR_TCN_OUT_DIM
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

        expert_outputs_all_forecast = torch.stack([exp(global_moe_input) for exp in self.experts_forecast], dim=1)
        expert_outputs_all_fail = torch.stack([exp(global_moe_input) for exp in self.experts_fail], dim=1)
        expert_outputs_all_rca = torch.stack([exp(global_moe_input) for exp in self.experts_rca], dim=1)

        moe_forecast_output, fi_f, Pi_f = self._apply_moe_switch(global_moe_input, self.gating_forecast,
                                                                 expert_outputs_all_forecast)
        moe_fail_output, fi_fail, Pi_fail = self._apply_moe_switch(global_moe_input, self.gating_fail,
                                                                   expert_outputs_all_fail)
        moe_rca_output, fi_rca, Pi_rca = self._apply_moe_switch(global_moe_input, self.gating_rca,
                                                                expert_outputs_all_rca)
        aux_loss_terms = {"forecast": (fi_f, Pi_f), "fail": (fi_fail, Pi_fail), "rca": (fi_rca, Pi_rca)}

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
        return pred_abs_denorm, fail_logits, rca_logits, aux_loss_terms


# --- Augmentation Functions (same as V5) ---
def augment_jitter(x, mask, strength=0.03):
    if strength == 0: return x; noise = torch.normal(0., strength, x.shape, device=x.device)
    return (x + noise) * mask.unsqueeze(1)


def augment_magnitude_warp(x, mask, strength=0.05, n_knots=4):
    if strength == 0 or n_knots <= 1: return x; aug = x.clone(); B, SL, _ = x.shape
    for i in range(B):
        if mask[i].sum() == 0: continue
        t_k = torch.linspace(0, SL - 1, n_knots, device=x.device);
        y_k = torch.normal(1., strength, (n_knots,), device=x.device)
        x_c = torch.arange(SL, device=x.device).float()
        try:
            curve_np = np.interp(x_c.cpu().numpy(), t_k.cpu().numpy(), y_k.cpu().numpy())
            curve = torch.from_numpy(curve_np).float().to(x.device).unsqueeze(-1)
            aug[i, :, mask[i] == 1.0] *= curve
        except:
            pass
    return aug


# --- Training Loop (mostly same as V5, adjusted for V6 model) ---
def train_model():
    print(f"Device: {DEVICE}, AMP: {AMP_ENABLED}");
    all_f = glob.glob(os.path.join(TRAIN_DIR, "*.csv")) + glob.glob(os.path.join(VALID_DIR, "*.csv"))
    if not all_f: print("No CSVs. Exit."); return
    max_s_overall = get_max_sensors_from_files(all_f)
    if max_s_overall == 0: print("Max sensors 0. Exit."); return
    print(f"Max sensors (cap {MAX_SENSORS_CAP}): {max_s_overall}");
    np.savez(PREPROCESSOR_SAVE_PATH, max_s_overall=max_s_overall)
    train_ds = MultivariateTimeSeriesDataset(TRAIN_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS, RCA_FAILURE_LOOKAHEAD,
                                             max_s_overall)
    valid_ds = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS, RCA_FAILURE_LOOKAHEAD,
                                             max_s_overall)
    if len(train_ds) == 0: print("Train dataset empty. Exit."); return
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=AMP_ENABLED)
    valid_dl = DataLoader(valid_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=AMP_ENABLED) if len(
        valid_ds) > 0 else None

    model = FoundationalTimeSeriesModelV6(  # Using V6
        max_sensors=max_s_overall, seq_len=SEQ_LEN,
        sensor_input_dim=SENSOR_INPUT_DIM, sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        tcn_levels=TCN_LEVELS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT,
        transformer_d_model=TRANSFORMER_D_MODEL, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        moe_global_input_dim=MOE_GLOBAL_INPUT_DIM, num_experts_per_task=NUM_EXPERTS_PER_TASK,
        moe_hidden_dim_expert=MOE_HIDDEN_DIM_EXPERT, moe_output_dim=MOE_OUTPUT_DIM,
        expert_dropout_rate=EXPERT_DROPOUT_RATE,
        pred_horizons_len=len(PRED_HORIZONS), fail_horizons_len=len(FAIL_HORIZONS)
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)
    steps_epoch = len(train_dl);
    total_s = EPOCHS * steps_epoch;
    warmup_s = int(WARMUP_RATIO * total_s)

    def lr_lambda(curr_s):
        if warmup_s > 0 and curr_s < warmup_s: return float(curr_s) / float(max(1, warmup_s))
        prog = float(curr_s - warmup_s) / float(max(1, total_s - warmup_s));
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda);
    scaler = GradScaler(enabled=AMP_ENABLED)
    huber_fn = nn.HuberLoss(HUBER_DELTA, reduction='none');
    focal_elem = FocalLoss(FOCAL_ALPHA_PARAM, FOCAL_GAMMA, reduction='none');
    focal_mean = FocalLoss(FOCAL_ALPHA_PARAM, FOCAL_GAMMA, reduction='mean')
    wp, wf, wr = 1.0, 1.0, 0.5

    for ep in range(EPOCHS):
        model.train();
        tot_L, tot_pL, tot_fL, tot_rL, tot_auxL = 0, 0, 0, 0, 0
        for b_idx, b in enumerate(train_dl):
            in_f, mask, last_k, p_del_t, f_t, r_t = b["input_features"].to(DEVICE), b["sensor_mask"].to(DEVICE), b[
                "last_known_values"].to(DEVICE), b["pred_delta_targets"].to(DEVICE), b["fail_targets"].to(DEVICE), b[
                "rca_targets"].to(DEVICE)
            curr_in_f = in_f
            if JITTER_STRENGTH_RATIO > 0: curr_in_f = augment_jitter(curr_in_f, mask, JITTER_STRENGTH_RATIO)
            if MAG_WARP_STRENGTH_RATIO > 0: curr_in_f = augment_magnitude_warp(curr_in_f, mask, MAG_WARP_STRENGTH_RATIO,
                                                                               MAG_WARP_KNOTS)
            p_abs_t = last_k.unsqueeze(-1) + p_del_t
            loss_in, loss_pt, loss_ft, loss_rt = curr_in_f, p_abs_t, f_t, r_t
            if random.random() < MIXUP_PROB:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA);
                p_idx = torch.randperm(curr_in_f.size(0), device=DEVICE)
                loss_in = lam * curr_in_f + (1 - lam) * curr_in_f[p_idx]
                loss_pt = lam * p_abs_t + (1 - lam) * p_abs_t[p_idx]
                loss_ft = lam * f_t + (1 - lam) * f_t[p_idx]
                loss_rt = lam * r_t + (1 - lam) * r_t[p_idx]
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=AMP_ENABLED):
                p_abs_denorm, f_logits, r_logits, aux_terms = model(loss_in, mask)
                Lp_un = huber_fn(p_abs_denorm, loss_pt);
                act_p_elem = mask.sum().clamp(min=1) * len(PRED_HORIZONS)
                Lp = (Lp_un * mask.unsqueeze(-1)).sum() / act_p_elem.clamp(min=1e-9)
                Lf = focal_mean(f_logits, loss_ft)
                Lr_un = focal_elem(r_logits, loss_rt)
                Lr = (Lr_un * mask).sum() / mask.sum().clamp(min=1e-9)
                Laux_b = sum(p[0].size(0) * torch.sum(p[0] * p[1]) for p in aux_terms.values()) * AUX_LOSS_COEFF
                Lcomb = wp * Lp + wf * Lf + wr * Lr + Laux_b
            if torch.isnan(Lcomb) or torch.isinf(Lcomb):
                print(f"NaN/Inf L. Skip. P:{Lp.item():.2f} F:{Lf.item():.2f} R:{Lr.item():.2f} Aux:{Laux_b.item():.2f}")
                sched.step();
                continue
            scaler.scale(Lcomb).backward();
            scaler.unscale_(opt);
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(opt);
            scaler.update();
            sched.step()
            tot_L += Lcomb.item();
            tot_pL += Lp.item();
            tot_fL += Lf.item();
            tot_rL += Lr.item();
            tot_auxL += Laux_b.item()
            if b_idx > 0 and b_idx % (len(train_dl) // 5 if len(train_dl) > 5 else 1) == 0:
                gn = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                print(
                    f"E{ep + 1} B{b_idx}/{len(train_dl)} LR:{opt.param_groups[0]['lr']:.1e} GNs:{gn:.1f} L:{Lcomb.item():.2f}(P:{Lp.item():.2f} F:{Lf.item():.2f} R:{Lr.item():.2f} Aux:{Laux_b.item():.2f})")
        n_b = max(1, len(train_dl))
        print(
            f"E{ep + 1} AvgTrL:{tot_L / n_b:.2f} (P:{tot_pL / n_b:.2f} F:{tot_fL / n_b:.2f} R:{tot_rL / n_b:.2f} Aux:{tot_auxL / n_b:.2f})")
        if valid_dl and len(valid_dl) > 0:
            model.eval();
            tot_Lv, tot_pLv, tot_fLv, tot_rLv, tot_auxLv = 0, 0, 0, 0, 0
            with torch.no_grad():
                for b_v in valid_dl:
                    in_fv, mask_v, last_kv, p_del_tv, f_tv, r_tv = b_v["input_features"].to(DEVICE), b_v[
                        "sensor_mask"].to(DEVICE), b_v["last_known_values"].to(DEVICE), b_v["pred_delta_targets"].to(
                        DEVICE), b_v["fail_targets"].to(DEVICE), b_v["rca_targets"].to(DEVICE)
                    p_abs_tv = last_kv.unsqueeze(-1) + p_del_tv
                    with autocast(enabled=AMP_ENABLED):
                        p_abs_den_v, f_log_v, r_log_v, aux_v = model(in_fv, mask_v)
                        act_p_elem_v = mask_v.sum().clamp(min=1) * len(PRED_HORIZONS)
                        Lpv = (huber_fn(p_abs_den_v, p_abs_tv) * mask_v.unsqueeze(-1)).sum() / act_p_elem_v.clamp(
                            min=1e-9)
                        Lfv = focal_mean(f_log_v, f_tv)
                        Lrv = (focal_elem(r_log_v, r_tv) * mask_v).sum() / mask_v.sum().clamp(min=1e-9)
                        Laux_v_b = sum(p[0].size(0) * torch.sum(p[0] * p[1]) for p in aux_v.values()) * AUX_LOSS_COEFF
                        Lcomb_v = wp * Lpv + wf * Lfv + wr * Lrv + Laux_v_b
                    if not (torch.isnan(Lcomb_v) or torch.isinf(Lcomb_v)):
                        tot_Lv += Lcomb_v.item();
                        tot_pLv += Lpv.item();
                        tot_fLv += Lfv.item();
                        tot_rLv += Lrv.item();
                        tot_auxLv += Laux_v_b.item()
            n_bv = max(1, len(valid_dl))
            print(
                f"E{ep + 1} AvgValL:{tot_Lv / n_bv:.2f} (P:{tot_pLv / n_bv:.2f} F:{tot_fLv / n_bv:.2f} R:{tot_rLv / n_bv:.2f} Aux:{tot_auxLv / n_bv:.2f})")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved: {MODEL_SAVE_PATH}, Preprocessor: {PREPROCESSOR_SAVE_PATH}")


if __name__ == '__main__':
    train_model()
