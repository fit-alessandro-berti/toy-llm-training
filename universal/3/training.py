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
EPOCHS = 15  # Or adjust based on new LR's convergence speed
LEARNING_RATE = 1e-3  # Increased Learning Rate
ADAM_BETAS = (0.9, 0.98)
ADAM_WEIGHT_DECAY = 1e-2
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_RATIO = 0.05  # 5% of total steps for warmup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == 'cuda'  # Enable AMP only if CUDA is available

MODEL_SAVE_PATH = "foundation_timeseries_model_v3.pth"  # New version for optimized model
PREPROCESSOR_SAVE_PATH = "preprocessor_config_v3.npz"

HUBER_DELTA = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


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
        # LayerNorm will be added after this block in PerSensorEncoderCNN if needed

    def forward(self, x):  # x: [Batch, Channels, SeqLen]
        x = self.conv1(x)
        x = x[:, :, :-self.padding] if self.padding > 0 else x
        x = self.relu(x)
        return x


# --- Helper: Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha).to(inputs.device)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# --- Data Handling (same as V2) ---
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
            "rca_targets": torch.from_numpy(rca_targets)
        }


# --- Model Architecture (V2 with LayerNorm additions) ---
class PerSensorEncoderCNN(nn.Module):
    def __init__(self, input_dim, proj_dim, cnn_out_dim, seq_len, num_layers, kernel_size, dilation_base):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=seq_len)

        self.manual_cnn_layers = nn.ModuleList()
        current_dim_manual = proj_dim
        for i in range(num_layers):
            out_d = cnn_out_dim if i == num_layers - 1 else proj_dim
            # Each block is CausalConv1DBlock + LayerNorm
            block_layers = [CausalConv1DBlock(current_dim_manual, out_d, kernel_size, dilation_base ** i)]
            # LayerNorm expects [Batch, SeqLen, Features] or [Batch, Features]
            # Output of CausalConv1DBlock (after permute) is [Batch, SeqLen, Features=out_d]
            # So LayerNorm(out_d) is appropriate.
            # Adding LayerNorm *after* each block's activation (inside forward loop is cleaner)
            self.manual_cnn_layers.append(CausalConv1DBlock(current_dim_manual, out_d, kernel_size, dilation_base ** i))
            current_dim_manual = out_d
        # Final LayerNorm after all CNN blocks, applied on cnn_out_dim
        self.final_norm = nn.LayerNorm(cnn_out_dim)

    def forward(self, x):  # x: [Batch*MaxSensors, SeqLen, InputDim=1]
        x = self.input_proj(x)
        x = self.pos_encoder(x)  # x: [B*MS, SeqLen, ProjDim]

        # Permute for Conv1D: [B*MS, ProjDim, SeqLen]
        x = x.permute(0, 2, 1)
        for layer in self.manual_cnn_layers:
            x = layer(x)  # Output of last layer will be [B*MS, CnnOutDim, SeqLen]

        # Permute back: [B*MS, SeqLen, CnnOutDim]
        x = x.permute(0, 2, 1)
        x = self.final_norm(x)  # Apply LayerNorm on the feature dimension
        return x


class InterSensorTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, max_sensors):  # embed_dim is d_model
        super().__init__()
        self.pos_encoder_inter_sensor = nn.Parameter(torch.zeros(1, max_sensors, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True,
                                                   dim_feedforward=embed_dim * 2,
                                                   norm_first=True)  # norm_first can help
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)  # LayerNorm after transformer encoder

    def forward(self, x, src_key_padding_mask):
        x = x + self.pos_encoder_inter_sensor[:, :x.size(1), :]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.output_norm(x)  # Apply LayerNorm
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


class FoundationalTimeSeriesModelV2(nn.Module):
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

    model = FoundationalTimeSeriesModelV2(
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

    # Scheduler with warmup and cosine decay
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_steps)

    def lr_lambda_combined(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing phase (factor from 1 down to 0)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_combined)

    # Mixed Precision Scaler
    scaler = GradScaler(enabled=AMP_ENABLED)

    huber_loss_fn = nn.HuberLoss(delta=HUBER_DELTA, reduction='none')
    focal_loss_fn_elementwise = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='none')
    focal_loss_fn_mean = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean')

    w_pred, w_fail, w_rca = 1.0, 1.0, 0.5

    for epoch in range(EPOCHS):
        model.train()
        total_loss_train, total_pred_loss_train, total_fail_loss_train, total_rca_loss_train = 0, 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(DEVICE)
            sensor_mask = batch["sensor_mask"].to(DEVICE)
            pred_delta_targets = batch["pred_delta_targets"].to(DEVICE)
            fail_targets = batch["fail_targets"].to(DEVICE)
            rca_targets = batch["rca_targets"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)  # set_to_none for memory efficiency

            with autocast(enabled=AMP_ENABLED):
                pred_delta_output, fail_output_logits, rca_output_logits = model(input_features, sensor_mask)

                loss_pred_unmasked = huber_loss_fn(pred_delta_output, pred_delta_targets)
                active_pred_elements = sensor_mask.sum().clamp(min=1) * len(PRED_HORIZONS)
                loss_pred = (loss_pred_unmasked * sensor_mask.unsqueeze(-1)).sum() / active_pred_elements.clamp(
                    min=1e-9)

                loss_fail = focal_loss_fn_mean(fail_output_logits, fail_targets)

                loss_rca_unmasked = focal_loss_fn_elementwise(rca_output_logits, rca_targets)
                loss_rca = (loss_rca_unmasked * sensor_mask).sum() / sensor_mask.sum().clamp(min=1e-9)

                combined_loss = w_pred * loss_pred + w_fail * loss_fail + w_rca * loss_rca

            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print(f"NaN/Inf loss detected at Epoch {epoch + 1}, Batch {batch_idx + 1}. Skipping batch.")
                # Skipping optimizer step for this batch if loss is invalid
                scheduler.step()  # Still step scheduler to keep it in sync with global step count
                continue

            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss_train += combined_loss.item()
            total_pred_loss_train += loss_pred.item() if not (torch.isnan(loss_pred) or torch.isinf(loss_pred)) else 0
            total_fail_loss_train += loss_fail.item() if not (torch.isnan(loss_fail) or torch.isinf(loss_fail)) else 0
            total_rca_loss_train += loss_rca.item() if not (torch.isnan(loss_rca) or torch.isinf(loss_rca)) else 0

            if batch_idx > 0 and batch_idx % (len(train_loader) // 4 if len(train_loader) > 4 else 1) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                grad_norm_val = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm_val += p.grad.data.norm(2).item() ** 2
                grad_norm_val = grad_norm_val ** 0.5

                print(
                    f"E{epoch + 1} B{batch_idx}/{len(train_loader)}, LR:{current_lr:.2e}, GradN:{grad_norm_val:.2f}, TrL:{combined_loss.item():.3f} (P:{loss_pred.item():.3f} F:{loss_fail.item():.3f} R:{loss_rca.item():.3f})")

        num_b_tr = max(1, len(train_loader))
        print(
            f"E{epoch + 1} AvgTrL: {total_loss_train / num_b_tr:.3f} (P:{total_pred_loss_train / num_b_tr:.3f} F:{total_fail_loss_train / num_b_tr:.3f} R:{total_rca_loss_train / num_b_tr:.3f})")

        if valid_loader and len(valid_loader) > 0:
            model.eval()
            total_loss_val, total_pred_loss_val, total_fail_loss_val, total_rca_loss_val = 0, 0, 0, 0
            with torch.no_grad():
                for batch in valid_loader:
                    input_features = batch["input_features"].to(DEVICE)
                    sensor_mask = batch["sensor_mask"].to(DEVICE)
                    pred_delta_targets = batch["pred_delta_targets"].to(DEVICE)
                    fail_targets = batch["fail_targets"].to(DEVICE)
                    rca_targets = batch["rca_targets"].to(DEVICE)

                    with autocast(enabled=AMP_ENABLED):  # Autocast also for validation if layers might benefit
                        pred_delta_output, fail_output_logits, rca_output_logits = model(input_features, sensor_mask)

                        active_pred_elements_val = sensor_mask.sum().clamp(min=1) * len(PRED_HORIZONS)
                        loss_pred_val = (huber_loss_fn(pred_delta_output, pred_delta_targets) * sensor_mask.unsqueeze(
                            -1)).sum() / active_pred_elements_val.clamp(min=1e-9)
                        loss_fail_val = focal_loss_fn_mean(fail_output_logits, fail_targets)
                        loss_rca_val = (focal_loss_fn_elementwise(rca_output_logits,
                                                                  rca_targets) * sensor_mask).sum() / sensor_mask.sum().clamp(
                            min=1e-9)

                        combined_loss_val = w_pred * loss_pred_val + w_fail * loss_fail_val + w_rca * loss_rca_val

                    if not (torch.isnan(combined_loss_val) or torch.isinf(combined_loss_val)):
                        total_loss_val += combined_loss_val.item()
                        total_pred_loss_val += loss_pred_val.item() if not (
                                    torch.isnan(loss_pred_val) or torch.isinf(loss_pred_val)) else 0
                        total_fail_loss_val += loss_fail_val.item() if not (
                                    torch.isnan(loss_fail_val) or torch.isinf(loss_fail_val)) else 0
                        total_rca_loss_val += loss_rca_val.item() if not (
                                    torch.isnan(loss_rca_val) or torch.isinf(loss_rca_val)) else 0

            num_b_val = max(1, len(valid_loader))
            print(
                f"E{epoch + 1} AvgValL: {total_loss_val / num_b_val:.3f} (P:{total_pred_loss_val / num_b_val:.3f} F:{total_fail_loss_val / num_b_val:.3f} R:{total_rca_loss_val / num_b_val:.3f})")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Preprocessor config (max_sensors_overall) saved to {PREPROCESSOR_SAVE_PATH}")


if __name__ == '__main__':
    train_model()
