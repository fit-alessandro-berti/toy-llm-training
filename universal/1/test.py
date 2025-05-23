import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- Configuration (MUST MATCH TRAINING SCRIPT) ---
# Data paths
BASE_DATA_DIR = "generated_time_series_data"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # Test on VALIDATION data

# Model & Training Parameters (from training script)
SEQ_LEN = 64
PRED_HORIZONS = [1, 3, 5]
FAIL_HORIZONS = [3, 5, 10]
RCA_FAILURE_LOOKAHEAD = FAIL_HORIZONS[0]

MAX_SENSORS_CAP = 20  # From training script

# Architectural Params (from training script)
SENSOR_EMBED_DIM = 32
TRANSFORMER_DIM = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2
MOE_HIDDEN_DIM = 128
NUM_EXPERTS = 4
MOE_OUTPUT_DIM = 64

# Testing Params
BATCH_SIZE = 4  # Smaller batch size for testing to see more diverse samples if MAX_BATCHES_TO_TEST is small
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCHES_TO_TEST = 5  # Number of batches to run inference on and print results for
SAMPLES_PER_BATCH_TO_PRINT = 2  # Number of samples within each tested batch to print details for

MODEL_LOAD_PATH = "foundation_timeseries_model.pth"
PREPROCESSOR_LOAD_PATH = "preprocessor_config.npz"


# --- Data Handling (Replicated from training script for consistency) ---
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
            if not sensor_cols:
                print(f"Warning: No sensor columns found in {fp}. Skipping.")
                continue

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
                "filepath": fp  # Store filepath for reference
            })

            max_lookahead = max(max(self.pred_horizons), max(self.fail_horizons), self.rca_failure_lookahead)
            for i in range(len(df) - self.seq_len - max_lookahead + 1):
                self.window_indices.append((file_idx, i))
        print(f"Loaded {len(self.data_cache)} files, created {len(self.window_indices)} windows for testing.")

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        file_idx, window_start_idx = self.window_indices[idx]
        cached_item = self.data_cache[file_idx]

        features_full = cached_item["features"]
        failure_flags_full = cached_item["failure_flags"]
        num_actual_sensors = cached_item["num_actual_sensors"]
        filepath = cached_item["filepath"]

        input_seq_features = features_full[window_start_idx: window_start_idx + self.seq_len]

        padded_input_seq_features = np.zeros((self.seq_len, self.max_sensors_global), dtype=np.float32)
        padded_input_seq_features[:, :num_actual_sensors] = input_seq_features

        sensor_mask = np.zeros(self.max_sensors_global, dtype=np.float32)
        sensor_mask[:num_actual_sensors] = 1.0

        pred_targets = np.zeros((self.max_sensors_global, len(self.pred_horizons)), dtype=np.float32)
        for i, h in enumerate(self.pred_horizons):
            target_idx = window_start_idx + self.seq_len + h - 1
            if target_idx < len(features_full):
                pred_targets[:num_actual_sensors, i] = features_full[target_idx, :num_actual_sensors]

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
        if rca_fail_window_end <= len(failure_flags_full):
            if np.any(failure_flags_full[rca_fail_window_start: rca_fail_window_end]):
                is_imminent_failure = True

        if is_imminent_failure and num_actual_sensors > 0:
            failure_sub_window_features = features_full[rca_fail_window_start: rca_fail_window_end, :num_actual_sensors]
            if failure_sub_window_features.shape[0] > 0:
                current_input_window_features = features_full[window_start_idx: window_start_idx + self.seq_len,
                                                :num_actual_sensors]
                if current_input_window_features.shape[0] > 0:
                    input_means = np.mean(current_input_window_features, axis=0)
                    input_stds = np.std(current_input_window_features, axis=0)
                    input_stds[input_stds < 1e-6] = 1e-6

                    for s_idx in range(num_actual_sensors):
                        sensor_fail_values = failure_sub_window_features[:, s_idx]
                        if np.any(np.abs(sensor_fail_values - input_means[s_idx]) > 3 * input_stds[s_idx]):
                            rca_targets[s_idx] = 1.0

        return {
            "input_features": torch.from_numpy(padded_input_seq_features),
            "sensor_mask": torch.from_numpy(sensor_mask),
            "pred_targets": torch.from_numpy(pred_targets),  # Ground truth for comparison
            "fail_targets": torch.from_numpy(fail_targets),  # Ground truth
            "rca_targets": torch.from_numpy(rca_targets),  # Ground truth
            "filepath": filepath,  # For reference
            "window_start_idx": window_start_idx  # For reference
        }


# --- Model Architecture (Replicated from training script) ---
class SensorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return hn.squeeze(0)


class InterSensorTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers, max_sensors):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_sensors, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True,
                                                   dim_feedforward=embed_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        x = x + self.pos_encoder[:, :x.size(1), :]
        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class FoundationalTimeSeriesModel(nn.Module):
    def __init__(self, max_sensors, sensor_input_dim, sensor_embed_dim,
                 transformer_dim, transformer_nhead, transformer_nlayers,
                 moe_hidden_dim, num_experts, moe_output_dim,
                 pred_horizons_len, fail_horizons_len):
        super().__init__()
        self.max_sensors = max_sensors
        self.sensor_encoder = SensorEncoder(sensor_input_dim, sensor_embed_dim)

        if sensor_embed_dim != transformer_dim:
            self.transformer_input_proj = nn.Linear(sensor_embed_dim, transformer_dim)
        else:
            self.transformer_input_proj = nn.Identity()

        self.inter_sensor_transformer = InterSensorTransformer(
            embed_dim=transformer_dim,
            nhead=transformer_nhead,
            num_layers=transformer_nlayers,
            max_sensors=max_sensors
        )

        self.gating_network = GatingNetwork(transformer_dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(transformer_dim, moe_hidden_dim, moe_output_dim) for _ in range(num_experts)
        ])

        self.pred_head = nn.Linear(transformer_dim + moe_output_dim, pred_horizons_len)
        self.fail_head = nn.Linear(moe_output_dim, fail_horizons_len)
        self.rca_head = nn.Linear(transformer_dim + moe_output_dim, 1)

    def forward(self, x_features, sensor_mask):
        batch_size, seq_len, _ = x_features.shape
        x_reshaped = x_features.permute(0, 2, 1).unsqueeze(-1)
        x_reshaped = x_reshaped.reshape(batch_size * self.max_sensors, seq_len, 1)
        sensor_embeddings_flat = self.sensor_encoder(x_reshaped)
        sensor_embeddings = sensor_embeddings_flat.reshape(batch_size, self.max_sensors, -1)
        sensor_embeddings = sensor_embeddings * sensor_mask.unsqueeze(-1)
        projected_sensor_embeddings = self.transformer_input_proj(sensor_embeddings)
        transformer_padding_mask = (sensor_mask == 0)
        transformer_out = self.inter_sensor_transformer(projected_sensor_embeddings, transformer_padding_mask)
        expanded_sensor_mask = sensor_mask.unsqueeze(-1)
        summed_features = (transformer_out * expanded_sensor_mask).sum(dim=1)
        active_sensors_count = sensor_mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_features = summed_features / active_sensors_count
        gating_scores = self.gating_network(global_features)
        expert_outputs_list = [expert(global_features) for expert in self.experts]
        stacked_expert_outputs = torch.stack(expert_outputs_list, dim=1)
        moe_global_output = (gating_scores.unsqueeze(-1) * stacked_expert_outputs).sum(dim=1)
        moe_global_output_expanded = moe_global_output.unsqueeze(1).expand(-1, self.max_sensors, -1)
        per_sensor_combined_features = torch.cat([transformer_out, moe_global_output_expanded], dim=-1)
        pred_output = self.pred_head(per_sensor_combined_features)
        fail_output = self.fail_head(moe_global_output)
        rca_output = self.rca_head(per_sensor_combined_features).squeeze(-1)
        return pred_output, fail_output, rca_output


# --- Test Script Logic ---
def test_model():
    print(f"Using device: {DEVICE}")

    # Load preprocessor config
    try:
        preprocessor_config = np.load(PREPROCESSOR_LOAD_PATH)
        max_sensors_overall = int(preprocessor_config['max_sensors_overall'])  # Ensure it's an int
    except FileNotFoundError:
        print(f"Error: Preprocessor config file not found at {PREPROCESSOR_LOAD_PATH}. Exiting.")
        return
    except KeyError:
        print(f"Error: 'max_sensors_overall' not found in preprocessor config. Exiting.")
        return

    print(f"Loaded max_sensors_overall from preprocessor: {max_sensors_overall}")

    # Initialize dataset and dataloader for validation set
    valid_dataset = MultivariateTimeSeriesDataset(VALID_DIR, SEQ_LEN, PRED_HORIZONS, FAIL_HORIZONS,
                                                  RCA_FAILURE_LOOKAHEAD, max_sensors_overall)
    if len(valid_dataset) == 0:
        print(f"No data found in {VALID_DIR} or no valid windows created. Exiting.")
        return
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    model = FoundationalTimeSeriesModel(
        max_sensors=max_sensors_overall,
        sensor_input_dim=1,
        sensor_embed_dim=SENSOR_EMBED_DIM,
        transformer_dim=TRANSFORMER_DIM,
        transformer_nhead=TRANSFORMER_NHEAD,
        transformer_nlayers=TRANSFORMER_NLAYERS,
        moe_hidden_dim=MOE_HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        moe_output_dim=MOE_OUTPUT_DIM,
        pred_horizons_len=len(PRED_HORIZONS),
        fail_horizons_len=len(FAIL_HORIZONS)
    ).to(DEVICE)

    # Load trained model state
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

    # Inference loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            if batch_idx >= MAX_BATCHES_TO_TEST:
                print(f"\nReached MAX_BATCHES_TO_TEST ({MAX_BATCHES_TO_TEST}). Stopping inference.")
                break

            print(f"\n--- Processing Batch {batch_idx + 1} ---")
            input_features = batch["input_features"].to(DEVICE)
            sensor_mask = batch["sensor_mask"].to(DEVICE)
            # Ground truth targets for comparison
            pred_targets_gt = batch["pred_targets"].to(DEVICE)
            fail_targets_gt = batch["fail_targets"].to(DEVICE)
            rca_targets_gt = batch["rca_targets"].to(DEVICE)

            batch_s = input_features.size(0)  # Actual batch size for this batch

            pred_output, fail_output, rca_output = model(input_features, sensor_mask)

            # Print results for a few samples in the batch
            num_samples_this_batch_to_print = min(SAMPLES_PER_BATCH_TO_PRINT, batch_s)
            for i in range(num_samples_this_batch_to_print):
                sample_filepath = batch["filepath"][i]  # Get filepath for this specific sample
                sample_window_start = batch["window_start_idx"][i].item()

                print(
                    f"\n  --- Sample {i + 1} from Batch {batch_idx + 1} (File: {os.path.basename(sample_filepath)}, Window Start: {sample_window_start}) ---")
                active_sensor_indices = torch.where(sensor_mask[i] == 1.0)[0].cpu().tolist()
                num_active_sensors_sample = len(active_sensor_indices)

                if num_active_sensors_sample == 0:
                    print("    No active sensors in this sample.")
                    continue

                # 1. Forecasting
                print("    Forecasting Task:")
                for h_idx, horizon_steps in enumerate(PRED_HORIZONS):
                    print(f"      Horizon: +{horizon_steps} steps")
                    sensors_to_print_forecast = active_sensor_indices[:min(2, num_active_sensors_sample)]
                    if not sensors_to_print_forecast: print("        (No active sensors to show forecast for)")
                    for s_global_idx in sensors_to_print_forecast:
                        actual = pred_targets_gt[i, s_global_idx, h_idx].item()
                        predicted = pred_output[i, s_global_idx, h_idx].item()
                        print(f"        Sensor {s_global_idx + 1}: Actual={actual:.4f}, Predicted={predicted:.4f}")

                # 2. Failure Prediction
                print("    Failure Prediction Task:")
                predicted_fail_probs = torch.sigmoid(fail_output[i]).cpu().tolist()
                actual_fail_status = fail_targets_gt[i].cpu().tolist()
                for h_idx, horizon_steps in enumerate(FAIL_HORIZONS):
                    print(
                        f"      Horizon: within next {horizon_steps} steps - Actual: {actual_fail_status[h_idx]:.0f}, Predicted Prob: {predicted_fail_probs[h_idx]:.4f}")

                # 3. Root Cause Analysis
                print("    Root Cause Analysis Task:")
                # Condition to show RCA: if ground truth failure or high predicted prob for shortest RCA horizon
                idx_rca_horizon = FAIL_HORIZONS.index(RCA_FAILURE_LOOKAHEAD)
                rca_condition_met = predicted_fail_probs[idx_rca_horizon] > 0.6 or actual_fail_status[
                    idx_rca_horizon] == 1.0

                if rca_condition_met:
                    rca_scores_sample_all_sensors = torch.sigmoid(rca_output[i]).cpu().tolist()
                    active_rca_scores = []
                    for s_global_idx in active_sensor_indices:
                        # s_global_idx is 0-indexed here
                        active_rca_scores.append((s_global_idx + 1, rca_scores_sample_all_sensors[s_global_idx]))

                    active_rca_scores.sort(key=lambda x: x[1], reverse=True)
                    print(f"      (RCA condition met for {RCA_FAILURE_LOOKAHEAD}-step horizon)")
                    print(f"      Top contributing sensors (Sensor #, RCA Score):")

                    if not active_rca_scores: print("        (No active sensors for RCA)")
                    for s_id, score in active_rca_scores[:min(5, len(active_rca_scores))]:
                        actual_rca_target_for_sensor = rca_targets_gt[i, s_id - 1].item()  # s_id is 1-based
                        print(
                            f"        Sensor {s_id}: Score={score:.4f} (Actual RCA Target: {actual_rca_target_for_sensor:.0f})")
                else:
                    print(f"      (RCA condition not met for this sample for {RCA_FAILURE_LOOKAHEAD}-step horizon)")

    print("\n--- Testing complete ---")


if __name__ == '__main__':
    test_model()
