# -*- coding: utf-8 -*-
"""
Transformer for Multivariate Time Series Anomaly Detection in Manufacturing (CLI Version)

This script demonstrates using a Transformer model to detect anomalies
in simulated multivariate sensor data from a manufacturing process,
suitable for command-line execution without graphical output.
The concept aligns with process mining by monitoring the 'state' (sensor readings)
of a process 'case' (e.g., a production batch) over time and detecting deviations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import time
import random
import sys # To flush output buffer

# --- Configuration ---
SEED = 42
N_FEATURES = 5         # Number of sensors/process parameters
SEQ_LENGTH = 60        # How many time steps in each sequence
PREDICT_STEPS = 1      # How many steps ahead to predict (or reconstruct) - set to 1 for reconstruction
BATCH_SIZE = 64
N_EPOCHS = 20          # Number of training epochs (Reduced for quicker CLI demo)
LEARNING_RATE = 0.001
D_MODEL = 128          # Transformer embedding dimension
N_HEADS = 8            # Number of attention heads
N_ENCODER_LAYERS = 3   # Number of Transformer encoder layers
DIM_FEEDFORWARD = 512  # Dimension of the feedforward network
DROPOUT = 0.1
THRESHOLD_PERCENTILE = 99 # Percentile for anomaly threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("--- Configuration ---")
print(f"Using device: {DEVICE}")
print(f"Number of features (sensors): {N_FEATURES}")
print(f"Sequence length: {SEQ_LENGTH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of epochs: {N_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Transformer d_model: {D_MODEL}")
print(f"Transformer heads: {N_HEADS}")
print(f"Transformer encoder layers: {N_ENCODER_LAYERS}")
print("-" * 30)


# --- 1. Generate Synthetic Manufacturing Data ---

def generate_data(n_samples, n_features, anomaly_ratio=0.05, anomaly_magnitude=2.0):
    """Generates synthetic multivariate time series data with anomalies."""
    time_steps = np.arange(0, n_samples, 1)
    data = np.zeros((n_samples, n_features))
    frequencies = np.random.uniform(0.02, 0.1, size=n_features)
    amplitudes = np.random.uniform(0.5, 1.5, size=n_features)
    phases = np.random.uniform(0, np.pi, size=n_features)

    # Base signals (representing normal sensor readings)
    for i in range(n_features):
        data[:, i] = amplitudes[i] * np.sin(frequencies[i] * time_steps + phases[i]) + \
                     np.random.normal(0, 0.1, n_samples) # Add some noise
        # Add some linear trend and interaction
        if i > 0:
           data[:, i] += 0.1 * data[:, i-1] + 0.0005 * time_steps

    # Inject anomalies
    anomaly_mask = np.zeros(n_samples, dtype=bool)
    n_anomalies = int(n_samples * anomaly_ratio)
    if n_anomalies == 0 and anomaly_ratio > 0: # Ensure at least one anomaly if ratio > 0
        n_anomalies = 1
    if n_anomalies > 0:
        anomaly_indices = np.random.choice(n_samples - 10, n_anomalies, replace=False) # Avoid end of series

        for idx in anomaly_indices:
            anomaly_mask[idx] = True
            anomaly_type = random.choice(['spike', 'shift', 'pattern_change'])
            affected_feature = random.randint(0, n_features - 1)
            anomaly_len = random.randint(1, max(5, int(0.01*n_samples))) # Anomaly duration
            end_idx = min(idx + anomaly_len, n_samples)

            if anomaly_type == 'spike':
                data[idx:end_idx, affected_feature] += anomaly_magnitude * np.random.choice([-1, 1]) * amplitudes[affected_feature]
            elif anomaly_type == 'shift':
                 data[idx:end_idx, affected_feature] += anomaly_magnitude * amplitudes[affected_feature] * 0.5
            elif anomaly_type == 'pattern_change':
                 data[idx:end_idx, affected_feature] += np.random.normal(0, 0.5 * amplitudes[affected_feature], size=end_idx-idx)

            anomaly_mask[idx:end_idx] = True # Mark the whole duration

    return data, anomaly_mask

# Generate data: Train (normal only), Test (normal + anomalies)
N_TRAIN_SAMPLES = 5000
N_TEST_SAMPLES = 2000

print("\n--- Generating Data ---")
train_data_raw, _ = generate_data(N_TRAIN_SAMPLES, N_FEATURES, anomaly_ratio=0) # No anomalies in training
test_data_raw, test_anomaly_mask_raw = generate_data(N_TEST_SAMPLES, N_FEATURES, anomaly_ratio=0.1, anomaly_magnitude=2.5)

print(f"Generated training data shape: {train_data_raw.shape}")
print(f"Generated test data shape: {test_data_raw.shape}")
print(f"Number of anomalous points in raw test set: {test_anomaly_mask_raw.sum()}")
print("-" * 30)


# --- 2. Preprocess Data ---
print("\n--- Preprocessing Data ---")

# Scale data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data_raw)
test_data_scaled = scaler.transform(test_data_raw) # Use the same scaler fitted on train data
print("Data scaled using MinMaxScaler (fitted on training data).")

def create_sequences(data, seq_length):
    """Creates overlapping sequences from time series data."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i:(i + seq_length)] # Target is the sequence itself for reconstruction
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_data_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_data_scaled, SEQ_LENGTH)

# Create sequence-level anomaly labels for evaluation
test_anomaly_labels_seq = []
for i in range(len(test_data_raw) - SEQ_LENGTH):
    is_anomalous = np.any(test_anomaly_mask_raw[i : i + SEQ_LENGTH])
    test_anomaly_labels_seq.append(is_anomalous)
test_anomaly_labels_seq = np.array(test_anomaly_labels_seq)

print(f"Training sequences shape: {X_train.shape}, {y_train.shape}")
print(f"Test sequences shape: {X_test.shape}, {y_test.shape}")
print(f"Number of anomalous sequences in test set (ground truth): {sum(test_anomaly_labels_seq)}")

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
print("Data converted to PyTorch tensors.")

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("DataLoaders created.")
print("-" * 30)


# --- 3. Define Transformer Model ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerAnomalyDetector(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_encoder_layers, dim_feedforward, dropout=0.1, seq_len=SEQ_LENGTH):
        super(TransformerAnomalyDetector, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.input_embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
        self.output_layer = nn.Linear(d_model, n_features)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.output_layer.bias.data.zero_()

    def forward(self, src, src_mask=None):
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)
        embedded_src = embedded_src.transpose(0, 1)
        pos_encoded_src = self.pos_encoder(embedded_src)
        pos_encoded_src = pos_encoded_src.transpose(0, 1)
        encoder_output = self.transformer_encoder(pos_encoded_src, mask=src_mask)
        output = self.output_layer(encoder_output)
        return output

# Instantiate the model
model = TransformerAnomalyDetector(
    n_features=N_FEATURES,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_encoder_layers=N_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    seq_len=SEQ_LENGTH
).to(DEVICE)

print("\n--- Model Architecture ---")
# print(model) # Optionally print the full model structure
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {num_params:,}")
print("-" * 30)


# --- 4. Train the Model ---

criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n--- Training ---")
start_time = time.time()

train_losses = []
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        reconstructions = model(batch_x)
        loss = criterion(reconstructions, batch_y).mean() # Average loss for the batch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1
        # Print progress within epoch if desired (can be verbose)
        # if batch_count % 50 == 0:
        #     print(f"Epoch [{epoch+1}/{N_EPOCHS}], Batch [{batch_count}/{len(train_loader)}], Batch Loss: {loss.item():.6f}", end='\r')
        #     sys.stdout.flush()


    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{N_EPOCHS}], Average Loss: {epoch_loss:.6f}")
    sys.stdout.flush() # Ensure epoch loss is printed immediately

training_time = time.time() - start_time
print(f"\n--- Training Finished ---")
print(f"Total Training Time: {training_time:.2f} seconds")
print("-" * 30)


# --- 5. Perform Inference & Anomaly Detection ---

print("\n--- Anomaly Detection ---")
model.eval() # Set model to evaluation mode

# 5.1 Calculate Reconstruction Errors on Training Data (to find threshold)
print("Calculating reconstruction errors on training data...")
train_reconstruction_errors = []
with torch.no_grad():
    for batch_x, batch_y in DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False):
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        reconstructions = model(batch_x)
        error = criterion(reconstructions, batch_y)
        error_per_sequence = error.mean(dim=(1, 2))
        train_reconstruction_errors.extend(error_per_sequence.cpu().numpy())

train_reconstruction_errors = np.array(train_reconstruction_errors)
print(f"Calculated reconstruction errors for {len(train_reconstruction_errors)} training sequences.")

# 5.2 Determine Anomaly Threshold
anomaly_threshold = np.percentile(train_reconstruction_errors, THRESHOLD_PERCENTILE)
print(f"Anomaly Threshold ({THRESHOLD_PERCENTILE}th percentile of training errors): {anomaly_threshold:.6f}")

# 5.3 Calculate Reconstruction Errors on Test Data
print("Calculating reconstruction errors on test data...")
test_reconstruction_errors = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        reconstructions = model(batch_x)
        error = criterion(reconstructions, batch_y)
        error_per_sequence = error.mean(dim=(1, 2))
        test_reconstruction_errors.extend(error_per_sequence.cpu().numpy())

test_reconstruction_errors = np.array(test_reconstruction_errors)
print(f"Calculated reconstruction errors for {len(test_reconstruction_errors)} test sequences.")

# 5.4 Identify Anomalies in Test Data
predicted_anomalies = test_reconstruction_errors > anomaly_threshold
print(f"Number of sequences predicted as anomalies in test set: {predicted_anomalies.sum()}")
print(f"Actual number of anomalous sequences in test set: {sum(test_anomaly_labels_seq)}")
print("-" * 30)

# --- 6. Evaluate Results ---

# Basic Evaluation (if ground truth labels are available)
true_positives = np.sum(predicted_anomalies & test_anomaly_labels_seq)
false_positives = np.sum(predicted_anomalies & ~test_anomaly_labels_seq)
false_negatives = np.sum(~predicted_anomalies & test_anomaly_labels_seq)
true_negatives = np.sum(~predicted_anomalies & ~test_anomaly_labels_seq)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n--- Evaluation Metrics (Sequence Level) ---")
print(f"True Positives (TP):  {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")
print(f"True Negatives (TN):  {true_negatives}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1-Score:             {f1_score:.4f}")
print("-" * 30)

# Optional: Print sample predictions
print("\n--- Sample Test Sequence Predictions ---")
num_samples_to_show = 15
print(f"{'Seq Index':<10} {'Recon Error':<15} {'Threshold':<15} {'Predicted':<10} {'Actual':<10}")
print("-" * 60)
indices_to_show = np.random.choice(len(test_reconstruction_errors), num_samples_to_show, replace=False)
indices_to_show.sort() # Show in order
for i in indices_to_show:
    error = test_reconstruction_errors[i]
    pred_label = "Anomaly" if predicted_anomalies[i] else "Normal"
    actual_label = "Anomaly" if test_anomaly_labels_seq[i] else "Normal"
    print(f"{i:<10} {error:<15.6f} {anomaly_threshold:<15.6f} {pred_label:<10} {actual_label:<10}")
print("-" * 60)


print("\n--- Script Finished ---")