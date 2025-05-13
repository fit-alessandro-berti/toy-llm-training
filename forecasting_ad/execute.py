# -*- coding: utf-8 -*-
"""
Transformer for Multivariate Time Series Forecasting in Manufacturing

This script demonstrates a complete example (training + inference) of using
a Transformer architecture with PyTorch for forecasting multivariate time series
data, simulating a manufacturing process scenario. It connects process mining
concepts (activity sequences) with sensor data.

Designed for command-line execution, no graphical output.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import math
import time
import random

# 1. Configuration & Parameters
# ---------------------------
# Data Simulation Parameters
N_SAMPLES = 1000  # Total number of time steps in the simulated process
N_MACHINES = 3    # Number of machines in the factory
ACTIVITIES = ['Idle', 'Setup', 'Processing_A', 'Processing_B', 'Maintenance', 'Quality_Check']

# Model Parameters
SEQ_LENGTH = 20      # Input sequence length (lookback window)
PRED_LENGTH = 5      # Prediction horizon (how many steps ahead to forecast)
D_MODEL = 64         # Dimension of the model (embedding size, must be divisible by N_HEAD)
N_HEAD = 4           # Number of attention heads in the Transformer
N_LAYERS = 3         # Number of Transformer encoder layers
D_FF = 128           # Dimension of the feedforward network in Transformer
DROPOUT = 0.1        # Dropout rate
OUTPUT_FEATURES = 2  # Number of features to forecast (e.g., Sensor 1, Quality)

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 10 # Reduced for quicker demonstration
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"--- Configuration ---")
print(f"Sequence Length: {SEQ_LENGTH}, Prediction Length: {PRED_LENGTH}")
print(f"Model Dimension: {D_MODEL}, Heads: {N_HEAD}, Layers: {N_LAYERS}")
print(f"Output Features: {OUTPUT_FEATURES}")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
print("-" * 20)

# 2. Data Simulation
# ------------------
def simulate_manufacturing_data(n_samples, n_machines, activities):
    """Generates synthetic multivariate time series data for manufacturing."""
    data = []
    current_activity = {m: 'Idle' for m in range(n_machines)}
    sensor1 = {m: 50.0 for m in range(n_machines)}
    sensor2 = {m: 0.1 for m in range(n_machines)}
    quality = {m: 95.0 for m in range(n_machines)}

    activity_encoder = LabelEncoder()
    activity_encoder.fit(activities)
    n_activities = len(activities)

    print("Simulating data...")
    for i in range(n_samples):
        for machine_id in range(n_machines):
            # Simulate activity transitions (simple state machine logic)
            rand_val = random.random()
            last_activity = current_activity[machine_id]

            if last_activity == 'Idle':
                if rand_val < 0.3: current_activity[machine_id] = 'Setup'
            elif last_activity == 'Setup':
                 if rand_val < 0.8: current_activity[machine_id] = random.choice(['Processing_A', 'Processing_B'])
                 else: current_activity[machine_id] = 'Idle' # Failed setup
            elif last_activity.startswith('Processing'):
                 if rand_val < 0.7: current_activity[machine_id] = 'Quality_Check'
                 elif rand_val < 0.8: current_activity[machine_id] = 'Maintenance' # Needs maintenance
                 else: pass # Continue processing
            elif last_activity == 'Quality_Check':
                 if rand_val < 0.9: current_activity[machine_id] = 'Idle' # Pass
                 else: current_activity[machine_id] = 'Maintenance' # Fail -> Maintenance
            elif last_activity == 'Maintenance':
                 if rand_val < 0.7: current_activity[machine_id] = 'Idle' # Fixed

            activity_num = activity_encoder.transform([current_activity[machine_id]])[0]

            # Simulate sensor readings based on activity
            s1_noise = np.random.normal(0, 2)
            s2_noise = np.random.normal(0, 0.05)
            q_noise = np.random.normal(0, 1)

            if current_activity[machine_id] == 'Processing_A':
                sensor1[machine_id] += 1.5 + s1_noise
                sensor2[machine_id] += 0.02 + s2_noise
                quality[machine_id] -= 0.5 + abs(q_noise) # Quality degrades slightly
            elif current_activity[machine_id] == 'Processing_B':
                sensor1[machine_id] += 0.8 + s1_noise
                sensor2[machine_id] -= 0.01 + s2_noise
                quality[machine_id] -= 0.3 + abs(q_noise)
            elif current_activity[machine_id] == 'Maintenance':
                sensor1[machine_id] = 50.0 + s1_noise # Reset somewhat
                sensor2[machine_id] = 0.1 + s2_noise
                quality[machine_id] = 98.0 + q_noise # Improve quality after maintenance
            elif current_activity[machine_id] == 'Setup':
                sensor1[machine_id] += s1_noise
                sensor2[machine_id] += s2_noise
                quality[machine_id] += q_noise * 0.1
            else: # Idle, Quality Check
                 sensor1[machine_id] += s1_noise * 0.5
                 sensor2[machine_id] += s2_noise * 0.5
                 quality[machine_id] += q_noise * 0.1

            # Clamp values to reasonable ranges
            sensor1[machine_id] = max(30, min(100, sensor1[machine_id]))
            sensor2[machine_id] = max(0, min(1, sensor2[machine_id]))
            quality[machine_id] = max(70, min(100, quality[machine_id]))

            data.append({
                'timestamp': i,
                'machine_id': machine_id,
                'activity_code': activity_num,
                'sensor1': sensor1[machine_id],
                'sensor2': sensor2[machine_id],
                'quality': quality[machine_id]
            })

    df = pd.DataFrame(data)
    print("Data simulation complete.")
    return df, activity_encoder, n_activities

# Generate data
df, activity_encoder, n_activities = simulate_manufacturing_data(N_SAMPLES, N_MACHINES, ACTIVITIES)

print("\n--- Sampled Data Head ---")
print(df.head())
print("-" * 20)

# 3. Data Preprocessing
# ---------------------
print("Preprocessing data...")
# Separate data per machine for easier sequence creation
data_per_machine = [df[df['machine_id'] == m].copy() for m in range(N_MACHINES)]

# Normalize numerical features (Sensor1, Sensor2, Quality)
# We'll fit scalers globally, but apply per machine for sequence creation
scalers = {}
numerical_cols = ['sensor1', 'sensor2', 'quality']
target_cols = ['sensor1', 'quality'] # Features we want to predict
target_indices = [numerical_cols.index(col) for col in target_cols] # Indices within numerical_cols

for col in numerical_cols:
    scaler = MinMaxScaler()
    # Fit on the entire column across all machines
    scalers[col] = scaler.fit(df[[col]])
    # Apply scaling per machine
    for m_df in data_per_machine:
        m_df[f'{col}_norm'] = scalers[col].transform(m_df[[col]])

# Create sequences
def create_sequences(machine_data, seq_length, pred_length, numerical_cols_norm, target_indices):
    """Creates input sequences and corresponding target sequences."""
    X, y = [], []
    activity_codes = machine_data['activity_code'].values
    numerical_data = machine_data[[f'{col}_norm' for col in numerical_cols]].values

    if len(machine_data) < seq_length + pred_length:
        return np.array([]), np.array([]) # Not enough data for this machine

    for i in range(len(machine_data) - seq_length - pred_length + 1):
        # Input sequence: activity codes + normalized numerical data
        seq_activities = activity_codes[i : i + seq_length]
        seq_numerical = numerical_data[i : i + seq_length]
        # Combine activity code as another feature dimension
        # Shape: (seq_length, num_features + 1)
        input_seq = np.hstack((seq_activities.reshape(-1, 1), seq_numerical))

        # Target sequence: future values of selected numerical features
        target_seq = numerical_data[i + seq_length : i + seq_length + pred_length, target_indices]
        # Shape: (pred_length, num_target_features)

        X.append(input_seq)
        y.append(target_seq)

    return np.array(X), np.array(y)

# Combine sequences from all machines
all_X, all_y = [], []
for m_df in data_per_machine:
    X_m, y_m = create_sequences(m_df, SEQ_LENGTH, PRED_LENGTH, numerical_cols, target_indices)
    if X_m.size > 0: # Check if sequences were created
        all_X.append(X_m)
        all_y.append(y_m)

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

print(f"Created sequences: X shape={X.shape}, y shape={y.shape}") # X: (n_sequences, seq_length, n_features+1), y: (n_sequences, pred_length, n_output_features)

# Split data (simple split for demonstration)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
print("-" * 20)

# Create DataLoader
class ManufacturingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ManufacturingDataset(X_train, y_train)
test_dataset = ManufacturingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 4. Transformer Model Definition
# -------------------------------
class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe) # Not a model parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerForecaster(nn.Module):
    def __init__(self, input_features, num_activities, d_model, nhead, num_encoder_layers,
                 dim_feedforward, dropout, output_features, seq_length, pred_length):
        super(TransformerForecaster, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.output_features = output_features

        # Embedding for the activity code (categorical feature)
        self.activity_embedding = nn.Embedding(num_activities, d_model // 4) # Allocate part of d_model

        # Linear layer to project combined input features to d_model
        # Input features = (numerical features) + (embedding dim)
        # Input features = (input_features - 1) + d_model // 4
        combined_feature_dim = (input_features - 1) + (d_model // 4)
        self.input_projection = nn.Linear(combined_feature_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # Output layer to predict the target features for the prediction horizon
        # We flatten the output of the transformer and predict all steps at once
        self.output_linear = nn.Linear(d_model * seq_length, pred_length * output_features)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.activity_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()

    def forward(self, src):
        """
        Args:
            src: Input tensor, shape [batch_size, seq_len, input_features]
                 input_features includes activity_code + numerical features
        """
        # Separate activity code and numerical features
        activity_codes = src[:, :, 0].long() # Shape: [batch_size, seq_len]
        numerical_features = src[:, :, 1:]   # Shape: [batch_size, seq_len, num_numerical_features]

        # Embed activities
        activity_embed = self.activity_embedding(activity_codes) # Shape: [batch_size, seq_len, d_model//4]

        # Combine embedded activities and numerical features
        combined_features = torch.cat((activity_embed, numerical_features), dim=-1) # Shape: [batch_size, seq_len, combined_feature_dim]

        # Project to d_model
        src_proj = self.input_projection(combined_features) # Shape: [batch_size, seq_len, d_model]
        src_proj = src_proj * math.sqrt(self.d_model) # Scale embedding

        # Add positional encoding (needs shape [seq_len, batch_size, d_model]) -> Transpose before PE
        # Note: TransformerEncoderLayer expects batch_first=True, so input should be [batch_size, seq_len, d_model]
        # However, PositionalEncoding expects [seq_len, batch_size, d_model].
        # Let's adapt PositionalEncoding or the input flow. Adapting input flow:
        src_permuted = src_proj.permute(1, 0, 2) # Shape: [seq_len, batch_size, d_model]
        src_pos = self.pos_encoder(src_permuted)
        src_pos = src_pos.permute(1, 0, 2) # Back to [batch_size, seq_len, d_model]

        # Pass through Transformer Encoder
        # src_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_length).to(DEVICE) # Optional: If causality needed
        # output = self.transformer_encoder(src_pos, mask=src_mask)
        output = self.transformer_encoder(src_pos) # Shape: [batch_size, seq_len, d_model]

        # Flatten the output and predict future steps
        output_flat = output.reshape(output.size(0), -1) # Shape: [batch_size, seq_len * d_model]
        prediction = self.output_linear(output_flat) # Shape: [batch_size, pred_length * output_features]

        # Reshape prediction to [batch_size, pred_length, output_features]
        prediction = prediction.view(output.size(0), self.pred_length, self.output_features)

        return prediction

# Instantiate the model
input_dim = X_train.shape[2] # Number of features in the input sequence (activity_code + numerical)
model = TransformerForecaster(
    input_features=input_dim,
    num_activities=n_activities,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_encoder_layers=N_LAYERS,
    dim_feedforward=D_FF,
    dropout=DROPOUT,
    output_features=OUTPUT_FEATURES,
    seq_length=SEQ_LENGTH,
    pred_length=PRED_LENGTH
).to(DEVICE)

print("\n--- Model Architecture ---")
print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {num_params:,}")
print("-" * 20)

# 5. Training Loop
# ----------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Learning rate scheduler

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
        optimizer.step()

        epoch_loss += loss.item()

        # Optional: Print batch progress
        # if batch_idx % 50 == 0:
        #     print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_epoch_loss = epoch_loss / len(train_loader)
    scheduler.step() # Update learning rate
    epoch_duration = time.time() - epoch_start_time

    # Evaluate on test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
    avg_test_loss = test_loss / len(test_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_epoch_loss:.4f} | Test Loss: {avg_test_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_duration:.2f}s")

total_training_time = time.time() - start_time
print(f"\nTraining finished in {total_training_time:.2f} seconds.")
print("-" * 20)


# 6. Inference/Forecasting Example
# --------------------------------
print("Performing inference on a sample from the test set...")
model.eval()

# Get a sample from the test set
sample_idx = random.randint(0, len(X_test) - 1)
input_sequence, true_target = test_dataset[sample_idx]
input_sequence = input_sequence.unsqueeze(0).to(DEVICE) # Add batch dimension and move to device
true_target = true_target.cpu().numpy() # Keep target on CPU for comparison

# Make prediction
with torch.no_grad():
    prediction_norm = model(input_sequence) # Shape: [1, pred_length, output_features]

# Inverse transform the prediction and the true target to original scale
prediction_norm = prediction_norm.squeeze(0).cpu().numpy() # Remove batch dim, move to CPU

# Create dummy arrays with correct shape for inverse transform
# Prediction needs shape (pred_length, num_numerical_features)
# True target needs shape (pred_length, num_numerical_features)
num_numerical_features = len(numerical_cols)
pred_full = np.zeros((PRED_LENGTH, num_numerical_features))
true_full = np.zeros((PRED_LENGTH, num_numerical_features))

# Place the predicted/true values into the correct columns
pred_full[:, target_indices] = prediction_norm
true_full[:, target_indices] = true_target

# Inverse transform using the fitted scalers
prediction_rescaled = np.zeros_like(pred_full)
true_rescaled = np.zeros_like(true_full)

for i, col in enumerate(numerical_cols):
    if col in target_cols: # Only inverse transform columns that were predicted
        col_idx_in_target = target_cols.index(col) # Find index in the output
        prediction_rescaled[:, i] = scalers[col].inverse_transform(pred_full[:, i].reshape(-1, 1)).flatten()
        true_rescaled[:, i] = scalers[col].inverse_transform(true_full[:, i].reshape(-1, 1)).flatten()

# Extract only the target columns for printing
predicted_values = prediction_rescaled[:, target_indices]
true_values = true_rescaled[:, target_indices]

# Print the last few steps of the input and the forecast
print("\n--- Inference Example ---")
print(f"Input Sequence (Last 5 steps, showing Activity Code and Normalized Sensor/Quality):")
# Convert input back to numpy for easier printing
input_sequence_np = input_sequence.squeeze(0).cpu().numpy()
for i in range(max(0, SEQ_LENGTH - 5), SEQ_LENGTH):
    activity_name = activity_encoder.inverse_transform([int(input_sequence_np[i, 0])])[0]
    sensor1_norm = input_sequence_np[i, 1]
    sensor2_norm = input_sequence_np[i, 2]
    quality_norm = input_sequence_np[i, 3]
    print(f"  Step {i-SEQ_LENGTH}: Act={activity_name:<12} S1_norm={sensor1_norm:.2f} S2_norm={sensor2_norm:.2f} Q_norm={quality_norm:.2f}")

print(f"\nForecast for next {PRED_LENGTH} steps (Rescaled):")
print(f"Target Features: {target_cols}")
print("       Predicted |    True")
print("----------------------------")
for i in range(PRED_LENGTH):
    pred_str = f"[{predicted_values[i, 0]:>7.2f}, {predicted_values[i, 1]:>7.2f}]"
    true_str = f"[{true_values[i, 0]:>7.2f}, {true_values[i, 1]:>7.2f}]"
    print(f"Step +{i+1}: {pred_str} | {true_str}")

print("-" * 20)
print("Script finished.")
