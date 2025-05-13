# -*- coding: utf-8 -*-
"""
Transformer for Predictive Maintenance on Multivariate Time Series Data.

This script demonstrates a Transformer model for predicting equipment failures
in a manufacturing setting, using synthetic multivariate time series data
that includes sensor readings and process event indicators.

Concepts:
- Predictive Maintenance: Predict failures before they happen.
- Multivariate Time Series: Data with multiple variables recorded over time.
- Transformer Model: A deep learning architecture effective for sequence data.
- Process Mining Aspects: Incorporates event-based features alongside sensor data.
- Command-Line Execution: Designed to run without graphical output.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import math
import random
import time

# 1. Configuration Parameters
# ---------------------------
# Data Generation
N_SAMPLES = 1000         # Total number of time series samples to generate
SEQ_LENGTH = 50          # Length of each input sequence (time steps)
N_FEATURES = 5           # Number of features (e.g., 3 sensors + 2 event types)
PREDICTION_HORIZON = 5   # Predict failure within the next k steps
FAILURE_PROB_BASELINE = 0.05 # Baseline probability of failure sequence
FAILURE_INDICATOR_STRENGTH = 0.7 # How strongly indicators predict failure

# Model Parameters
D_MODEL = 64             # Embedding dimension (should be divisible by N_HEAD)
N_HEAD = 4               # Number of attention heads
N_ENCODER_LAYERS = 2     # Number of Transformer encoder layers
DIM_FEEDFORWARD = 128    # Dimension of the feedforward network
DROPOUT = 0.1            # Dropout rate

# Training Parameters
N_EPOCHS = 10            # Number of training epochs
BATCH_SIZE = 32          # Batch size for training
LEARNING_RATE = 0.001    # Learning rate for the optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"--- Configuration ---")
print(f"Device: {DEVICE}")
print(f"Sequence Length: {SEQ_LENGTH}")
print(f"Number of Features: {N_FEATURES}")
print(f"Prediction Horizon: {PREDICTION_HORIZON}")
print(f"Model Dimension: {D_MODEL}")
print(f"Number of Heads: {N_HEAD}")
print(f"Number of Encoder Layers: {N_ENCODER_LAYERS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {N_EPOCHS}")
print("-" * 20)

# 2. Data Generation
# ------------------
# Simulate manufacturing data: 3 sensor readings (continuous) + 2 event types (binary)
# Target: Binary classification - predict if failure occurs within PREDICTION_HORIZON steps

def generate_synthetic_data(n_samples, seq_length, n_features, horizon, failure_prob, indicator_strength):
    """Generates synthetic multivariate time series data."""
    print("Generating synthetic data...")
    data = []
    labels = []

    for i in range(n_samples):
        # Decide if this sequence will lead to a failure
        is_failure_sequence = random.random() < failure_prob
        sequence = np.zeros((seq_length + horizon, n_features))
        label = 0 # Default: No failure

        # Generate baseline sensor data (random walk)
        sequence[:, 0] = np.cumsum(np.random.randn(seq_length + horizon)) * 0.1 # Temp
        sequence[:, 1] = np.cumsum(np.random.randn(seq_length + horizon)) * 0.1 + 5 # Pressure
        sequence[:, 2] = np.abs(np.cumsum(np.random.randn(seq_length + horizon)) * 0.05) # Vibration

        # Generate event data (sparse binary)
        sequence[:, 3] = (np.random.rand(seq_length + horizon) < 0.1).astype(float) # Event Type A (e.g., manual stop)
        sequence[:, 4] = (np.random.rand(seq_length + horizon) < 0.05).astype(float) # Event Type B (e.g., error code)

        if is_failure_sequence:
            label = 1
            # Inject failure indicators into the *input* sequence part
            failure_point = seq_length # Failure happens right after the input sequence ends
            start_indicator = max(0, failure_point - horizon - int(seq_length * 0.3)) # Indicators appear before failure

            # Increase sensor readings or specific event frequency before failure
            indicator_mask = np.zeros(seq_length + horizon)
            indicator_mask[start_indicator:failure_point] = 1

            # Modify data based on indicator strength
            sequence[:, 0] += np.random.randn(seq_length + horizon) * indicator_strength * 5 * indicator_mask # Temp spike
            sequence[:, 2] += np.random.rand(seq_length + horizon) * indicator_strength * 2 * indicator_mask # Vibration increase
            sequence[:, 4] += (np.random.rand(seq_length + horizon) < indicator_strength * 0.5).astype(float) * indicator_mask # More Error codes

        # Extract input sequence and label
        input_sequence = sequence[:seq_length, :]
        # Label is 1 if failure occurs within the horizon *after* the input sequence
        # For simplicity here, we tied the label directly to is_failure_sequence generation
        # A more realistic check would be: label = 1 if np.sum(failure_signal[seq_length:seq_length+horizon]) > 0 else 0

        data.append(input_sequence)
        labels.append(label)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples} samples...")

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64) # Use int64 for CrossEntropyLoss

    print("Data generation complete.")
    print(f"Data shape: {data.shape}") # (N_SAMPLES, SEQ_LENGTH, N_FEATURES)
    print(f"Labels shape: {labels.shape}") # (N_SAMPLES,)
    print(f"Failure sequences: {np.sum(labels)}/{n_samples} ({np.sum(labels)/n_samples*100:.2f}%)")
    print("-" * 20)
    return data, labels

# Generate data
data, labels = generate_synthetic_data(
    N_SAMPLES, SEQ_LENGTH, N_FEATURES, PREDICTION_HORIZON,
    FAILURE_PROB_BASELINE, FAILURE_INDICATOR_STRENGTH
)

# 3. Data Preprocessing & Loading
# -------------------------------
# Simple Scaling (StandardScaler is often better for real data)
data_mean = np.mean(data, axis=(0, 1))
data_std = np.std(data, axis=(0, 1))
# Avoid division by zero for constant features (like sparse events)
data_std[data_std == 0] = 1
data = (data - data_mean) / data_std

# Convert to PyTorch Tensors
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long) # Use torch.long for CrossEntropyLoss

# Create Dataset and DataLoader
dataset = TensorDataset(data_tensor, labels_tensor)

# Split data (simple split, consider time-based split for real scenarios)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Data loaded and preprocessed.")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print("-" * 20)

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
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    """Transformer model for time series classification."""
    def __init__(self, n_features, d_model, n_head, n_encoder_layers, dim_feedforward, dropout, num_classes=2):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        # Input embedding layer: projects n_features to d_model
        self.input_embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Expect input as (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers
        )

        # Output layer: Maps the output of the transformer to class probabilities
        # We use the output corresponding to the *first* token (like BERT's [CLS])
        # or average pooling over the sequence. Let's use average pooling.
        self.output_layer = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Input tensor, shape [batch_size, seq_len, n_features]
            src_mask: Not typically used for encoder-only models unless specific masking is needed.
            src_key_padding_mask: Mask for padding tokens (if used). Shape [batch_size, seq_len].

        Returns:
            Output tensor, shape [batch_size, num_classes]
        """
        # 1. Embed Input
        # src shape: [batch_size, seq_len, n_features]
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)
        # embedded_src shape: [batch_size, seq_len, d_model]

        # 2. Add Positional Encoding
        # TransformerEncoderLayer expects (seq_len, batch_size, d_model) if batch_first=False
        # but we use batch_first=True, so input is (batch_size, seq_len, d_model)
        pos_encoded_src = self.pos_encoder(embedded_src.transpose(0, 1)).transpose(0, 1)
        # pos_encoded_src shape: [batch_size, seq_len, d_model]

        # 3. Pass through Transformer Encoder
        # src_key_padding_mask: Tensor of shape (N, S) where N is the batch size, S is the source sequence length.
        # If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged.
        # If a FloatTensor is provided, it will be added to the attention weight.
        # We don't have padding here, so we don't need a mask.
        transformer_output = self.transformer_encoder(pos_encoded_src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # transformer_output shape: [batch_size, seq_len, d_model]

        # 4. Pooling and Final Classification
        # Use mean pooling over the sequence dimension
        pooled_output = transformer_output.mean(dim=1) # Average across the sequence length
        # pooled_output shape: [batch_size, d_model]

        output = self.output_layer(pooled_output)
        # output shape: [batch_size, num_classes]
        return output

# Instantiate the model
model = TransformerPredictor(
    n_features=N_FEATURES,
    d_model=D_MODEL,
    n_head=N_HEAD,
    n_encoder_layers=N_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    num_classes=2 # Binary classification (Failure / No Failure)
).to(DEVICE)

print("Model architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
print("-" * 20)

# 5. Training Loop
# ----------------
criterion = nn.CrossEntropyLoss() # Suitable for classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train() # Set model to training mode
    total_loss = 0
    start_time = time.time()

    for i, (batch_data, batch_labels) in enumerate(dataloader):
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Input shape: [batch_size, seq_len, n_features]
        outputs = model(batch_data)
        # Output shape: [batch_size, num_classes]
        # Labels shape: [batch_size]

        # Calculate loss
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % (len(dataloader) // 5) == 0: # Print progress ~5 times per epoch
             elapsed = time.time() - start_time
             print(f'  Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s')
             start_time = time.time() # Reset timer for next segment

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # Forward pass
            outputs = model(batch_data) # Shape: [batch_size, num_classes]

            # Calculate loss
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted_classes = torch.max(outputs, 1) # Get the index of the max log-probability
            correct_predictions += (predicted_classes == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

print("Starting training...")
start_training_time = time.time()

for epoch in range(1, N_EPOCHS + 1):
    epoch_start_time = time.time()

    # Train
    avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    # Evaluate
    avg_val_loss, val_accuracy = evaluate(model, val_loader, criterion, DEVICE)

    epoch_duration = time.time() - epoch_start_time
    print("-" * 50)
    print(f'Epoch {epoch}/{N_EPOCHS} Summary:')
    print(f'  Train Loss: {avg_train_loss:.4f}')
    print(f'  Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')
    print(f'  Epoch Duration: {epoch_duration:.2f}s')
    print("-" * 50)


total_training_time = time.time() - start_training_time
print(f"Training finished in {total_training_time:.2f} seconds.")
print("-" * 20)

# 6. Inference Example
# --------------------
print("Running inference example...")

# Take one sample from the validation set (or generate a new one)
sample_data, sample_label = val_dataset[0] # Get the first sample (tensor, label)
# sample_data shape: [seq_len, n_features]
# sample_label: scalar (0 or 1)

# Prepare the sample for the model (add batch dimension and move to device)
input_tensor = sample_data.unsqueeze(0).to(DEVICE) # Shape: [1, seq_len, n_features]

# Run inference
model.eval() # Set model to evaluation mode
with torch.no_grad():
    output = model(input_tensor) # Shape: [1, num_classes]

# Interpret the output
probabilities = torch.softmax(output, dim=1) # Convert logits to probabilities
predicted_prob_failure = probabilities[0, 1].item() # Probability of class 1 (Failure)
predicted_class = torch.argmax(probabilities, dim=1).item() # Predicted class (0 or 1)

print(f"Input sample shape: {sample_data.shape}")
print(f"True Label: {'Failure' if sample_label == 1 else 'No Failure'} (Class {sample_label})")
print(f"Model Output (Logits): {output.cpu().numpy()}")
print(f"Model Output (Probabilities): {probabilities.cpu().numpy()}")
print(f"Predicted Class: {'Failure' if predicted_class == 1 else 'No Failure'} (Class {predicted_class})")
print(f"Predicted Probability of Failure (within {PREDICTION_HORIZON} steps): {predicted_prob_failure:.4f}")

print("-" * 20)
print("Script finished.")
