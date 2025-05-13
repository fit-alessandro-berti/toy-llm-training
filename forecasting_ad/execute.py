# -*- coding: utf-8 -*-
"""
Transformer Model for Multivariate Time Series Forecasting
Scenario: Manufacturing Process Monitoring & Forecasting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import time
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuration ---
# Data parameters
N_SAMPLES = 2000  # Number of time steps in the dataset
N_FEATURES_NUM = 2  # Number of numerical features (e.g., temperature, pressure)
N_FEATURES_CAT = 1  # Number of categorical features (e.g., machine state)
N_CATEGORIES = 4    # Number of unique machine states (e.g., Idle, Running, Maintenance, Error)
SEQ_LEN = 60        # Input sequence length (lookback window)
PRED_LEN = 10       # Prediction horizon (how many steps to forecast)

# Model parameters
D_MODEL = 64        # Dimension of the model (embedding size)
N_HEAD = 4          # Number of attention heads
N_LAYERS = 3        # Number of Transformer encoder layers
DIM_FEEDFORWARD = 128 # Dimension of the feedforward network
DROPOUT = 0.1       # Dropout rate

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10 # Keep low for a quick demo, increase for better results
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Input sequence length: {SEQ_LEN}, Prediction length: {PRED_LEN}")

# --- 1. Synthetic Data Generation ---
def generate_manufacturing_data(n_samples, n_features_num, n_categories):
    """Generates synthetic multivariate time series data."""
    print("Generating synthetic manufacturing data...")
    time_steps = np.arange(n_samples)
    data = np.zeros((n_samples, n_features_num + 1)) # +1 for categorical

    # Numerical features (e.g., sensors) - combine trends and seasonality
    for i in range(n_features_num):
        trend = 0.01 * time_steps
        seasonality = 5 * np.sin(2 * np.pi * time_steps / (100 + i*20))
        noise = np.random.normal(0, 1, n_samples)
        data[:, i] = trend + seasonality + noise + 20 # Base value

    # Categorical feature (e.g., machine state) - simulate cycles
    cycle_len = 150
    states = []
    for t in time_steps:
        if (t % cycle_len) < 80: # Running
            state = 1
        elif (t % cycle_len) < 110: # Idle
            state = 0
        elif (t % cycle_len) < 130: # Maintenance
            state = 2
        else: # Error (less frequent)
             state = 3
        # Add some noise/randomness
        if np.random.rand() < 0.05:
             state = np.random.randint(0, n_categories)
        states.append(state)
    data[:, n_features_num] = np.array(states)

    print(f"Generated data shape: {data.shape}")
    return data

# --- 2. Data Preprocessing ---
def create_sequences(data, seq_len, pred_len):
    """Creates input sequences and corresponding target sequences."""
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:(i + seq_len)]
        y = data[(i + seq_len):(i + seq_len + pred_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(raw_data, n_features_num, n_categories, seq_len, pred_len):
    """Applies scaling, encoding, and sequencing."""
    print("Preprocessing data...")

    # Define preprocessing steps for numerical and categorical features
    # Note: OneHotEncoder creates n_categories binary columns
    numerical_features = list(range(n_features_num))
    categorical_features = [n_features_num]

    # Scaler for numerical features
    num_pipeline = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    # Encoder for categorical features
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(categories=[range(n_categories)], handle_unknown='ignore', sparse_output=False)) # Use sparse_output=False for dense array
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any (none in this case)
    )

    # Fit and transform the data
    # Important: Fit only on training data in a real scenario
    # Here, we fit on all data for simplicity of demonstration
    data_processed = preprocessor.fit_transform(raw_data)
    print(f"Data shape after preprocessing (scaling/encoding): {data_processed.shape}")

    # Get feature names after transformation (useful for understanding output)
    feature_names_out = preprocessor.get_feature_names_out()
    print(f"Feature names after transformation: {feature_names_out}")

    # Calculate the total number of features after one-hot encoding
    n_features_processed = data_processed.shape[1]

    # Create sequences
    X, y = create_sequences(data_processed, seq_len, pred_len)
    print(f"Created sequences: X shape {X.shape}, y shape {y.shape}") # X: (samples, seq_len, features), y: (samples, pred_len, features)

    # Split data (simple split for demo)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, preprocessor, n_features_processed, feature_names_out

# --- 3. Transformer Model Definition ---
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
        self.register_buffer('pe', pe) # Makes 'pe' part of the model state, but not a parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerForecaster(nn.Module):
    """Transformer Encoder based model for time series forecasting."""
    def __init__(self, input_dim, d_model, n_head, n_layers, dim_feedforward, dropout, output_dim):
        super(TransformerForecaster, self).__init__()
        self.d_model = d_model

        # Input embedding layer (linear projection)
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        # Output layer to project back to the number of features * prediction length
        # We want to predict all features for PRED_LEN steps ahead.
        # Option 1: Predict sequence directly (requires reshaping later)
        # self.output_layer = nn.Linear(d_model * SEQ_LEN, output_dim * PRED_LEN) # Predict all at once

        # Option 2: Predict step-by-step or use final hidden state
        # Using the output of the last time step of the encoder
        self.output_layer = nn.Linear(d_model, output_dim) # Predicts features for one step

        # Option 3: Use mean/max pooling over sequence dimension before output layer
        # self.output_layer = nn.Linear(d_model, output_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        """Generates a mask to prevent attention to future positions."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input sequence, shape [batch_size, seq_len, input_dim]
            src_mask: Mask for the source sequence (optional)
        """
        # 1. Embed input
        # src shape: [batch_size, seq_len, input_dim]
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model)
        # src_embedded shape: [batch_size, seq_len, d_model]

        # 2. Add positional encoding
        # Need shape [seq_len, batch_size, d_model] for pos_encoder
        src_embedded = src_embedded.permute(1, 0, 2) # [seq_len, batch_size, d_model]
        src_pos = self.pos_encoder(src_embedded)
        # src_pos shape: [seq_len, batch_size, d_model]

        # 3. Pass through Transformer Encoder
        # TransformerEncoder expects [seq_len, batch_size, d_model] if batch_first=False (default)
        # Or [batch_size, seq_len, d_model] if batch_first=True
        # Our encoder layer has batch_first=True
        src_pos = src_pos.permute(1, 0, 2) # Back to [batch_size, seq_len, d_model]

        # Generate mask if not provided (standard for forecasting)
        if src_mask is None:
            # Mask prevents attending to future positions within the input sequence
            # Not strictly necessary if just using encoder output for prediction,
            # but good practice if model might be adapted for autoregressive generation.
            # For simple forecasting (fixed window -> fixed forecast), mask might not be needed.
            # Let's include it for completeness.
            # device = src.device
            # src_mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
            pass # Let's try without mask first for simplicity in this setup


        # Transformer Encoder expects mask shape [seq_len, seq_len]
        encoder_output = self.transformer_encoder(src_pos, mask=src_mask)
        # encoder_output shape: [batch_size, seq_len, d_model]

        # 4. Decode to output features
        # Use the output corresponding to the *last* input time step to predict the future
        # last_step_output = encoder_output[:, -1, :] # Shape: [batch_size, d_model]
        # predictions = self.output_layer(last_step_output) # Shape: [batch_size, output_dim]
        # This predicts only *one* step ahead based on the last input state.

        # To predict PRED_LEN steps, we can either:
        # a) Apply the output layer to *all* output steps and take the last PRED_LEN ones (if model learned this mapping)
        # b) Apply the output layer to the last step and use this as the first prediction, then feed back autoregressively (more complex)
        # c) Modify the output layer to directly predict PRED_LEN steps from the final state or the full sequence.

        # Let's try applying the linear layer to *all* time steps of the encoder output
        # and assume the model learns to use the sequence context appropriately.
        # We'll then select the *last* PRED_LEN outputs as our forecast.
        # This is a simplification but common in some forecasting setups.
        predictions = self.output_layer(encoder_output) # Shape: [batch_size, seq_len, output_dim]

        # Return the predictions corresponding to the forecast horizon
        # We assume the model learns to place the relevant forecast info in the final outputs.
        # This is a modeling choice. A common alternative is to predict only the next step
        # or use a decoder. For simplicity, we predict the whole output sequence length
        # and expect the training forces the model to learn the mapping.
        # We need the predictions for the *next* PRED_LEN steps, which aren't directly
        # output here. Let's adjust the target handling or model structure.

        # --- Revised approach: Predict PRED_LEN steps from the last hidden state ---
        # Use the output corresponding to the *last* input time step
        last_step_output = encoder_output[:, -1, :] # Shape: [batch_size, d_model]

        # Modify output layer to predict PRED_LEN * output_dim features
        # Reshape needed after prediction
        # self.output_layer = nn.Linear(d_model, output_dim * PRED_LEN) # Define this in __init__

        # If output_layer predicts only output_dim (one step):
        # We need an autoregressive loop here for multi-step forecast, which adds complexity.

        # --- Simplest approach for demo: Predict full target sequence directly ---
        # Let's modify the output layer to predict the desired shape directly
        # This requires changing the __init__ and potentially the loss calculation
        # Let's stick to the idea of predicting `output_dim` features at each output step
        # and train the model to predict the target sequence `y` shifted relative to `x`.
        # The loss will compare `predictions` (shape B, S, F) with `y` (shape B, P, F).
        # This requires `y` to be aligned with the *end* of the `predictions`.

        # Let's return the full sequence output from the encoder passed through the final layer.
        # The loss function will handle comparing the relevant parts.
        return predictions # Shape: [batch_size, seq_len, output_dim]


# --- 4. Training Loop ---
def train_model(model, dataloader, criterion, optimizer, epoch, device, pred_len):
    model.train()  # Set model to training mode
    total_loss = 0.
    start_time = time.time()

    for batch, (X_batch, y_batch) in enumerate(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # X_batch shape: [batch_size, seq_len, n_features_processed]
        # y_batch shape: [batch_size, pred_len, n_features_processed]

        optimizer.zero_grad()
        output = model(X_batch)
        # output shape: [batch_size, seq_len, n_features_processed]

        # We need to compare the *last* `pred_len` outputs of the model
        # with the target `y_batch`.
        loss = criterion(output[:, -pred_len:, :], y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

        if batch % 50 == 0 and batch > 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / (batch + 1)
            cur_loss = total_loss / (batch + 1)
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(dataloader):5d} batches | '
                  f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.4f}')

    epoch_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | training loss {epoch_loss:5.4f}')
    print('-' * 89)
    return epoch_loss


# --- Evaluation Function (Optional but Recommended) ---
def evaluate_model(model, dataloader, criterion, device, pred_len):
    model.eval() # Set model to evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output[:, -pred_len:, :], y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 5. Inference/Forecasting Function ---
def forecast(model, input_sequence, preprocessor, device, n_features_processed, pred_len, feature_names_out):
    """
    Makes a forecast using the trained model.

    Args:
        model: The trained PyTorch model.
        input_sequence: The last `seq_len` data points (raw format, before preprocessing).
                        Shape: (seq_len, n_features_raw)
        preprocessor: The fitted scikit-learn preprocessor.
        device: The torch device ('cpu' or 'cuda').
        n_features_processed: Total number of features after preprocessing.
        pred_len: The forecast horizon.
        feature_names_out: List of feature names after preprocessing.

    Returns:
        forecast_raw: The forecast in the original data scale. Shape: (pred_len, n_features_raw)
    """
    model.eval() # Set model to evaluation mode

    print("\n--- Starting Forecast ---")
    print(f"Input sequence shape (raw): {input_sequence.shape}")

    # 1. Preprocess the input sequence
    input_processed = preprocessor.transform(input_sequence)
    print(f"Input sequence shape (processed): {input_processed.shape}")

    # 2. Convert to tensor and add batch dimension
    input_tensor = torch.tensor(input_processed, dtype=torch.float32).unsqueeze(0).to(device) # Shape: [1, seq_len, n_features_processed]

    # 3. Get model prediction
    with torch.no_grad():
        prediction = model(input_tensor) # Shape: [1, seq_len, n_features_processed]

    # 4. Extract the forecast part (last pred_len steps of the output)
    forecast_processed = prediction[0, -pred_len:, :].cpu().numpy() # Shape: [pred_len, n_features_processed]
    print(f"Output forecast shape (processed): {forecast_processed.shape}")

    # 5. Inverse transform the prediction
    # Ensure the shape matches what the preprocessor expects for inverse_transform
    # It usually expects the same number of columns it was fitted on.
    # If forecast_processed has fewer columns (e.g., only predicted numerical), adjust accordingly.
    # Here, forecast_processed should have n_features_processed columns.

    # Need to handle numerical and categorical separately for inverse transform
    num_feature_indices = [i for i, name in enumerate(feature_names_out) if name.startswith('num__')]
    cat_feature_indices = [i for i, name in enumerate(feature_names_out) if name.startswith('cat__')]
    n_num_features_processed = len(num_feature_indices)
    n_cat_features_processed = len(cat_feature_indices) # This is n_categories due to one-hot

    # Create a dummy array with the correct shape for inverse transform
    dummy_array_for_inverse = np.zeros((pred_len, n_features_processed))
    dummy_array_for_inverse[:, :] = forecast_processed # Fill with predicted values

    # Inverse transform using the fitted preprocessor
    forecast_raw = preprocessor.inverse_transform(dummy_array_for_inverse)
    print(f"Output forecast shape (raw): {forecast_raw.shape}")

    # Post-process categorical: convert one-hot back to labels
    # The categorical part of forecast_raw will have the original single column
    # The inverse_transform of OneHotEncoder might yield floats, find the argmax
    # Note: inverse_transform handles this directly if pipelines are set up correctly.
    # Let's assume forecast_raw[:, N_FEATURES_NUM] contains the predicted category index (possibly float).
    # We might need to round or apply argmax if the inverse transform wasn't perfect for categorical.
    # For simplicity, let's assume inverse_transform gives reasonable values.
    # Round the categorical prediction
    if N_FEATURES_CAT > 0:
         forecast_raw[:, N_FEATURES_NUM] = np.round(forecast_raw[:, N_FEATURES_NUM]).astype(int)
         # Clamp values to be within valid category range
         forecast_raw[:, N_FEATURES_NUM] = np.clip(forecast_raw[:, N_FEATURES_NUM], 0, N_CATEGORIES - 1)


    print("--- Forecast Complete ---")
    return forecast_raw


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Generate Data
    raw_data = generate_manufacturing_data(N_SAMPLES, N_FEATURES_NUM, N_CATEGORIES)
    n_features_raw = raw_data.shape[1]

    # 2. Preprocess Data
    X_train, y_train, X_test, y_test, preprocessor, n_features_processed, feature_names_out = preprocess_data(
        raw_data, N_FEATURES_NUM, N_CATEGORIES, SEQ_LEN, PRED_LEN
    )

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model, Loss, Optimizer
    model = TransformerForecaster(
        input_dim=n_features_processed,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=n_features_processed # Predict all processed features
    ).to(DEVICE)

    # Use MSELoss for simplicity, even with one-hot encoded categories.
    # A combined loss (MSE for numerical, CrossEntropy for categorical) would be better
    # but adds complexity to the output layer and loss calculation.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Learning rate scheduler

    print("\n--- Starting Training ---")
    # 4. Training Loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_model(model, train_dataloader, criterion, optimizer, epoch, DEVICE, PRED_LEN)
        val_loss = evaluate_model(model, test_dataloader, criterion, DEVICE, PRED_LEN)
        print(f'| end of epoch {epoch:3d} | validation loss {val_loss:5.4f}')
        print('-' * 89)
        scheduler.step() # Adjust learning rate

    print("--- Training Complete ---")

    # 5. Perform Inference on a sample from the test set
    # Take the last sequence from the raw data as input for forecasting
    # Ensure we have enough data points before the end
    if len(raw_data) >= SEQ_LEN:
        input_sequence_raw = raw_data[-SEQ_LEN:] # Last SEQ_LEN points

        # Make the forecast
        forecast_result_raw = forecast(
            model,
            input_sequence_raw,
            preprocessor,
            DEVICE,
            n_features_processed,
            PRED_LEN,
            feature_names_out
        )

        print("\n--- Example Forecast (Raw Scale) ---")
        print(f"Forecasting the next {PRED_LEN} steps:")
        # Print header
        header = "Step | " + " | ".join([f"Feat_{i}" for i in range(N_FEATURES_NUM)]) + " | MachineState"
        print(header)
        print("-" * len(header))
        # Print forecast
        for i in range(PRED_LEN):
            num_vals = " | ".join([f"{forecast_result_raw[i, j]:.2f}" for j in range(N_FEATURES_NUM)])
            cat_val = int(forecast_result_raw[i, N_FEATURES_NUM])
            print(f"{i+1:4d} | {num_vals} | {cat_val:^12d}")

        # Optional: Compare with actual values if available
        if len(raw_data) >= SEQ_LEN + PRED_LEN:
             actual_values = raw_data[-(SEQ_LEN + PRED_LEN):-PRED_LEN] # This indexing seems wrong
             actual_values = raw_data[-PRED_LEN:] # The actual values following the input sequence
             print("\n--- Actual Values for Comparison ---")
             print(header)
             print("-" * len(header))
             for i in range(PRED_LEN):
                 num_vals = " | ".join([f"{actual_values[i, j]:.2f}" for j in range(N_FEATURES_NUM)])
                 cat_val = int(actual_values[i, N_FEATURES_NUM])
                 print(f"{i+1:4d} | {num_vals} | {cat_val:^12d}")

    else:
        print("\nNot enough data points in raw_data to perform forecast with the specified SEQ_LEN.")
