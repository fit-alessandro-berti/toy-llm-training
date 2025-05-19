import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Hyperparameters
SEQ_LENGTH = 200
GAP_START_IDX = 80
GAP_END_IDX = 120
N_EPOCHS = 300 # Increased for better convergence
LR = 0.001
ENCODER_INPUT_FEATURES = 2  # Series A (masked), Series B
DECODER_INPUT_FEATURES = 1  # Series B value during the gap
HIDDEN_SIZE = 64
NUM_LSTM_LAYERS = 1 # Using 1 layer for simplicity in this example
BATCH_SIZE = 1 # For this problem, we process one pair of series at a time

# Derived constants
GAP_LENGTH = GAP_END_IDX - GAP_START_IDX

def generate_data(seq_length, gap_start, gap_end):
    """
    Generates two related time series, A and B.
    Series A will have a missing block.
    """
    t = np.linspace(0, 20, seq_length)

    # Series B: A sine wave with some noise and decay
    series_b = np.sin(t * 2.5) * np.exp(-t * 0.03) + np.random.normal(0, 0.15, seq_length)

    # Series A: Related to Series B, with another component and noise
    series_a_true = 0.6 * series_b + 0.4 * np.cos(t * 1.8) + np.random.normal(0, 0.1, seq_length)

    # Create masked Series A
    series_a_masked = series_a_true.copy()
    series_a_masked[gap_start:gap_end] = 0.0  # Mask with zeros

    # Convert to PyTorch tensors
    # Shapes: (seq_length, num_features) for easier handling later, will unsqueeze for batch
    series_a_true_tensor = torch.FloatTensor(series_a_true).view(seq_length, 1)
    series_a_masked_tensor = torch.FloatTensor(series_a_masked).view(seq_length, 1)
    series_b_tensor = torch.FloatTensor(series_b).view(seq_length, 1)
    
    return series_a_true_tensor, series_a_masked_tensor, series_b_tensor

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False) # Expects (seq_len, batch, feature)

    def forward(self, x):
        # x shape: (seq_len, batch_size, input_size)
        _, (hidden, cell) = self.lstm(x)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False) # Expects (seq_len, batch, feature)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        # x shape: (1, batch_size, input_size) for a single time step
        # hidden_state shape: (num_layers, batch_size, hidden_size)
        # cell_state shape: (num_layers, batch_size, hidden_size)
        output, (hidden, cell) = self.lstm(x, (hidden_state, cell_state))
        # output shape: (1, batch_size, hidden_size)
        prediction = self.fc(output.squeeze(0)) # Squeeze seq_len dim, then fc
        # prediction shape: (batch_size, output_size)
        return prediction, hidden, cell

class Seq2SeqImputer(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqImputer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, series_a_masked, series_b_full, series_b_gap, gap_length):
        # series_a_masked shape: (seq_len, batch_size, 1)
        # series_b_full shape: (seq_len, batch_size, 1)
        # series_b_gap shape: (gap_len, batch_size, 1)

        # Encoder input: concatenate masked A and full B
        encoder_input = torch.cat((series_a_masked, series_b_full), dim=2)
        # encoder_input shape: (seq_len, batch_size, ENCODER_INPUT_FEATURES)
        
        encoder_hidden, encoder_cell = self.encoder(encoder_input)
        # encoder_hidden/cell shape: (num_layers, batch_size, hidden_size)

        # Decoder
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        predictions = []

        for i in range(gap_length):
            # Decoder input for this step is the corresponding value from series_b_gap
            decoder_input_step = series_b_gap[i, :, :].unsqueeze(0) # (1, batch_size, DECODER_INPUT_FEATURES)
            
            # Pass previous hidden state and current input
            prediction_step, decoder_hidden, decoder_cell = self.decoder(decoder_input_step, decoder_hidden, decoder_cell)
            # prediction_step shape: (batch_size, 1)
            predictions.append(prediction_step)
            
        # Concatenate predictions for the gap
        imputed_gap = torch.stack(predictions).squeeze(-1) # (gap_length, batch_size)
        if imputed_gap.shape[1] == 1: # If batch_size is 1, make it (gap_length, 1)
            imputed_gap = imputed_gap.squeeze(1).unsqueeze(-1)

        return imputed_gap


def train_model(model, series_a_true, series_a_masked, series_b,
                gap_start_idx, gap_end_idx, n_epochs, lr, device):
    print("Starting training...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare data for model input (add batch dimension)
    # Expected shape by LSTM (if batch_first=False): (seq_len, batch_size, num_features)
    series_a_masked_batch = series_a_masked.unsqueeze(1).to(device) # (seq_len, 1, 1)
    series_b_full_batch = series_b.unsqueeze(1).to(device)       # (seq_len, 1, 1)

    # Extract the part of series_b that corresponds to the gap
    series_b_gap_batch = series_b[gap_start_idx:gap_end_idx, :].unsqueeze(1).to(device) # (gap_len, 1, 1)
    
    # Target for the loss function
    target_gap = series_a_true[gap_start_idx:gap_end_idx, :].to(device) # (gap_len, 1)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        imputed_gap = model(series_a_masked_batch, series_b_full_batch, series_b_gap_batch, GAP_LENGTH)
        # imputed_gap shape: (gap_length, 1)
        # target_gap shape: (gap_length, 1)
        
        loss = criterion(imputed_gap, target_gap)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}')
    print("Training finished.")

def test_model(model, series_a_true, series_a_masked, series_b,
               gap_start_idx, gap_end_idx, device):
    print("\nStarting testing...")
    model.eval()

    series_a_masked_batch = series_a_masked.unsqueeze(1).to(device)
    series_b_full_batch = series_b.unsqueeze(1).to(device)
    series_b_gap_batch = series_b[gap_start_idx:gap_end_idx, :].unsqueeze(1).to(device)
    
    target_gap_values = series_a_true[gap_start_idx:gap_end_idx, :].cpu().numpy().flatten()

    with torch.no_grad():
        imputed_gap_tensor = model(series_a_masked_batch, series_b_full_batch, series_b_gap_batch, GAP_LENGTH)
    
    imputed_gap_values = imputed_gap_tensor.cpu().numpy().flatten()

    mse = np.mean((target_gap_values - imputed_gap_values)**2)
    mae = np.mean(np.abs(target_gap_values - imputed_gap_values))

    print(f"Test MSE on the imputed gap: {mse:.6f}")
    print(f"Test MAE on the imputed gap: {mae:.6f}")

    print("\n--- Imputation Results (Original vs. Predicted) ---")
    print("Idx | Original | Predicted | Difference")
    print("----|----------|-----------|------------")
    for i in range(len(target_gap_values)):
        diff = target_gap_values[i] - imputed_gap_values[i]
        print(f"{gap_start_idx + i:3d} | {target_gap_values[i]:8.4f} | {imputed_gap_values[i]:9.4f} | {diff:10.4f}")
    print("--------------------------------------------------\n")
    
    # For verification: print some values of series_a_masked and series_b
    # print("\n--- Sample of series_a_masked (first 5, gap start, last 5) ---")
    # print(series_a_masked[:5].flatten().tolist())
    # print(series_a_masked[gap_start_idx-2:gap_start_idx+3].flatten().tolist()) # Around gap start
    # print(series_a_masked[-5:].flatten().tolist())

    # print("\n--- Sample of series_b (first 5, gap start, last 5) ---")
    # print(series_b[:5].flatten().tolist())
    # print(series_b[gap_start_idx-2:gap_start_idx+3].flatten().tolist())
    # print(series_b[-5:].flatten().tolist())


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Generate Data
    series_a_true, series_a_masked, series_b = generate_data(SEQ_LENGTH, GAP_START_IDX, GAP_END_IDX)
    
    # Print info about the generated data
    print(f"Generated data: SEQ_LENGTH={SEQ_LENGTH}, GAP_START_IDX={GAP_START_IDX}, GAP_END_IDX={GAP_END_IDX}, GAP_LENGTH={GAP_LENGTH}")
    print(f"Shape of series_a_true: {series_a_true.shape}")
    print(f"Shape of series_a_masked: {series_a_masked.shape}")
    print(f"Shape of series_b: {series_b.shape}")
    # Verify masking
    # print(f"Series A true at gap start: {series_a_true[GAP_START_IDX:GAP_START_IDX+5].flatten().tolist()}")
    # print(f"Series A masked at gap start: {series_a_masked[GAP_START_IDX:GAP_START_IDX+5].flatten().tolist()}")


    # 2. Initialize Model
    encoder = Encoder(input_size=ENCODER_INPUT_FEATURES, 
                      hidden_size=HIDDEN_SIZE, 
                      num_layers=NUM_LSTM_LAYERS).to(device)
    
    decoder = Decoder(input_size=DECODER_INPUT_FEATURES, 
                      hidden_size=HIDDEN_SIZE, 
                      output_size=1,  # Predicting a single value for series A
                      num_layers=NUM_LSTM_LAYERS).to(device)
    
    imputation_model = Seq2SeqImputer(encoder, decoder, device).to(device)

    # 3. Training Phase
    train_model(imputation_model, series_a_true, series_a_masked, series_b,
                GAP_START_IDX, GAP_END_IDX, N_EPOCHS, LR, device)

    # 4. Test Phase
    # For this script, we test on the same data used for training to demonstrate imputation.
    # In a real scenario, you'd have a separate test set.
    test_model(imputation_model, series_a_true, series_a_masked, series_b,
               GAP_START_IDX, GAP_END_IDX, device)

    print("Script finished.")