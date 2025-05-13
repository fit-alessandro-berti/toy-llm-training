# -*- coding: utf-8 -*-
"""
Transformer for Root Cause Analysis in Manufacturing Simulation

This script demonstrates training and inference of a Transformer model
to classify manufacturing process sequences as faulty or normal,
and uses attention weights for root cause analysis interpretability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --- Configuration ---
SEQ_LEN = 10         # Length of each process sequence
N_SENSORS = 3        # Number of sensor readings per step
N_PROCESS_STEPS = 5  # Number of distinct process steps (e.g., 'Start', 'OpA', 'Transfer', 'OpB', 'End')
EMBED_DIM = 32       # Embedding dimension for the transformer
N_HEADS = 4          # Number of attention heads
N_LAYERS = 2         # Number of transformer encoder layers
HIDDEN_DIM = 64      # Hidden dimension in feed-forward layers
DROPOUT = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 16
N_EPOCHS = 15
N_SAMPLES = 1000     # Total number of sequences to generate
FAULT_PROBABILITY = 0.2 # Probability of a sequence having a fault

# --- Data Simulation ---

def generate_manufacturing_data(n_samples, seq_len, n_sensors, n_process_steps, fault_prob):
    """
    Generates simulated manufacturing process data.

    Args:
        n_samples (int): Number of sequences to generate.
        seq_len (int): Length of each sequence.
        n_sensors (int): Number of sensor readings per step.
        n_process_steps (int): Number of distinct process steps.
        fault_prob (float): Probability of injecting a fault into a sequence.

    Returns:
        tuple: (sequences, labels, root_cause_indices)
               - sequences: List of sequences, each sequence is a list of tuples
                            [(step_id, [sensor_vals]), ...]
               - labels: List of labels (0 for normal, 1 for faulty).
               - root_cause_indices: List where each element is the index (or -1 if normal)
                                     where the root cause anomaly was injected.
    """
    process_step_names = [f"Step_{i}" for i in range(n_process_steps)]
    sequences = []
    labels = []
    root_cause_indices = [] # Store the index where the fault originates

    print(f"Generating {n_samples} sequences...")

    for i in range(n_samples):
        current_sequence = []
        is_faulty = random.random() < fault_prob
        fault_injected = False
        fault_index = -1

        # Decide where the fault might occur if this sequence is faulty
        potential_fault_step_idx = random.randint(1, seq_len - 2) # Avoid start/end
        potential_fault_sensor = random.randint(0, n_sensors - 1)
        potential_fault_process_step = random.randint(0, n_process_steps - 1)

        # Define normal sensor ranges (mean, std_dev)
        normal_sensor_means = np.random.rand(n_sensors) * 50 + 25 # e.g., values between 25 and 75
        normal_sensor_stddevs = np.random.rand(n_sensors) * 5 + 1   # e.g., std dev between 1 and 6

        for step_idx in range(seq_len):
            # Determine process step (can be random or follow a pattern)
            process_step_id = random.randint(0, n_process_steps - 1)

            # Generate normal sensor readings
            sensor_values = [np.random.normal(normal_sensor_means[s], normal_sensor_stddevs[s])
                             for s in range(n_sensors)]

            # Inject fault if applicable
            if is_faulty and not fault_injected and step_idx == potential_fault_step_idx and process_step_id == potential_fault_process_step:
                # Introduce an anomaly (e.g., spike or drop) in a specific sensor
                anomaly_magnitude = (random.random() - 0.5) * 10 * normal_sensor_stddevs[potential_fault_sensor] # Significant deviation
                sensor_values[potential_fault_sensor] += anomaly_magnitude
                fault_injected = True
                fault_index = step_idx
                # print(f"  Injecting fault at step {step_idx}, process {process_step_id}, sensor {potential_fault_sensor}") # Debug

            current_sequence.append((process_step_id, sensor_values))

        # Ensure a fault was actually injected if intended
        if is_faulty and not fault_injected:
             is_faulty = False # Didn't meet conditions, mark as normal
             fault_index = -1

        sequences.append(current_sequence)
        labels.append(1 if is_faulty else 0)
        root_cause_indices.append(fault_index)

        if (i + 1) % (n_samples // 10) == 0:
            print(f"  Generated {i+1}/{n_samples} sequences...")

    print("Data generation complete.")
    print(f"Faulty sequences: {sum(labels)}/{n_samples}")
    return sequences, labels, root_cause_indices

# --- Preprocessing ---

class ManufacturingDataset(Dataset):
    """PyTorch Dataset for manufacturing sequences."""
    def __init__(self, sequences, labels, seq_len, n_sensors, process_step_encoder, sensor_scaler):
        self.seq_len = seq_len
        self.n_sensors = n_sensors
        self.process_step_encoder = process_step_encoder
        self.sensor_scaler = sensor_scaler

        self.processed_sequences = []
        self.labels = torch.tensor(labels, dtype=torch.long)

        print("Preprocessing data...")
        num_skipped = 0
        for seq in sequences:
            if len(seq) != self.seq_len:
                # print(f"Warning: Skipping sequence with length {len(seq)} (expected {self.seq_len}).")
                num_skipped += 1
                continue # Skip sequences not matching the expected length exactly

            step_ids = [item[0] for item in seq]
            sensor_data = np.array([item[1] for item in seq]) # Shape: (seq_len, n_sensors)

            # Encode step IDs (already numerical in simulation, but could be strings)
            # In a real scenario, fit the encoder first: process_step_encoder.fit([item[0] for seq in all_sequences for item in seq])
            encoded_steps = torch.tensor(step_ids, dtype=torch.long)

            # Scale sensor data
            # In a real scenario, fit the scaler first: sensor_scaler.fit(all_sensor_data)
            scaled_sensors = torch.tensor(self.sensor_scaler.transform(sensor_data), dtype=torch.float32)

            # Combine step embedding index and sensor data for each time step
            # We'll handle the embedding lookup in the model's forward pass
            combined_features = torch.cat((encoded_steps.unsqueeze(1), scaled_sensors), dim=1) # Shape: (seq_len, 1 + n_sensors)
            self.processed_sequences.append(combined_features)

        if num_skipped > 0:
             print(f"Warning: Skipped {num_skipped} sequences due to length mismatch.")
        if not self.processed_sequences:
            raise ValueError("No valid sequences found after preprocessing. Check sequence lengths.")

        # Adjust labels if sequences were skipped
        valid_indices = [i for i, seq in enumerate(sequences) if len(seq) == self.seq_len]
        self.labels = self.labels[valid_indices]

        print(f"Preprocessing complete. Number of valid sequences: {len(self.processed_sequences)}")


    def __len__(self):
        return len(self.processed_sequences)

    def __getitem__(self, idx):
        return self.processed_sequences[idx], self.labels[idx]

# --- Transformer Model ---

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer so it's not a model parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerRCA(nn.Module):
    """Transformer model for Root Cause Analysis classification."""
    def __init__(self, n_process_steps, n_sensors, embed_dim, n_heads, n_layers, hidden_dim, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_sensors = n_sensors

        # Embedding layer for process steps
        self.step_embedding = nn.Embedding(n_process_steps, embed_dim // 2) # Allocate part of embed_dim

        # Linear layer to project sensor data to match embedding dimension part
        self.sensor_projection = nn.Linear(n_sensors, embed_dim - (embed_dim // 2))

        # Combine step embedding and sensor projection dimension must match embed_dim
        assert self.step_embedding.embedding_dim + self.sensor_projection.out_features == embed_dim, \
            "Combined dimension mismatch"

        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=seq_len + 1) # +1 for CLS token

        # Standard Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Linear(embed_dim, 2) # Output: 2 classes (Normal, Faulty)

        # Special token (like CLS in BERT) to aggregate sequence information
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # Learnable CLS token

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.step_embedding.weight.data.uniform_(-initrange, initrange)
        self.sensor_projection.weight.data.uniform_(-initrange, initrange)
        self.sensor_projection.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()

    def forward(self, src):
        """
        Forward pass of the model.

        Args:
            src (Tensor): Input tensor of shape [batch_size, seq_len, 1 + n_sensors].
                          The first column contains process step IDs, the rest are sensor values.

        Returns:
            tuple: (output_logits, attention_weights)
                   - output_logits: Tensor of shape [batch_size, 2] (scores for Normal, Faulty)
                   - attention_weights: List of attention tensors from each layer/head (or None if not captured)
        """
        batch_size, seq_len, _ = src.shape

        # Separate step IDs and sensor data
        step_ids = src[:, :, 0].long()          # Shape: [batch_size, seq_len]
        sensor_data = src[:, :, 1:]             # Shape: [batch_size, seq_len, n_sensors]

        # Get step embeddings and project sensor data
        step_embed = self.step_embedding(step_ids) # Shape: [batch_size, seq_len, embed_dim // 2]
        sensor_proj = self.sensor_projection(sensor_data) # Shape: [batch_size, seq_len, embed_dim - embed_dim // 2]

        # Concatenate embeddings and projections
        combined_embed = torch.cat((step_embed, sensor_proj), dim=2) # Shape: [batch_size, seq_len, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape: [batch_size, 1, embed_dim]
        src_with_cls = torch.cat((cls_tokens, combined_embed), dim=1) # Shape: [batch_size, seq_len + 1, embed_dim]

        # Add positional encoding (needs shape [seq_len+1, batch_size, embed_dim])
        src_with_cls_pos = src_with_cls.permute(1, 0, 2) # Shape: [seq_len+1, batch_size, embed_dim]
        src_with_cls_pos = self.pos_encoder(src_with_cls_pos)
        src_with_cls_pos = src_with_cls_pos.permute(1, 0, 2) # Back to [batch_size, seq_len+1, embed_dim]


        # --- Pass through Transformer Encoder ---
        # To get attention weights, we need to access the internals, which is tricky with nn.TransformerEncoder
        # Option 1: Modify nn.TransformerEncoder or implement manually (complex)
        # Option 2: Use a library hook (possible, but adds complexity)
        # Option 3 (Simplest for demo): We won't explicitly return attention weights from forward *during training*.
        # We will run a separate forward pass *during inference* on a single item,
        # modifying the model temporarily or manually stepping through layers to capture attention.
        # For this script, we'll focus on the inference part for attention analysis.

        # Note: nn.TransformerEncoderLayer expects input shape [seq_len, batch_size, embed_dim] if batch_first=False (default)
        # or [batch_size, seq_len, embed_dim] if batch_first=True. We used batch_first=True.
        transformer_output = self.transformer_encoder(src_with_cls_pos) # Shape: [batch_size, seq_len + 1, embed_dim]

        # Use the output corresponding to the CLS token for classification
        cls_output = transformer_output[:, 0, :] # Shape: [batch_size, embed_dim]

        # Final classification layer
        logits = self.classifier(cls_output) # Shape: [batch_size, 2]

        # During inference/analysis, we'll need a way to get attention. We'll handle that separately.
        return logits, None # Return None for attention during standard forward pass

# --- Training Function ---

def train_model(model, dataloader, criterion, optimizer, device, n_epochs):
    """Trains the Transformer model."""
    model.train() # Set model to training mode
    print("\n--- Starting Training ---")
    for epoch in range(n_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output_logits, _ = model(data) # Ignore attention weights during training for simplicity
            loss = criterion(output_logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output_logits.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            if batch_idx % 50 == 0:
                 print(f"  Epoch {epoch+1}/{n_epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        accuracy = 100. * correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{n_epochs} Completed | Avg Loss: {avg_epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")
    print("--- Training Finished ---")

# --- Inference and Root Cause Analysis ---

def analyze_sequence_attention(model, sequence_data, device, process_step_encoder, sensor_scaler, seq_len, n_sensors, n_process_steps):
    """
    Performs inference on a single sequence and analyzes attention weights.
    This requires manually accessing the attention mechanism of the encoder layers.
    """
    model.eval() # Set model to evaluation mode

    # --- Preprocess the single sequence ---
    if len(sequence_data) != seq_len:
        print(f"Error: Sequence length mismatch ({len(sequence_data)} vs {seq_len})")
        return

    step_ids = [item[0] for item in sequence_data]
    sensor_values = np.array([item[1] for item in sequence_data])

    encoded_steps = torch.tensor(step_ids, dtype=torch.long).unsqueeze(0) # Add batch dim
    scaled_sensors = torch.tensor(sensor_scaler.transform(sensor_values), dtype=torch.float32).unsqueeze(0) # Add batch dim

    # Create input tensor [batch_size=1, seq_len, 1 + n_sensors]
    input_tensor = torch.cat((encoded_steps.unsqueeze(2), scaled_sensors), dim=2).to(device)

    print("\n--- Analyzing Sequence ---")
    print("Input Sequence (Process Step ID, [Scaled Sensors]):")
    for i in range(seq_len):
        step_name = f"Step_{step_ids[i]}" # Simple mapping back
        sensors_str = ", ".join([f"{s:.2f}" for s in scaled_sensors[0, i].tolist()])
        print(f"  Time {i}: {step_name} [{sensors_str}]")

    # --- Manual Forward Pass to Capture Attention ---
    with torch.no_grad():
        # 1. Prepare input embeddings + CLS token + Positional Encoding
        batch_size = 1
        step_embed = model.step_embedding(input_tensor[:, :, 0].long())
        sensor_proj = model.sensor_projection(input_tensor[:, :, 1:])
        combined_embed = torch.cat((step_embed, sensor_proj), dim=2)
        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        src_with_cls = torch.cat((cls_tokens, combined_embed), dim=1)
        src_with_cls_pos = src_with_cls.permute(1, 0, 2)
        src_with_cls_pos = model.pos_encoder(src_with_cls_pos)
        src_with_cls_pos = src_with_cls_pos.permute(1, 0, 2) # Back to [batch_size, seq_len+1, embed_dim]

        # 2. Pass through encoder layers and capture attention
        attention_outputs = [] # Store attention weights from each layer
        current_input = src_with_cls_pos

        # Hook to capture attention weights from MultiheadAttention
        attention_weights_list = []
        def hook_fn(module, input, output):
             # output[1] contains the attention weights (batch_size, num_heads, seq_len, seq_len)
             # For TransformerEncoderLayer, it's (N, S, S) or (S, S) if batch_first=False
             # Check the actual output format based on PyTorch version and settings
             # For batch_first=True, MultiheadAttention output[1] is (N, L, S) -> (batch, target_seq_len, source_seq_len)
             # We are interested in self-attention, so L=S = seq_len + 1
             # print(f"Hook captured output type: {type(output)}, len: {len(output) if isinstance(output, tuple) else 'N/A'}")
             if isinstance(output, tuple) and len(output) > 1:
                 # print(f"Attention weights shape: {output[1].shape}") # Debug shape
                 attention_weights_list.append(output[1].detach().cpu()) # Store weights from this layer/head

        hooks = []
        for layer in model.transformer_encoder.layers:
            # Register hook on the self-attention module within the layer
            hook = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(hook)
            current_input = layer(current_input) # Pass input through the layer
            # Note: The hook executes *after* the forward pass of self_attn is complete

        transformer_output = current_input # Output from the last layer

        # Remove hooks after use
        for hook in hooks:
            hook.remove()

        # 3. Classification
        cls_output = transformer_output[:, 0, :] # CLS token output
        logits = model.classifier(cls_output)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

        print(f"\nPrediction: {'Faulty' if prediction == 1 else 'Normal'}")
        print(f"Probabilities: Normal={probabilities[0, 0]:.4f}, Faulty={probabilities[0, 1]:.4f}")

        # --- Analyze Attention Weights ---
        if not attention_weights_list:
            print("\nCould not capture attention weights (check hook implementation or model structure).")
            return

        # Focus on the attention paid *by* the CLS token *to* the input sequence tokens
        # Average attention weights across layers and heads for simplicity
        # Shape of each element in attention_weights_list: [batch_size=1, seq_len+1, seq_len+1] (after averaging heads if needed)
        # We need to handle the potential num_heads dimension if not averaged by the hook capture
        # Assuming hook captures [batch_size, num_heads, target_seq_len, source_seq_len]
        # Or potentially [batch_size, target_seq_len, source_seq_len] if averaged internally

        num_layers_captured = len(attention_weights_list)
        print(f"\nAnalyzing Attention Weights (Captured from {num_layers_captured} layers)...")

        if num_layers_captured > 0:
            # Let's average across all layers and heads
            # Assuming shape [batch_size, num_heads, target_len, source_len]
            avg_attention = torch.zeros_like(attention_weights_list[0][0, 0, :, :]) # Shape [target_len, source_len]
            total_heads = 0

            for layer_attention in attention_weights_list:
                # layer_attention shape: [batch_size, num_heads, target_len, source_len]
                # or [batch_size, target_len, source_len] if heads are averaged by MHA setting
                if layer_attention.dim() == 4: # Has heads dimension
                    num_heads_in_layer = layer_attention.shape[1]
                    # Average over heads for this layer
                    layer_avg_heads = layer_attention.mean(dim=1).squeeze(0) # Shape [target_len, source_len]
                elif layer_attention.dim() == 3: # Already averaged over heads or single head
                     layer_avg_heads = layer_attention.squeeze(0) # Shape [target_len, source_len]
                else:
                    print(f"Unexpected attention weight dimension: {layer_attention.dim()}")
                    continue

                avg_attention += layer_avg_heads
                total_heads += num_heads_in_layer if layer_attention.dim() == 4 else 1 # Count layers if averaged

            if total_heads > 0:
                 avg_attention /= num_layers_captured # Average across layers

            # We want the attention paid BY the CLS token (index 0) TO all other tokens (including itself)
            cls_attention_to_tokens = avg_attention[0, :] # Shape [seq_len + 1]

            # Normalize attention scores to sum to 1 (optional, for easier interpretation)
            cls_attention_to_tokens = torch.softmax(cls_attention_to_tokens, dim=0)

            print("Average Attention Scores from CLS Token to Input Tokens:")
            print("  Token Index | Type        | Attention Score")
            print("---------------------------------------------")
            print(f"  CLS Token   | CLS         | {cls_attention_to_tokens[0]:.4f}") # Attention to itself
            for i in range(seq_len):
                step_name = f"Step_{step_ids[i]}"
                print(f"  Time {i:<4}    | {step_name:<10} | {cls_attention_to_tokens[i+1]:.4f}") # i+1 because index 0 is CLS

            # Identify top K attended tokens as potential root causes
            k = 3 # Show top 3 attended steps
            top_k_scores, top_k_indices = torch.topk(cls_attention_to_tokens[1:], k) # Exclude CLS token

            print(f"\nTop {k} Attended Input Steps (Potential Root Causes):")
            for i in range(k):
                idx = top_k_indices[i].item()
                score = top_k_scores[i].item()
                step_name = f"Step_{step_ids[idx]}"
                sensors_str = ", ".join([f"{s:.2f}" for s in scaled_sensors[0, idx].tolist()])
                print(f"  1. Time {idx}: {step_name} [{sensors_str}] (Attention: {score:.4f})")

        else:
            print("No attention weights were captured.")


# --- Main Execution ---
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Generate Data
    sequences, labels, root_causes = generate_manufacturing_data(
        N_SAMPLES, SEQ_LEN, N_SENSORS, N_PROCESS_STEPS, FAULT_PROBABILITY
    )

    # 2. Preprocess Data
    # Fit scalers/encoders based on the generated data
    all_sensor_data = np.vstack([item[1] for seq in sequences for item in seq])
    sensor_scaler = StandardScaler().fit(all_sensor_data)

    # Process step encoder (not strictly needed here as IDs are 0-based ints)
    process_step_encoder = LabelEncoder().fit(list(range(N_PROCESS_STEPS)))

    # Split data
    train_seq, test_seq, train_labels, test_labels, train_rc, test_rc = train_test_split(
        sequences, labels, root_causes, test_size=0.2, random_state=42, stratify=labels
    )

    # Create Datasets and DataLoaders
    train_dataset = ManufacturingDataset(train_seq, train_labels, SEQ_LEN, N_SENSORS, process_step_encoder, sensor_scaler)
    test_dataset = ManufacturingDataset(test_seq, test_labels, SEQ_LEN, N_SENSORS, process_step_encoder, sensor_scaler)

    # Handle potential empty datasets after filtering
    if not train_dataset or not test_dataset:
        print("Error: Not enough valid data to create datasets. Exiting.")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Use batch_size=1 for test loader if analyzing individual sequences later
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # 3. Initialize Model, Loss, Optimizer
    model = TransformerRCA(
        n_process_steps=N_PROCESS_STEPS,
        n_sensors=N_SENSORS,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        seq_len=SEQ_LEN
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train the Model
    train_model(model, train_loader, criterion, optimizer, device, N_EPOCHS)

    # 5. Inference and Analysis on a Test Sample
    # Find a faulty sequence in the test set for analysis
    faulty_test_indices = [i for i, label in enumerate(test_labels) if label == 1 and len(test_seq[i]) == SEQ_LEN]

    if faulty_test_indices:
        sample_index_in_test_seq = faulty_test_indices[0] # Take the first faulty one
        sample_sequence = test_seq[sample_index_in_test_seq]
        true_label = test_labels[sample_index_in_test_seq]
        true_root_cause_idx = test_rc[sample_index_in_test_seq] # Index where fault was injected

        print(f"\n--- Analyzing a Faulty Test Sequence (Index {sample_index_in_test_seq} in test_seq) ---")
        print(f"True Label: {'Faulty' if true_label == 1 else 'Normal'}")
        print(f"Simulated Root Cause Step Index: {true_root_cause_idx}")

        analyze_sequence_attention(
            model,
            sample_sequence,
            device,
            process_step_encoder,
            sensor_scaler,
            SEQ_LEN,
            N_SENSORS,
            N_PROCESS_STEPS
        )
    else:
        print("\nNo faulty sequences found in the test set to analyze.")

    # Optional: Evaluate on the whole test set (without attention analysis)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("\n--- Evaluating on Test Set ---")
    with torch.no_grad():
        for data, target in test_loader: # Using test_loader with batch_size=1
             data, target = data.to(device), target.to(device)
             outputs, _ = model(data)
             loss = criterion(outputs, target)
             test_loss += loss.item()
             _, predicted = torch.max(outputs.data, 1)
             total += target.size(0)
             correct += (predicted == target).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}% ({correct}/{total})")

