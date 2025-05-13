import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# --- Hyperparameters ---
SEQ_LENGTH = 20  # Number of time steps in a process sequence
NUM_SENSORS = 5  # Number of sensors
D_MODEL = 64  # Dimension of the model (embedding size)
N_HEADS = 4  # Number of attention heads
NUM_ENCODER_LAYERS = 2  # Number of Transformer encoder layers
DIM_FEEDFORWARD = 128  # Dimension of the feedforward network
DROPOUT = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 20  # Keep low for quick demo, increase for better performance
NUM_SEQUENCES_TRAIN = 1000
NUM_SEQUENCES_TEST = 200


# --- 1. Data Simulation ---
# Simulate manufacturing process data
# Each sequence represents a product going through a process
# Each time step has readings from NUM_SENSORS
# A fault is introduced based on certain patterns

def generate_data(num_sequences, seq_length, num_sensors):
    """
    Generates synthetic sensor data and fault labels.
    A fault is more likely if:
    - Sensor 0 reading is high (> 0.7) for more than 3 consecutive steps AND
    - Sensor 2 reading is low (< -0.7) concurrently in at least one of those steps.
    OR
    - Sensor 1 has a sharp spike (> 1.5) and Sensor 3 drops significantly (< -1.5) simultaneously.
    """
    sequences = np.random.randn(num_sequences, seq_length, num_sensors).astype(np.float32)
    labels = np.zeros(num_sequences, dtype=np.int64)

    for i in range(num_sequences):
        fault_condition1_met = False
        high_sensor0_streak = 0

        # Condition 1 check
        for t in range(seq_length):
            if sequences[i, t, 0] > 0.8:  # Sensor 0 high
                high_sensor0_streak += 1
                if high_sensor0_streak > 2 and sequences[i, t, 2] < -0.8:  # Sensor 2 low
                    fault_condition1_met = True
                    break
            else:
                high_sensor0_streak = 0

        if fault_condition1_met:
            labels[i] = 1
            # Amplify the pattern slightly to make it more learnable
            for t_fault in range(max(0, t - 2), t + 1):  # Iterate over the identified fault window
                sequences[i, t_fault, 0] = np.clip(sequences[i, t_fault, 0] * 1.5, -3.0, 3.0)
                sequences[i, t_fault, 2] = np.clip(sequences[i, t_fault, 2] * 1.5, -3.0, 3.0)
            continue  # Move to next sequence if fault assigned

        # Condition 2 check (rarer, more direct correlation)
        for t in range(seq_length):
            if sequences[i, t, 1] > 1.8 and sequences[i, t, 3] < -1.8:
                labels[i] = 1
                # Amplify
                sequences[i, t, 1] = np.clip(sequences[i, t, 1] * 1.2, -3.0, 3.0)
                sequences[i, t, 3] = np.clip(sequences[i, t, 3] * 1.2, -3.0, 3.0)
                break

    return sequences, labels


# --- 2. PyTorch Dataset ---
class SensorDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# --- 3. Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# --- 4. Transformer Model ---
class TransformerRCA(nn.Module):
    def __init__(self, num_sensors, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerRCA, self).__init__()
        self.d_model = d_model

        # Input embedding layer (projects sensor readings to d_model)
        self.input_embedding = nn.Linear(num_sensors, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=SEQ_LENGTH + 1)  # +1 for CLS token

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # (1, 1, D_MODEL)

        # Output layer for classification
        self.output_layer = nn.Linear(d_model, 2)  # 2 classes: No Fault, Fault

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, return_attention=False):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, num_sensors]
            return_attention: bool, if True, returns attention weights from the last layer
        """
        batch_size = src.size(0)

        # Embed input
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        src_with_cls = torch.cat((cls_tokens, src_embedded), dim=1)  # (batch_size, seq_len+1, d_model)

        # Add positional encoding - needs (seq_len, batch_size, d_model)
        src_with_cls_permuted = src_with_cls.permute(1, 0, 2)  # (seq_len+1, batch_size, d_model)
        src_pos_encoded = self.pos_encoder(src_with_cls_permuted)
        src_pos_encoded = src_pos_encoded.permute(1, 0, 2)  # (batch_size, seq_len+1, d_model)

        # Transformer Encoder
        # To get attention weights, we need to access the layers.
        # This is a simplified way; for full access, one might need to modify nn.TransformerEncoder
        # or use hooks. For this example, we'll focus on the conceptual output.
        # The standard nn.TransformerEncoder does not directly return attention weights from all layers.
        # We can get the attention from a single nn.TransformerEncoderLayer if we use it directly.
        # For simplicity, we'll rely on the conceptual understanding that attention is computed.
        # For actual weight extraction, hooks are generally preferred on nn.MultiheadAttention modules.

        # For a more direct way to get attention, let's use a single encoder layer and extract its attention
        # This is a workaround for simplicity. A production model might use hooks.
        if return_attention and hasattr(self.transformer_encoder.layers[0].self_attn, 'forward'):
            # Temporarily replace forward to capture attention
            original_mha_forward = self.transformer_encoder.layers[-1].self_attn.forward

            def new_mha_forward(query, key, value, **kwargs):
                # The MHA forward typically returns (attn_output, attn_output_weights)
                # kwargs often include key_padding_mask and need_weights
                attn_output, attn_weights = original_mha_forward(query, key, value, **kwargs)
                self.last_attention_weights = attn_weights  # Store it
                return attn_output, attn_weights

            # Apply the hook-like mechanism for the last layer's self-attention
            last_layer_self_attn = self.transformer_encoder.layers[-1].self_attn

            # Store original forward and then replace
            # This is a bit hacky; proper hooks are cleaner for complex models
            _old_forward = last_layer_self_attn.forward

            def custom_forward_wrapper(*args, **kwargs):
                # Ensure need_weights=True is passed if not already
                # The MHA in TransformerEncoderLayer is called with need_weights=True by default if it's training or if specified.
                # We are interested in the weights from the CLS token to other tokens.
                kwargs['need_weights'] = True
                attn_output, attn_output_weights = _old_forward(*args, **kwargs)
                self.last_attention_weights = attn_output_weights
                return attn_output  # MHA in encoder layer only returns attn_output

            last_layer_self_attn.forward = custom_forward_wrapper

            encoder_output = self.transformer_encoder(src_pos_encoded)

            # Restore original forward
            last_layer_self_attn.forward = _old_forward

        else:
            encoder_output = self.transformer_encoder(src_pos_encoded)  # (batch_size, seq_len+1, d_model)
            self.last_attention_weights = None  # Ensure it's None if not captured

        # Get the CLS token's output representation
        cls_output = encoder_output[:, 0, :]  # (batch_size, d_model)

        # Pass through output layer
        output = self.output_layer(cls_output)  # (batch_size, 2)

        if return_attention:
            return output, self.last_attention_weights
        return output


# --- 5. Training Loop ---
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for i, (sequences, labels) in enumerate(dataloader):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    print("Training finished.")


# --- 6. Inference and Root Cause Analysis (RCA) ---
def analyze_sequence(model, sequence_tensor, device, seq_length, num_sensors, threshold=0.1):
    """
    Performs inference on a single sequence and analyzes attention weights for RCA.
    Args:
        model: The trained TransformerRCA model.
        sequence_tensor: A single sequence tensor of shape [1, seq_len, num_sensors].
        device: The device to run inference on.
        seq_length: Original sequence length (without CLS token).
        num_sensors: Number of sensors.
        threshold: Attention score threshold to consider a sensor/timestep influential.
    """
    model.eval()
    with torch.no_grad():
        output, attention_weights = model(sequence_tensor.to(device), return_attention=True)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print(f"\n--- Inference and RCA for a Sample Sequence ---")
    print(f"Input Sequence (first 3 steps, all sensors): \n{sequence_tensor[0, :3, :]}")
    print(f"Predicted Class: {'FAULT' if predicted_class == 1 else 'NO FAULT'}")
    print(f"Prediction Probabilities: No Fault={probabilities[0, 0]:.4f}, Fault={probabilities[0, 1]:.4f}")

    if predicted_class == 1 and attention_weights is not None:
        print("\nRoot Cause Analysis (Attention Weights from CLS token):")
        # attention_weights shape: (batch_size, num_heads, query_len, key_len)
        # For CLS token, query_len is seq_len+1, key_len is seq_len+1.
        # We are interested in attention from CLS (query_idx=0) to sensor tokens (key_idx=1 to seq_len+1)

        # Average attention weights across heads
        # attention_weights for MHA in TransformerEncoderLayer is (batch_size, L, S) where L is target seq len, S is source seq len
        # Here, for self-attention, L=S=seq_len+1.
        # We want attention from CLS token (index 0) to all other tokens.
        cls_attention_to_sequence = attention_weights[0, 0,
                                    1:]  # Attn from CLS to sequence tokens (batch_idx 0, CLS_idx 0, keys 1 onwards)
        # Shape: (seq_len) if MHA returns (N, L, S)
        # If MHA returns (N, num_heads, L, S) then it's attention_weights[0, :, 0, 1:].mean(dim=0)

        # The nn.MultiheadAttention in nn.TransformerEncoderLayer returns weights of shape (N, L, S)
        # where N is batch size, L is target sequence length, S is source sequence length.
        # Here, N=1, L=S=SEQ_LENGTH+1.
        # We want the attention from the CLS token (query at index 0) to all other tokens in the sequence.

        # The actual attention_weights from `self_attn` in `TransformerEncoderLayer`
        # has shape (batch_size, target_seq_len, source_seq_len).
        # Here, target_seq_len = source_seq_len = SEQ_LENGTH + 1 (due to CLS token).
        # We are interested in the attention scores where the CLS token is the query.
        # So, we look at `attention_weights[0, 0, :]`, which are scores from CLS to all tokens (including itself).
        # We skip the attention to itself (index 0) and take `attention_weights[0, 0, 1:]`.

        cls_to_tokens_attention = attention_weights[0, 0, 1:]  # Shape: (seq_length)

        print(
            f"Attention from CLS token to sequence steps (averaged over heads if applicable, here it's direct from MHA):")

        influential_steps = []
        for t in range(seq_length):  # Iterate through actual sequence steps (excluding CLS)
            score = cls_to_tokens_attention[t].item()
            if score > threshold:  # A simple threshold for influence
                influential_steps.append({'time_step': t, 'attention_score': score})

        if influential_steps:
            print(f"Potentially influential time steps (attention > {threshold}):")
            # Sort by attention score descending
            influential_steps_sorted = sorted(influential_steps, key=lambda x: x['attention_score'], reverse=True)
            for step_info in influential_steps_sorted[:5]:  # Print top 5
                print(f"  Time Step: {step_info['time_step']}, Attention Score: {step_info['attention_score']:.4f}")
                # Further, one could analyze which sensor at this timestep, but MHA combines features before attention.
                # For sensor-specific attention, a different architecture or attention mechanism might be needed,
                # or one could look at the magnitude of sensor values at these highly attended time steps.
                print(f"    Sensor values at this step: {sequence_tensor[0, step_info['time_step'], :].cpu().numpy()}")
        else:
            print("No specific time step stood out significantly based on the current attention threshold.")
            print(
                f"Highest attention score observed: {cls_to_tokens_attention.max().item():.4f} at time step {cls_to_tokens_attention.argmax().item()}")

    elif predicted_class == 1 and attention_weights is None:
        print(
            "\nRoot Cause Analysis: Attention weights were not captured for this model configuration or inference path.")


# --- 7. Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate Data
    print("Generating training data...")
    train_sequences, train_labels = generate_data(NUM_SEQUENCES_TRAIN, SEQ_LENGTH, NUM_SENSORS)
    print(f"Generated {len(train_labels)} training sequences. Fault rate: {np.mean(train_labels):.2%}")

    print("Generating test data...")
    test_sequences, test_labels = generate_data(NUM_SEQUENCES_TEST, SEQ_LENGTH, NUM_SENSORS)
    print(f"Generated {len(test_labels)} test sequences. Fault rate: {np.mean(test_labels):.2%}")

    # Create DataLoaders
    train_dataset = SensorDataset(train_sequences, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = SensorDataset(test_sequences, test_labels)  # Not used in training, but good for eval
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model, Loss, Optimizer
    model = TransformerRCA(
        num_sensors=NUM_SENSORS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the Model
    train_model(model, train_dataloader, criterion, optimizer, NUM_EPOCHS, device)

    # --- Perform Inference and RCA on a few samples from the test set ---
    print("\n--- Running Inference and RCA on Test Samples ---")

    # Find one faulty and one non-faulty example if possible
    faulty_idx = np.where(test_labels == 1)[0]
    non_faulty_idx = np.where(test_labels == 0)[0]

    sample_sequences_for_rca = []
    if len(faulty_idx) > 0:
        sample_sequences_for_rca.append(torch.tensor(test_sequences[faulty_idx[0]], dtype=torch.float32).unsqueeze(0))
        print(f"Selected a known FAULTY sequence (index {faulty_idx[0]}) for detailed analysis.")
    else:
        print("No faulty sequences found in the test set for detailed analysis. Using a random one.")
        sample_sequences_for_rca.append(torch.tensor(test_sequences[0], dtype=torch.float32).unsqueeze(0))

    if len(non_faulty_idx) > 0:
        sample_sequences_for_rca.append(
            torch.tensor(test_sequences[non_faulty_idx[0]], dtype=torch.float32).unsqueeze(0))
        print(f"Selected a known NON-FAULTY sequence (index {non_faulty_idx[0]}) for detailed analysis.")
    else:
        print(
            "No non-faulty sequences found in the test set for detailed analysis. Using another random one if available.")
        if len(test_sequences) > 1:
            sample_sequences_for_rca.append(torch.tensor(test_sequences[1], dtype=torch.float32).unsqueeze(0))

    for i, sample_seq_tensor in enumerate(sample_sequences_for_rca):
        print(f"\n--- Analyzing Test Sample {i + 1} ---")
        # The analyze_sequence function needs a model that can return attention.
        # The current way of getting attention weights in TransformerRCA is a bit of a hack.
        # For robust attention retrieval, using hooks on the nn.MultiheadAttention modules is better.
        # Let's ensure the `return_attention=True` path in `TransformerRCA.forward` is correctly set up.
        # The `analyze_sequence` function expects `model.last_attention_weights` to be populated.
        # The MHA module within TransformerEncoderLayer itself computes attention.
        # We need to ensure that the `need_weights=True` is passed to MHA and weights are captured.
        # The current custom_forward_wrapper in TransformerRCA tries to do this.
        analyze_sequence(model, sample_seq_tensor, device, SEQ_LENGTH, NUM_SENSORS,
                         threshold=0.05)  # Lowered threshold for demo

    print("\nScript finished.")

    # Example of how to use hooks for attention (more robust way):
    # This is for demonstration of the hook concept, not integrated into the main flow above for brevity.
    # captured_attention = {}
    # def get_attention_hook(module, input, output):
    #     # output of MHA is (attn_output, attn_output_weights)
    #     # We are interested in attn_output_weights
    #     # output[1] shape: (batch_size, num_heads, query_len, key_len) or (batch_size, query_len, key_len)
    #     captured_attention[module_name] = output[1].detach().cpu()

    # # In model __init__ or before forward pass:
    # # module_name = "encoder.layer0.self_attn"
    # # model.transformer_encoder.layers[0].self_attn.register_forward_hook(get_attention_hook)
    # # After model(input_data), captured_attention[module_name] will have the weights.
    # # Remember to remove hooks if they are no longer needed.
