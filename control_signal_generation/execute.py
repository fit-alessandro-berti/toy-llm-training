import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the controllable system (e.g., a simple linear system)
class SimpleSystem:
    def __init__(self, initial_state=0.0, noise_level=0.01):
        self.state = initial_state
        self.noise_level = noise_level

    def step(self, control_action):
        # System dynamics: next_state = current_state + control_action + noise
        self.state = self.state + control_action + np.random.normal(0, self.noise_level)
        return self.state

    def reset(self, initial_state=0.0):
        self.state = initial_state
        return self.state

# Data Generation
def generate_data(num_sequences, input_seq_len, output_seq_len, desired_state_range=(-5.0, 5.0)):
    """
    Generates data for training and testing.
    Each sequence consists of:
    - encoder_input: Current state of the system (a single value repeated input_seq_len times for simplicity,
                     or a sequence of past states if the system had more complex dynamics).
                     In this simple example, we will use a sequence of past states.
    - decoder_input: Typically a "start-of-sequence" token and then the previously generated control actions
                     during training (teacher forcing) or prediction. For simplicity in this generation,
                     we will pre-generate some 'dummy' decoder inputs that are not directly used by the system
                     simulation but help structure the Seq2Seq input. The actual control actions will be the target.
    - decoder_output (target): The sequence of control actions needed to reach a desired future state.
    - desired_future_state: The target state we want the system to reach after output_seq_len control actions.
    """
    encoder_inputs = []
    decoder_outputs = [] # These are the control actions (targets)
    desired_future_states_list = []
    initial_states_list = []

    system = SimpleSystem()

    for _ in range(num_sequences):
        initial_state = np.random.uniform(-2.0, 2.0) # Random initial state for more diverse data
        system.reset(initial_state)

        current_states_sequence = [system.state]
        for _ in range(input_seq_len -1): # Generate a short history of states
            # Apply random actions to get some initial dynamics if desired, or just observe
            # For simplicity, let's assume no control during this observation period for encoder input
            current_states_sequence.append(system.step(0.0)) # No control, just observe drift/noise

        encoder_input_sequence = list(current_states_sequence) # Use the collected states as input

        desired_future_state = np.random.uniform(desired_state_range[0], desired_state_range[1])

        # Simulate to find control actions to reach the desired state (can be tricky)
        # For this simple system, we can deterministically (or with some search) find actions.
        # Let's try a more direct approach: aim to reduce the error at each step.
        # This is a simplification. A real planner or optimal control solver might be needed for complex systems.
        control_actions_sequence = []
        temp_system = SimpleSystem() # Use a temporary system for planning
        temp_system.reset(encoder_input_sequence[-1]) # Start planning from the last observed state

        remaining_error = desired_future_state - temp_system.state
        for i in range(output_seq_len):
            # Simple proportional control strategy for data generation
            # This is a heuristic to generate plausible control actions.
            # The model will learn this mapping.
            if i < output_seq_len -1 :
                control_action = remaining_error / (output_seq_len - i)
            else:
                control_action = remaining_error # Apply full correction on the last step

            # Clip control action to a reasonable range if necessary (e.g. -1 to 1)
            control_action = np.clip(control_action, -1.0, 1.0)
            control_actions_sequence.append(control_action)
            temp_system.step(control_action)
            remaining_error = desired_future_state - temp_system.state


        encoder_inputs.append(encoder_input_sequence)
        decoder_outputs.append(control_actions_sequence)
        desired_future_states_list.append(desired_future_state)
        initial_states_list.append(initial_state)


    # Convert to tensors
    # Encoder input: sequence of states
    # Decoder output: sequence of control actions
    encoder_inputs_tensor = torch.FloatTensor(encoder_inputs).unsqueeze(-1) # (num_sequences, input_seq_len, 1)
    decoder_outputs_tensor = torch.FloatTensor(decoder_outputs).unsqueeze(-1) # (num_sequences, output_seq_len, 1)
    desired_future_states_tensor = torch.FloatTensor(desired_future_states_list).unsqueeze(-1) # (num_sequences, 1)
    initial_states_tensor = torch.FloatTensor(initial_states_list).unsqueeze(-1) # (num_sequences, 1)


    return encoder_inputs_tensor, decoder_outputs_tensor, desired_future_states_tensor, initial_states_tensor

# Seq2Seq Model Definition
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, future_state_target):
        # x shape: (batch_size, seq_len, input_dim)
        # future_state_target shape: (batch_size, 1) -> needs to be (batch_size, 1, 1) for LSTM or integrated differently
        # For simplicity, we'll concatenate the target state to each element of the input sequence.
        # This is a common way to condition the encoder on the target.
        future_state_target_expanded = future_state_target.unsqueeze(1).repeat(1, x.size(1), 1)
        combined_input = torch.cat((x, future_state_target_expanded), dim=2)

        # combined_input shape: (batch_size, seq_len, input_dim + 1)
        outputs, (hidden, cell) = self.lstm(combined_input)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim # This is the control action dimension (1 in this case)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True) # Input to LSTM is previous action
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size, 1, output_dim) - current input to the decoder (e.g., start token or previous output)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output.squeeze(1)) # (batch_size, output_dim)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_seq_len, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_seq_len = target_seq_len
        self.device = device

    def forward(self, source_seq, future_state_target, target_seq=None, teacher_forcing_ratio=0.5):
        # source_seq: (batch_size, input_seq_len, input_dim)
        # future_state_target: (batch_size, 1)
        # target_seq: (batch_size, target_seq_len, output_dim) - ground truth actions for teacher forcing

        batch_size = source_seq.shape[0]
        # The output of the decoder (control action) has 1 dimension
        decoder_output_dim = self.decoder.output_dim

        outputs = torch.zeros(batch_size, self.target_seq_len, decoder_output_dim).to(self.device)

        encoder_hidden, encoder_cell = self.encoder(source_seq, future_state_target)

        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        # Initial input to the decoder: a zero tensor or a learned "start token"
        # For simplicity, using a zero tensor representing "no action taken yet"
        decoder_input = torch.zeros(batch_size, 1, decoder_output_dim).to(self.device)

        for t in range(self.target_seq_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs[:, t] = decoder_output # Store the prediction

            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio if target_seq is not None else False

            if teacher_force and target_seq is not None:
                decoder_input = target_seq[:, t, :].unsqueeze(1) # Next input is current ground truth
            else:
                decoder_input = decoder_output.unsqueeze(1).detach() # Next input is current prediction (detach for stability)

        return outputs


# Training Function
def train_model(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0

    for i, (encoder_input_batch, decoder_output_batch, desired_state_batch, _) in enumerate(iterator):
        encoder_input_batch = encoder_input_batch.to(device)
        decoder_output_batch = decoder_output_batch.to(device) # Ground truth control actions
        desired_state_batch = desired_state_batch.to(device)

        optimizer.zero_grad()

        # Pass encoder_input, desired_state, and ground truth control actions (for teacher forcing)
        output = model(encoder_input_batch, desired_state_batch, decoder_output_batch, teacher_forcing_ratio=0.5)

        # output shape: (batch_size, target_seq_len, output_dim)
        # decoder_output_batch shape: (batch_size, target_seq_len, output_dim)
        output_dim_size = output.shape[-1]
        output_flat = output.view(-1, output_dim_size)
        target_flat = decoder_output_batch.view(-1, output_dim_size)


        loss = criterion(output_flat, target_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Evaluation Function
def evaluate_model(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (encoder_input_batch, decoder_output_batch, desired_state_batch, _) in enumerate(iterator):
            encoder_input_batch = encoder_input_batch.to(device)
            decoder_output_batch = decoder_output_batch.to(device)
            desired_state_batch = desired_state_batch.to(device)

            # Turn off teacher forcing for evaluation
            output = model(encoder_input_batch, desired_state_batch, None, teacher_forcing_ratio=0.0)

            output_dim_size = output.shape[-1]
            output_flat = output.view(-1, output_dim_size)
            target_flat = decoder_output_batch.view(-1, output_dim_size)

            loss = criterion(output_flat, target_flat)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Test Function (Generate control signals for new scenarios)
def test_generate_signals(model, system, initial_state, desired_future_state, input_seq_len, output_seq_len, device, noise_level_sim=0.01):
    model.eval()
    system.reset(initial_state) # Reset system to the initial state for this test run
    system.noise_level = noise_level_sim # Set simulation noise

    print(f"\nTesting Signal Generation:")
    print(f"  Initial System State: {system.state:.4f}")
    print(f"  Desired Future State: {desired_future_state:.4f} (after {output_seq_len} steps)")

    # Prepare encoder input (current state of the system)
    # In a real scenario, you'd observe the system for input_seq_len steps.
    # For this test, we'll simulate a short history starting from initial_state
    current_states_sequence = [system.state]
    temp_system_for_input = SimpleSystem(initial_state=system.state, noise_level=0.0) # No noise for input prep
    for _ in range(input_seq_len -1):
        current_states_sequence.append(temp_system_for_input.step(0.0)) # Observe evolution without control

    encoder_input_tensor = torch.FloatTensor([current_states_sequence]).unsqueeze(-1).to(device) # (1, input_seq_len, 1)
    desired_state_tensor = torch.FloatTensor([[desired_future_state]]).to(device) # (1, 1)

    generated_control_actions = []
    with torch.no_grad():
        # Get the sequence of control actions from the model
        # No target_seq is provided, so teacher_forcing_ratio is effectively 0
        predicted_actions_tensor = model(encoder_input_tensor, desired_state_tensor, None, 0.0)
        # predicted_actions_tensor shape: (1, output_seq_len, 1)

    predicted_actions = predicted_actions_tensor.squeeze().cpu().numpy()
    if output_seq_len == 1: # Handle single action case
        predicted_actions = [predicted_actions.item()]


    print(f"  Generated Control Actions: {[f'{a:.4f}' for a in predicted_actions]}")

    # Simulate the system with the generated control actions
    achieved_states = [system.state]
    print(f"  Simulating system with generated actions (simulation noise: {noise_level_sim}):")
    print(f"    Step 0 (Initial): State = {system.state:.4f}")
    for i, action in enumerate(predicted_actions):
        current_state = system.step(action)
        achieved_states.append(current_state)
        print(f"    Step {i+1}: Action = {action:.4f}, New State = {current_state:.4f}")

    final_state = system.state
    error = abs(final_state - desired_future_state)
    print(f"  Final System State after {output_seq_len} steps: {final_state:.4f}")
    print(f"  Error from Desired State: {error:.4f}")
    return predicted_actions, achieved_states, final_state

def main():
    # Hyperparameters
    INPUT_DIM = 1  # State is a single value
    ENCODER_INPUT_DIM_ADJUSTED = INPUT_DIM + 1 # State + desired future state
    OUTPUT_DIM = 1 # Control action is a single value
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    INPUT_SEQ_LEN = 5   # How many past states to look at
    OUTPUT_SEQ_LEN = 3  # How many future control actions to predict
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    CLIP = 1
    BATCH_SIZE = 32
    NUM_TRAIN_SEQUENCES = 2000
    NUM_VAL_SEQUENCES = 400
    TEACHER_FORCING_RATIO_TRAIN = 0.6 # Use a fixed ratio for simplicity

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate Data
    print("Generating training data...")
    train_encoder_inputs, train_decoder_outputs, train_desired_states, _ = generate_data(NUM_TRAIN_SEQUENCES, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    print("Generating validation data...")
    val_encoder_inputs, val_decoder_outputs, val_desired_states, _ = generate_data(NUM_VAL_SEQUENCES, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)

    train_data = torch.utils.data.TensorDataset(train_encoder_inputs, train_decoder_outputs, train_desired_states, train_decoder_outputs) # Last one is dummy for dataloader structure
    val_data = torch.utils.data.TensorDataset(val_encoder_inputs, val_decoder_outputs, val_desired_states, val_decoder_outputs)

    train_iterator = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_iterator = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)

    # Initialize Model, Optimizer, Criterion
    encoder = Encoder(ENCODER_INPUT_DIM_ADJUSTED, HIDDEN_DIM, NUM_LAYERS)
    decoder = Decoder(OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    model = Seq2Seq(encoder, decoder, OUTPUT_SEQ_LEN, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Mean Squared Error for regression task

    print(f"Model initialized. Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training Loop
    best_val_loss = float('inf')
    print("\nStarting Training...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_iterator, optimizer, criterion, CLIP, device)
        val_loss = evaluate_model(model, val_iterator, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_control_model.pt')

        print(f"Epoch {epoch+1:02}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Load the best model for testing
    model.load_state_dict(torch.load('best_control_model.pt'))
    print("\nLoaded best model for testing.")

    # Test Phase
    test_system = SimpleSystem(noise_level=0.02) # Use a system with some noise for testing

    # Test Case 1
    initial_state_test_1 = 1.0
    desired_future_state_test_1 = 4.0
    print(f"\n--- Test Case 1 ---")
    test_generate_signals(model, test_system, initial_state_test_1, desired_future_state_test_1, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, device, noise_level_sim=0.02)

    # Test Case 2
    initial_state_test_2 = -2.0
    desired_future_state_test_2 = 0.5
    print(f"\n--- Test Case 2 ---")
    test_generate_signals(model, test_system, initial_state_test_2, desired_future_state_test_2, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, device, noise_level_sim=0.02)

    # Test Case 3: Reach a state requiring negative control
    initial_state_test_3 = 3.0
    desired_future_state_test_3 = -1.0
    print(f"\n--- Test Case 3 ---")
    test_generate_signals(model, test_system, initial_state_test_3, desired_future_state_test_3, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, device, noise_level_sim=0.02)

    # Test Case 4: Maintain current state (desired = initial, with some noise)
    initial_state_test_4 = 0.0
    desired_future_state_test_4 = 0.1 # slight change to see control
    print(f"\n--- Test Case 4 ---")
    test_generate_signals(model, test_system, initial_state_test_4, desired_future_state_test_4, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, device, noise_level_sim=0.02)

    # Test Case 5: Larger change
    initial_state_test_5 = -4.0
    desired_future_state_test_5 = 4.0
    print(f"\n--- Test Case 5 ---")
    test_generate_signals(model, test_system, initial_state_test_5, desired_future_state_test_5, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, device, noise_level_sim=0.02)


if __name__ == '__main__':
    # For reproducibility
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    main()