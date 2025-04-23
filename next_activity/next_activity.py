import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import time
import random
import string # To generate character mappings

# --- Configuration ---
class Config:
    # Data: List of process traces (sequences of activity strings)
    # Replace this with your actual process traces
    process_traces_str = [
        ("register request", "check ticket", "decide", "reject request", "send notification"),
        ("register request", "check ticket", "decide", "pay compensation", "send notification"),
        ("register request", "examine casually", "decide", "pay compensation", "send notification"),
        ("register request", "examine thoroughly", "check ticket", "decide", "reject request"),
        ("register request", "check ticket", "examine casually", "decide", "pay compensation"),
        ("register request", "check ticket", "examine thoroughly", "decide", "reject request", "reinitiate request"),
        ("register request", "examine casually", "check ticket", "decide", "pay compensation", "archive request"),
        ("register request", "examine thoroughly", "decide", "reject request"),
        ("register request", "check ticket", "pay compensation", "send notification"), # Example shorter trace
        ("register request", "reject request"), # Example very short trace
    ]

    random.shuffle(process_traces_str)
    # Split data (adjust ratio as needed)
    split_idx = int(len(process_traces_str) * 0.8)
    train_traces = process_traces_str[:split_idx]
    test_prefixes = [trace[:random.randint(1, max(1, len(trace)-1))] # Create some example prefixes for testing
                     for trace in process_traces_str[split_idx:] if len(trace) > 1]
    if not test_prefixes and len(process_traces_str) > 0 and len(process_traces_str[0]) > 1: # Ensure at least one test prefix if possible
        test_prefixes = [process_traces_str[0][:1]]


    # Special Tokens
    PAD_TOKEN = "[PAD]"
    SOS_TOKEN = "[SOS]" # Start of Sequence
    EOS_TOKEN = "[EOS]" # End of Sequence
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

    # Model Architecture (Adjust as needed)
    vocab_size = 0 # Will be set after tokenization
    d_model = 64   # Smaller model for potentially smaller dataset
    num_layers = 2
    num_heads = 4
    d_ff = 128
    dropout = 0.1
    max_seq_len = 50 # Max length for combined "[SOS] Trace [EOS]" sequence

    # Training
    batch_size = 4
    learning_rate = 1e-3
    epochs = 100 # Adjust as needed for convergence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_token_id = 0 # Will be set after tokenization

    # Generation (Suffix Prediction)
    gen_max_len = 30 # Max length of the *predicted suffix*
    gen_temperature = 0.3 # 0.0 for greedy decoding (most likely), > 0 for sampling
    gen_top_k = 0 # 0 to disable top-k sampling

config = Config()

# --- 1. Data Preparation ---

# Activity to Character Mapping
unique_activities = sorted(list(set(act for trace in config.process_traces_str for act in trace)))
# Ensure we have enough unique characters (using printable ASCII)
available_chars = string.ascii_letters + string.digits + string.punctuation
if len(unique_activities) > len(available_chars):
    raise ValueError(f"Not enough unique characters ({len(available_chars)}) "
                     f"to map {len(unique_activities)} unique activities.")

activity_to_char = {act: available_chars[i] for i, act in enumerate(unique_activities)}
char_to_activity = {v: k for k, v in activity_to_char.items()}
activity_chars = list(activity_to_char.values())

print("--- Activity Mapping ---")
for act, char in activity_to_char.items():
    print(f"'{act}' -> '{char}'")
print("------------------------")


# Tokenizer for Process Traces
class ProcessTokenizer:
    def __init__(self, activity_chars, special_tokens):
        self.special_tokens = special_tokens
        self.activity_chars = activity_chars
        self.vocab = special_tokens + activity_chars
        self.vocab_size = len(self.vocab)

        self.token_to_int = {token: i for i, token in enumerate(self.vocab)}
        self.int_to_token = {i: token for i, token in enumerate(self.vocab)}

        self.pad_id = self.token_to_int[config.PAD_TOKEN]
        self.sos_id = self.token_to_int[config.SOS_TOKEN]
        self.eos_id = self.token_to_int[config.EOS_TOKEN]

    def encode(self, char_sequence):
        """Encodes a sequence of activity characters into token IDs."""
        return [self.token_to_int.get(char, -1) for char in char_sequence] # Handle unknowns? (shouldn't happen with controlled vocab)

    def decode_chars(self, token_ids):
        """Decodes token IDs back into a sequence of characters, skipping special tokens."""
        return "".join([self.int_to_token.get(token_id, '')
                       for token_id in token_ids
                       if token_id not in (self.pad_id, self.sos_id, self.eos_id)])

    def decode_activities(self, token_ids, char_to_activity_map):
        """Decodes token IDs back into a list of activity strings."""
        chars = self.decode_chars(token_ids)
        return [char_to_activity_map.get(char, "[UNK_CHAR]") for char in chars]


# Initialize tokenizer and update config
tokenizer = ProcessTokenizer(activity_chars, config.special_tokens)
config.vocab_size = tokenizer.vocab_size
config.pad_token_id = tokenizer.pad_id
print(f"Vocabulary Size: {config.vocab_size}")
print(f"PAD ID: {config.pad_token_id}")
# print(f"Vocabulary: {' '.join(tokenizer.vocab)}")

# Dataset for Process Traces
class ProcessTraceDataset(Dataset):
    def __init__(self, traces_str, tokenizer, activity_to_char_map, max_seq_len):
        self.traces_str = traces_str
        self.tokenizer = tokenizer
        self.activity_to_char_map = activity_to_char_map
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.traces_str)

    def __getitem__(self, idx):
        trace_activities = self.traces_str[idx]

        # Map activities to characters
        trace_chars = []
        for act in trace_activities:
            char = self.activity_to_char_map.get(act)
            if char is None:
                 print(f"Warning: Activity '{act}' not found in mapping. Skipping trace index {idx}.")
                 # Return dummy data or handle appropriately. Here, we'll return None
                 # which should be filtered out by the collate_fn or DataLoader.
                 # A better approach might be to filter such traces during initialization.
                 return None # This item will be skipped by the default collate_fn if batch size > 1
                             # or cause an error if batch_size=1. Need careful handling.
                             # For simplicity, assume all activities are mapped.
            trace_chars.append(char)

        # Encode character sequence to token IDs
        token_ids = self.tokenizer.encode(trace_chars)

        # Prepare combined sequence: [SOS] Trace_Tokens [EOS]
        combined_tokens = ([self.tokenizer.sos_id] +
                           token_ids +
                           [self.tokenizer.eos_id])

        # Truncate if exceeds max length
        if len(combined_tokens) > self.max_seq_len:
             # Ensure EOS is always present if truncated
             combined_tokens = combined_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_id]

        # Input 'x' is the sequence except the last token
        x = torch.tensor(combined_tokens[:-1], dtype=torch.long)
        # Target 'y' is the sequence shifted right (predicts the next token)
        y = torch.tensor(combined_tokens[1:], dtype=torch.long)

        return x, y

# Collate function to handle padding within batches
def create_collate_fn(pad_token_id):
    def collate_fn(batch):
        # Filter out None items (e.g., from mapping errors in __getitem__)
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None # Return None if batch becomes empty

        # Separate sequences x and y
        batch_x, batch_y = zip(*batch)

        # Pad sequences in the batch
        padded_x = pad_sequence(batch_x, batch_first=True, padding_value=pad_token_id)
        padded_y = pad_sequence(batch_y, batch_first=True, padding_value=pad_token_id)

        return padded_x, padded_y
    return collate_fn

# Create dataset and dataloader
train_dataset = ProcessTraceDataset(config.train_traces, tokenizer, activity_to_char, config.max_seq_len)
collate_fn = create_collate_fn(config.pad_token_id)
# Use drop_last=True if you want to avoid smaller batches at the end,
# especially if using batch normalization (not used here) or if the collate_fn
# might return None for the last batch if it only contained problematic data.
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn, drop_last=True)


# --- 2. Model Architecture (Largely Unchanged from Original Script) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # Ensure max_len >= config.max_seq_len
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)

class TinyTransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model,
                                            padding_idx=config.pad_token_id)
        # Adjust max_len for positional encoding based on config
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout,
                                              max_len=config.max_seq_len + 1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True, # Input format is (batch, seq, feature)
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers
        )
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_padding_mask=None):
        """
        Args:
            src: Input tensor, shape [batch_size, seq_len]
            src_padding_mask: Bool tensor, shape [batch_size, seq_len], True where padded
        """
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        seq_len = src.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, src.device)

        # Decoder-only: Use src_emb as both target and memory input
        output = self.transformer_decoder(
            tgt=src_emb,
            memory=src_emb, # Source for keys/values is the same sequence
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=src_padding_mask, # Padding in target sequence
            memory_key_padding_mask=src_padding_mask # Padding in memory sequence (same)
        )
        logits = self.output_layer(output)
        return logits

# --- 3. Training Loop (Adapted for Padding) ---

model = TinyTransformerLM(config).to(config.device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id) # Ignores padding in target
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

def train(model, dataloader, criterion, optimizer, config, epoch):
    model.train()
    total_loss = 0.
    start_time = time.time()
    processed_batches = 0

    for i, (batch_x, batch_y) in enumerate(dataloader):
        # Handle potential None batch from collate_fn
        if batch_x is None or batch_y is None:
            print(f"Skipping empty batch {i}")
            continue

        batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
        src_padding_mask = (batch_x == config.pad_token_id) # True where padded

        optimizer.zero_grad()
        logits = model(batch_x, src_padding_mask=src_padding_mask) # [batch, seq_len, vocab_size]
        # Reshape for CrossEntropyLoss: [batch*seq_len, vocab_size] and [batch*seq_len]
        loss = criterion(logits.view(-1, config.vocab_size), batch_y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
        optimizer.step()

        total_loss += loss.item()
        processed_batches += 1

        log_interval = max(1, len(dataloader) // 5) # Log ~5 times per epoch
        if processed_batches > 0 and i % log_interval == 0 :
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'| Epoch {epoch:3d} | {i:4d}/{len(dataloader):4d} batches | '
                  f'lr {lr:02.2e} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            total_loss = 0 # Reset loss for next interval
            start_time = time.time()

# --- Training Execution ---
print(f"\nStarting Process Trace Prediction training on {config.device}...")
if len(train_dataloader) == 0:
    print("Error: Training dataloader is empty. Check data processing steps and filtering.")
else:
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train(model, train_dataloader, criterion, optimizer, config, epoch)
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s |')
        print('-' * 89)
        # Add evaluation pass here if you create a validation set/dataloader

# --- 4. Inference (Predicting the Suffix) ---

def predict_suffix(model, tokenizer, prefix_activities, activity_to_char_map, char_to_activity_map,
                     max_len, temperature, top_k, config):
    model.eval()

    # Map prefix activities to characters
    prefix_chars = []
    for act in prefix_activities:
        char = activity_to_char_map.get(act)
        if char is None:
             print(f"Warning: Activity '{act}' in prefix not found in training data mapping. Prediction may be unreliable.")
             # Option: Map to a special UNK char if you trained with one, or skip, or raise error.
             # Here, we'll just note it and continue, it might map to an unused token ID implicitly.
             continue # Skip unknown activities for prediction input
        prefix_chars.append(char)

    if not prefix_chars and prefix_activities:
         print("Warning: Prefix activities were all unknown. Cannot generate suffix.")
         return []

    # Encode characters to token IDs
    prefix_tokens = tokenizer.encode(prefix_chars)

    # Prepare input: [SOS] Prefix_Tokens
    input_tokens = [tokenizer.sos_id] + prefix_tokens
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(config.device)

    generated_token_ids = []

    with torch.no_grad():
        for _ in range(max_len):
            current_seq_len = input_ids.size(1)

            # Create padding mask (no padding needed for single sequence inference)
            # Model expects the argument, pass an all-False mask
            padding_mask = torch.zeros(1, current_seq_len, dtype=torch.bool, device=config.device)

            # Handle potential sequence length exceeding model's max_seq_len during generation
            if current_seq_len > config.max_seq_len:
                 # Keep only the last `max_seq_len` tokens for the model input
                 input_ids = input_ids[:, -config.max_seq_len:]
                 padding_mask = padding_mask[:, -config.max_seq_len:] # Adjust mask accordingly

            logits = model(input_ids, src_padding_mask=padding_mask) # [1, current_seq_len, vocab_size]
            last_token_logits = logits[:, -1, :] # [1, vocab_size] - logits for the *next* token

            # Apply temperature scaling and Top-K sampling / Greedy
            if temperature > 0:
                scaled_logits = last_token_logits / temperature
                # Apply Top-K filtering
                if top_k > 0 and top_k < config.vocab_size:
                    v, _ = torch.topk(scaled_logits, k=top_k)
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf') # Mask out logits below k-th
                # Sample from the filtered distribution
                probabilities = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1) # [1, 1]
            else: # Greedy decoding (temperature == 0)
                next_token_id = torch.argmax(last_token_logits, dim=-1).unsqueeze(0) # [1, 1]

            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_id:
                break

            # Append generated token to the result list *and* the input for next step
            generated_token_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    # Decode the generated token IDs back to activity strings
    predicted_suffix_activities = tokenizer.decode_activities(generated_token_ids, char_to_activity_map)
    return predicted_suffix_activities

# --- Inference Execution ---
print("\n--- Predicting Suffixes for Test Prefixes ---")

if not config.test_prefixes:
    print("No test prefixes were generated (perhaps source traces were too short or split resulted in empty test set).")
else:
    for prefix in config.test_prefixes:
        print(f"Prefix: {list(prefix)}")
        predicted_suffix = predict_suffix(model, tokenizer, prefix,
                                        activity_to_char, char_to_activity,
                                        max_len=config.gen_max_len,
                                        temperature=config.gen_temperature,
                                        top_k=config.gen_top_k,
                                        config=config)
        print(f"Predicted Suffix: {predicted_suffix}\n")
