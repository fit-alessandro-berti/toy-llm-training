import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import time
import json
import os
import random


def load_qa_data():
    files = os.listdir("data")
    files = sorted(files)
    qa_data = []
    for file in files:
        file_path = os.path.join("data", file)
        contents = json.load(open(file_path, "r"))
        qa_data.append((str(contents["values"]), contents["format"]))
    return qa_data


# --- Configuration ---
class Config:
    # Data
    # Simple dummy QA dataset. Replace with your actual data.
    # Format: list of tuples (question, answer)
    all_qa_data = load_qa_data()
    qa_data = all_qa_data[:int(len(all_qa_data)*1.1)]
    test_qa_data = all_qa_data[int(len(all_qa_data)*0.75):]

    #print(qa_data); input()
    #print(test_qa_data, len(test_qa_data))

    # Combine all text to build vocabulary initially
    corpus_for_vocab = " ".join([q + " " + a for q, a in qa_data])

    # Special Tokens
    PAD_TOKEN = "[PAD]"
    SOS_TOKEN = "[SOS]" # Start of Sequence
    EOS_TOKEN = "[EOS]" # End of Sequence
    SEP_TOKEN = "[SEP]" # Separator between question and answer
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN]

    # Model Architecture (Adjust as needed, check parameter count)
    vocab_size = 110 # Will be set after tokenization
    d_model = 128
    num_layers = 3
    num_heads = 4
    d_ff = 256
    dropout = 0.1
    max_seq_len = 1024 # Max length for combined "SOS Q SEP A EOS" sequence during training

    # Training
    batch_size = 8 # Might need smaller batch size due to longer sequences
    learning_rate = 1e-3
    epochs = 50 # Increase significantly for real training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_token_id = 0 # Will be set after tokenization

    # Generation
    gen_max_len = 50 # Max length of the *generated answer*
    gen_temperature = 0.7
    gen_top_k = 5

config = Config()

# --- 1. Data Preparation ---

# Simple Character-level Tokenizer (adapted for special tokens)
class QATokenizer:
    def __init__(self, corpus, special_tokens):
        self.special_tokens = special_tokens
        unique_chars = sorted(list(set(corpus)))
        self.vocab = special_tokens + [ch for ch in unique_chars if ch not in special_tokens]
        self.vocab_size = len(self.vocab)

        self.token_to_int = {token: i for i, token in enumerate(self.vocab)}
        self.int_to_token = {i: token for i, token in enumerate(self.vocab)}

        self.pad_id = self.token_to_int[config.PAD_TOKEN]
        self.sos_id = self.token_to_int[config.SOS_TOKEN]
        self.eos_id = self.token_to_int[config.EOS_TOKEN]
        self.sep_id = self.token_to_int[config.SEP_TOKEN]

    def encode(self, text, add_special_tokens=False):
        tokens = list(text) # Character-level
        if add_special_tokens:
             # Basic handling - might need more robust tokenization in practice
             # Ensure special tokens aren't split if they appear in text
             # (Less likely with char tokens, more relevant for word/subword)
             pass
        return [self.token_to_int.get(token, -1) for token in tokens] # Handle unknowns?

    def decode(self, token_ids):
        # Exclude padding tokens from the decoded string
        return "".join([self.int_to_token.get(token_id, '')
                       for token_id in token_ids if token_id != self.pad_id])

# Initialize tokenizer and update config
tokenizer = QATokenizer(config.corpus_for_vocab, config.special_tokens)
config.vocab_size = tokenizer.vocab_size
config.pad_token_id = tokenizer.pad_id
print(f"Vocabulary Size: {config.vocab_size}")
print(f"PAD ID: {config.pad_token_id}")
# print(f"Vocabulary: {' '.join(tokenizer.vocab)}") # Can be long

# Dataset for Question Answering
class QADataset(Dataset):
    def __init__(self, qa_data, tokenizer, max_seq_len):
        self.qa_data = qa_data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        question, answer = self.qa_data[idx]

        q_tokens = self.tokenizer.encode(question)
        a_tokens = self.tokenizer.encode(answer)

        # Prepare combined sequence: [SOS] Q [SEP] A [EOS]
        combined_tokens = ([self.tokenizer.sos_id] +
                           q_tokens +
                           [self.tokenizer.sep_id] +
                           a_tokens +
                           [self.tokenizer.eos_id])

        # Truncate if exceeds max length
        if len(combined_tokens) > self.max_seq_len:
             # Simple truncation - might cut off important info. Better strategies exist.
             # Ensure EOS is always present if truncated before it
             combined_tokens = combined_tokens[:self.max_seq_len-1] + [self.tokenizer.eos_id]


        # Input 'x' is the sequence except the last token
        x = torch.tensor(combined_tokens[:-1], dtype=torch.long)
        # Target 'y' is the sequence shifted right (starts from first Q token, ends with EOS)
        y = torch.tensor(combined_tokens[1:], dtype=torch.long)

        return x, y

# Collate function to handle padding within batches
def create_collate_fn(pad_token_id):
    def collate_fn(batch):
        # Separate sequences x and y
        batch_x, batch_y = zip(*batch)

        # Pad sequences in the batch to the length of the longest sequence *in that batch*
        padded_x = pad_sequence(batch_x, batch_first=True, padding_value=pad_token_id)
        padded_y = pad_sequence(batch_y, batch_first=True, padding_value=pad_token_id)

        return padded_x, padded_y
    return collate_fn

# Create dataset and dataloader
qa_dataset = QADataset(config.qa_data, tokenizer, config.max_seq_len)
collate_fn = create_collate_fn(config.pad_token_id)
qa_dataloader = DataLoader(qa_dataset, batch_size=config.batch_size,
                           shuffle=True, collate_fn=collate_fn)

# --- 2. Model Architecture (Largely Unchanged, but Positional Encoding needs updated max_len) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # Ensure max_len is sufficient
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] (because batch_first=True)
        # pe shape: [max_len, 1, d_model] -> needs slicing and broadcasting
        x = x + self.pe[:x.size(1), :].squeeze(1) # Slice pe to match seq_len and remove dim 1
        return self.dropout(x)


class TinyTransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        # Embedding layer now needs padding_idx so PAD tokens don't contribute gradients
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model,
                                            padding_idx=config.pad_token_id)
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout,
                                              max_len=config.max_seq_len + 1) # Use max_seq_len from config

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True, # Crucial: Input format is (batch, seq, feature)
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
        # Ensure padding index embedding is zero and stays zero
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz, device):
        """Generates a causal mask for the decoder."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_padding_mask=None):
        """
        Args:
            src: Input tensor, shape [batch_size, seq_len]
            src_padding_mask: Bool tensor, shape [batch_size, seq_len], True where padded
        """
        # 1. Embedding & Positional Encoding
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb) # Already handles batch_first

        # 2. Causal Mask (for self-attention)
        seq_len = src.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, src.device)

        # 3. Transformer Decoder
        # `memory` is not used in decoder-only models in the typical sense (no encoder).
        # We pass src_emb as both `tgt` and `memory`.
        # `tgt_key_padding_mask` should be the same as `src_padding_mask` here.
        output = self.transformer_decoder(
            tgt=src_emb,                     # Target sequence (input to decoder)
            memory=src_emb,                  # Memory (source for key/values, same as tgt)
            tgt_mask=tgt_mask,               # Causal mask for self-attention
            memory_mask=None,                # No cross-attention mask
            tgt_key_padding_mask=src_padding_mask, # Mask for padding in tgt
            memory_key_padding_mask=src_padding_mask # Mask for padding in memory (same here)
        )
        # output shape: [batch_size, seq_len, d_model]

        # 4. Output Layer
        logits = self.output_layer(output)
        # logits shape: [batch_size, seq_len, vocab_size]
        return logits

# --- 3. Training Loop (Adapted for Padding) ---

model = TinyTransformerLM(config).to(config.device)

# Calculate parameter count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")

# Loss function ignores padding tokens in the target
criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

def train(model, dataloader, criterion, optimizer, config):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)

        # Create padding mask (True where padded)
        # Shape: [batch_size, seq_len]
        src_padding_mask = (batch_x == config.pad_token_id)

        optimizer.zero_grad()

        # Forward pass with padding mask
        logits = model(batch_x, src_padding_mask=src_padding_mask)

        # Calculate loss - criterion ignores pad_token_id in batch_y automatically
        loss = criterion(logits.view(-1, config.vocab_size), batch_y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        log_interval = max(1, len(dataloader) // 5) # Prevent division by zero for small datasets
        if i > 0 and i % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'| Epoch {epoch:3d} | {i:4d}/{len(dataloader):4d} batches | '
                  f'lr {lr:02.2e} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            total_loss = 0
            start_time = time.time()

# --- Training Execution ---
print(f"Starting QA training on {config.device}...")
for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train(model, qa_dataloader, criterion, optimizer, config)
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s |')
    print('-' * 89)
    # Add evaluation pass here if you have a validation set

# --- 4. Inference (Text Generation for QA) ---

def generate_answer(model, tokenizer, question, max_len, temperature, top_k, config):
    model.eval()
    q_tokens = tokenizer.encode(question)

    # Prepare input: [SOS] Question [SEP]
    input_tokens = ([tokenizer.sos_id] + q_tokens + [tokenizer.sep_id])
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(config.device)

    generated_token_ids = []

    with torch.no_grad():
        for _ in range(max_len):
            # Create padding mask (no padding needed for single sequence inference usually)
            # But the model expects the argument, so pass None or an all-False mask
            current_seq_len = input_ids.size(1)
            padding_mask = torch.zeros(1, current_seq_len, dtype=torch.bool, device=config.device)

            # If sequence gets longer than model max_seq_len, truncate input
            if current_seq_len > config.max_seq_len:
                 # Keep the most recent `max_seq_len` tokens
                 input_ids = input_ids[:, -config.max_seq_len:]
                 padding_mask = padding_mask[:, -config.max_seq_len:] # Adjust mask accordingly


            logits = model(input_ids, src_padding_mask=padding_mask) # [1, current_seq_len, vocab_size]
            last_token_logits = logits[:, -1, :] # [1, vocab_size]

            # Apply temperature and Top-K sampling
            if temperature > 0:
                scaled_logits = last_token_logits / temperature
                if top_k > 0 and top_k < config.vocab_size: # Ensure top_k is valid
                    top_k_logits, top_k_indices = torch.topk(scaled_logits, k=top_k)
                    mask = torch.full_like(scaled_logits, float('-inf'))
                    mask.scatter_(1, top_k_indices, top_k_logits)
                    scaled_logits = mask
                probabilities = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1) # [1, 1]
            else: # Greedy decoding
                next_token_id = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)


            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_id:
                break

            # Append generated token to the result list *and* the input for next step
            generated_token_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    # Decode the generated tokens (the answer part)
    answer = tokenizer.decode(generated_token_ids)
    return answer

# --- Inference Execution ---
print("\n--- Generating Answers ---")

test_questions = [x[0] for x in Config.test_qa_data]

for q in test_questions:
    print(f"Q: {q}")

    generated_ans = generate_answer(model, tokenizer, q,
                                    max_len=config.gen_max_len,
                                    temperature=config.gen_temperature,
                                    top_k=config.gen_top_k,
                                    config=config)
    print(f"A: {generated_ans}\n")
