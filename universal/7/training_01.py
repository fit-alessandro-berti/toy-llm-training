import math
import torch
import torch.nn as nn

# --- Configuration (Minimal for the specified components) ---

# Architectural Params for Per-sensor Temporal Encoder
SEQ_LEN = 64  # Sequence length, needed for PositionalEncoding max_len
SENSOR_INPUT_DIM = 1  # Input dimension for each sensor's time series data point

# I. Per-sensor Temporal Encoder (TCN based)
SENSOR_TCN_PROJ_DIM = 32  # Dimension after initial projection, input to first TCN layer
SENSOR_TCN_OUT_DIM = 32  # Output channels of each TCN block and final TCN encoder output
TCN_LEVELS = 4  # Number of TCN residual blocks
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

# --- Helper: Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # pe shape [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, SeqLen, d_model]
        return x + self.pe[:, :x.size(1), :]


# --- TCN Residual Block ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        try:
            # For PyTorch 1.9+
            self.conv1 = nn.utils.parametrizations.weight_norm(self.conv1)
        except AttributeError:
            # For older PyTorch versions
            self.conv1 = nn.utils.weight_norm(self.conv1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        try:
            self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)
        except AttributeError:
            self.conv2 = nn.utils.weight_norm(self.conv2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()

    def forward(self, x):  # x: [Batch, Channels_in, SeqLen]
        out = self.conv1(x)
        # Adjust slicing for padding to maintain sequence length
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


# --- Per-Sensor TCN Encoder ---
class PerSensorEncoderTCN(nn.Module):
    def __init__(self, input_dim, proj_dim, tcn_out_dim, seq_len_for_pe, num_levels, kernel_size, dropout):
        super(PerSensorEncoderTCN, self).__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        # seq_len_for_pe is used for max_len in PositionalEncoding, actual sequence length determined by input
        self.pos_encoder = PositionalEncoding(proj_dim, max_len=seq_len_for_pe)

        tcn_blocks = []
        current_channels = proj_dim
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels_block = tcn_out_dim
            tcn_blocks.append(TemporalBlock(current_channels, out_channels_block, kernel_size, stride=1,
                                            dilation=dilation_size, dropout=dropout))
            current_channels = out_channels_block # For next block, input channels = output channels of current

        self.tcn_network = nn.Sequential(*tcn_blocks)
        self.final_norm = nn.LayerNorm(tcn_out_dim)

    def forward(self, x):
        # x: [Batch_Combined, SeqLen, InputDim] (Batch_Combined = Batch_Actual * NumSensors)
        x = self.input_proj(x) # Output: [Batch_Combined, SeqLen, ProjDim]
        x = self.pos_encoder(x) # Output: [Batch_Combined, SeqLen, ProjDim]

        x = x.permute(0, 2, 1)  # Output: [Batch_Combined, ProjDim, SeqLen] (for Conv1D)
        x = self.tcn_network(x) # Output: [Batch_Combined, TCN_Out_Dim, SeqLen]
        x = x.permute(0, 2, 1)  # Output: [Batch_Combined, SeqLen, TCN_Out_Dim] (back to original dim order)
        x = self.final_norm(x)  # Output: [Batch_Combined, SeqLen, TCN_Out_Dim]
        return x


# --- Simplified Model: Sensor Encoder + Prediction Head for Horizon 5 ---
class SensorEncoderPredictorH5(nn.Module):
    def __init__(self, sensor_input_dim, sensor_tcn_proj_dim, sensor_tcn_out_dim,
                 seq_len_for_pe, tcn_levels, tcn_kernel_size, tcn_dropout):
        super().__init__()

        self.sensor_input_dim = sensor_input_dim # Should be SENSOR_INPUT_DIM (e.g., 1)
        self.sensor_tcn_out_dim = sensor_tcn_out_dim

        self.per_sensor_encoder = PerSensorEncoderTCN(
            input_dim=sensor_input_dim,
            proj_dim=sensor_tcn_proj_dim,
            tcn_out_dim=sensor_tcn_out_dim,
            seq_len_for_pe=seq_len_for_pe, # Max sequence length for PE
            num_levels=tcn_levels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout
        )

        # Prediction head for a single horizon (horizon 5)
        # Input dimension is the output dimension of the TCN encoder
        self.pred_head_h5 = nn.Linear(sensor_tcn_out_dim, 1)

    def forward(self, x_features, sensor_mask=None):
        """
        Args:
            x_features (torch.Tensor): Input sensor data.
                                       Shape: [Batch, SeqLen, NumSensors].
                                       Assumed to be normalized.
            sensor_mask (torch.Tensor, optional): Boolean or float mask for sensors.
                                                 Shape: [Batch, NumSensors].
                                                 Used to mask the final output.
        Returns:
            torch.Tensor: Prediction for time horizon 5.
                          Shape: [Batch, NumSensors, 1].
        """
        batch_size, seq_len, num_sensors = x_features.shape

        # Reshape for PerSensorEncoderTCN:
        # Each sensor's time series is treated as an item in a larger batch.
        # [B, SL, NS] -> [B, NS, SL]
        x_permuted = x_features.permute(0, 2, 1)

        # [B, NS, SL] -> [B, NS, SL, SID] where SID is SENSOR_INPUT_DIM
        x_with_input_dim = x_permuted.unsqueeze(-1)
        if x_with_input_dim.shape[-1] != self.sensor_input_dim:
            raise ValueError(f"Last dimension of input after unsqeeze ({x_with_input_dim.shape[-1]}) "
                             f"must match SENSOR_INPUT_DIM ({self.sensor_input_dim})")

        # [B, NS, SL, SID] -> [B*NS, SL, SID]
        x_reshaped_for_encoder = x_with_input_dim.reshape(batch_size * num_sensors, seq_len, self.sensor_input_dim)

        # Encode: Output shape [B*NS, SL, SENSOR_TCN_OUT_DIM]
        sensor_temporal_features_flat = self.per_sensor_encoder(x_reshaped_for_encoder)

        # Reshape back: [B*NS, SL, SENSOR_TCN_OUT_DIM] -> [B, NS, SL, SENSOR_TCN_OUT_DIM]
        sensor_temporal_features = sensor_temporal_features_flat.reshape(
            batch_size, num_sensors, seq_len, self.sensor_tcn_out_dim
        )

        # Take features from the last time step for prediction
        # Shape: [B, NS, SENSOR_TCN_OUT_DIM]
        encoded_last_step = sensor_temporal_features[:, :, -1, :]

        # Predict horizon 5: Output shape [B, NS, 1]
        prediction_h5 = self.pred_head_h5(encoded_last_step)

        if sensor_mask is not None:
            # Ensure sensor_mask is [B, NS, 1] for broadcasting
            prediction_h5 = prediction_h5 * sensor_mask.unsqueeze(-1)

        return prediction_h5

if __name__ == '__main__':
    # Example Usage (assuming you have some dummy data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation
    model = SensorEncoderPredictorH5(
        sensor_input_dim=SENSOR_INPUT_DIM,
        sensor_tcn_proj_dim=SENSOR_TCN_PROJ_DIM,
        sensor_tcn_out_dim=SENSOR_TCN_OUT_DIM,
        seq_len_for_pe=SEQ_LEN, # Max length for PE, actual length from input
        tcn_levels=TCN_LEVELS,
        tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dropout=TCN_DROPOUT
    ).to(device)

    model.eval() # Set to evaluation mode if not training

    # Dummy input data
    batch_s = 4
    num_s = 10 # Number of sensors
    dummy_x_features = torch.randn(batch_s, SEQ_LEN, num_s).to(device)
    dummy_sensor_mask = torch.ones(batch_s, num_s).to(device) # Assuming all sensors are active
    # In a real scenario, some sensors might be masked:
    # dummy_sensor_mask[0, 2:] = 0 # Example: for 0-th batch item, sensors from index 2 are inactive

    with torch.no_grad():
        output_prediction_h5 = model(dummy_x_features, dummy_sensor_mask)

    print("Model instantiated and ran successfully.")
    print(f"Input x_features shape: {dummy_x_features.shape}")
    print(f"Input sensor_mask shape: {dummy_sensor_mask.shape}")
    print(f"Output prediction_h5 shape: {output_prediction_h5.shape}")
    # Expected output shape: [batch_s, num_s, 1]
