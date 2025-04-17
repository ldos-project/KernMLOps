import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.activation import ReLU
import numpy as np

# Define the LSTMModel class that matches the loaded model structure
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # Reshape input if needed - x should be [batch_size, seq_length, input_size]
        if len(x.shape) == 2:  # [batch_size, features]
            # Add sequence dimension
            x = x.unsqueeze(1)
        elif len(x.shape) == 1:  # [features]
            # Add batch and sequence dimensions
            x = x.unsqueeze(0).unsqueeze(0)

        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Apply remaining layers
        out = self.dropout(out[:, -1, :])  # Take the last time step output
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

# Add required classes to safe globals
torch.serialization.add_safe_globals([LSTM, LSTMModel, Dropout, Linear, ReLU])

# Load the model with weights_only=False
try:
    model = torch.load('test_model.pth', weights_only=False)
    print("Successfully loaded model")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Print model structure
print("\nModel structure:")
print(model)

# Print all named parameters and their shapes
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Print detailed parameter values for each layer
print("\n--- DETAILED PARAMETER VALUES ---")

# Print LSTM parameters
print("\nLSTM Weights:")
for name, param in model.named_parameters():
    if 'lstm' in name:
        print(f"{name}:")
        # For large parameters, print statistics instead of full values
        if param.numel() > 100:
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
            print(f"  Min: {param.min().item():.6f}")
            print(f"  Max: {param.max().item():.6f}")
            print(f"  First 10 values: {param.flatten()[:10].tolist()}")
            print(f"  Last 10 values: {param.flatten()[-10:].tolist()}")
        else:
            print(param.tolist())

# Print FC1 parameters
print("\nFC1 Weights and Biases:")
for name, param in model.named_parameters():
    if 'fc1' in name:
        print(f"{name}:")
        if param.numel() > 100:
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
            print(f"  Min: {param.min().item():.6f}")
            print(f"  Max: {param.max().item():.6f}")
            print(f"  First 10 values: {param.flatten()[:10].tolist()}")
            print(f"  Last 10 values: {param.flatten()[-10:].tolist()}")
        else:
            print(param.tolist())

# Print FC2 parameters
print("\nFC2 Weights and Biases:")
for name, param in model.named_parameters():
    if 'fc2' in name:
        print(f"{name}:")
        if param.numel() > 100:
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
            print(f"  Min: {param.min().item():.6f}")
            print(f"  Max: {param.max().item():.6f}")
            print(f"  First 10 values: {param.flatten()[:10].tolist()}")
            print(f"  Last 10 values: {param.flatten()[-10:].tolist()}")
        else:
            print(param.tolist())

# Set to evaluation mode (important for dropout layers)
model.eval()

# Create input tensor with 2 features
INPUT_SIZE = 2
input_tensor = torch.full((INPUT_SIZE,), 0.1, dtype=torch.float32)
print(f"\nInput tensor: {input_tensor}")

# Track intermediate outputs at each layer
print("\n--- INTERMEDIATE OUTPUTS ---")

# Create a copy of the model's forward method to add debugging
def forward_with_debug(model, x):
    # Reshape input if needed
    if len(x.shape) == 1:  # [features]
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, features]
    elif len(x.shape) == 2:  # [batch, features]
        x = x.unsqueeze(1)  # [batch, 1, features]

    print(f"Input shape after reshaping: {x.shape}")
    print(f"Input values: {x}")

    # Initialize hidden state with zeros
    batch_size = x.size(0)
    h0 = torch.zeros(1, batch_size, model.lstm.hidden_size, device=x.device)
    c0 = torch.zeros(1, batch_size, model.lstm.hidden_size, device=x.device)

    # LSTM forward pass
    lstm_out, (h_n, c_n) = model.lstm(x, (h0, c0))
    print(f"LSTM output shape: {lstm_out.shape}")
    print(f"LSTM output mean: {lstm_out.mean().item():.6f}, std: {lstm_out.std().item():.6f}")
    print(f"LSTM output first 5 values: {lstm_out[0, -1, :5].tolist()}")

    # Dropout (training mode affects this)
    dropout_out = model.dropout(lstm_out[:, -1, :])
    print(f"Dropout output shape: {dropout_out.shape}")
    print(f"Dropout output mean: {dropout_out.mean().item():.6f}, std: {dropout_out.std().item():.6f}")
    print(f"Dropout output first 5 values: {dropout_out[0, :5].tolist()}")

    # First Linear layer
    fc1_out = model.fc1(dropout_out)
    print(f"FC1 output shape: {fc1_out.shape}")
    print(f"FC1 output mean: {fc1_out.mean().item():.6f}, std: {fc1_out.std().item():.6f}")
    print(f"FC1 output values: {fc1_out[0].tolist()}")

    # ReLU activation
    relu_out = model.relu(fc1_out)
    print(f"ReLU output shape: {relu_out.shape}")
    print(f"ReLU output mean: {relu_out.mean().item():.6f}, std: {relu_out.std().item():.6f}")
    print(f"ReLU output values: {relu_out[0].tolist()}")

    # Second Linear layer (output)
    fc2_out = model.fc2(relu_out)
    print(f"FC2 output shape: {fc2_out.shape}")
    print(f"FC2 output values: {fc2_out[0].tolist()}")

    return fc2_out

# Run with debugging
with torch.no_grad():
    output = forward_with_debug(model, input_tensor)

# Format and print the final output
def print_arr(arr):
    # Remove batch dimension if it exists
    if arr.dim() > 1:
        arr = arr.squeeze(0)

    print(f"\nFinal output shape for printing: {arr.shape}")

    for i, x in enumerate(arr):
        x_float = float(x)
        int_x = int(x_float)
        dec_x = int((x_float - int_x) * 1000)
        print(f"Output[{i}]: {int_x}.{dec_x}")

print("\nFinal PyTorch Output:")
print_arr(output)

# Save the results to a file
with open('pytorch_results.txt', 'w') as f:
    # Remove batch dimension if it exists
    if output.dim() > 1:
        output_to_save = output.squeeze(0)
    else:
        output_to_save = output

    for x in output_to_save:
        x_float = float(x)
        int_x = int(x_float)
        dec_x = int((x_float - int_x) * 1000)
        f.write(f"{int_x}.{dec_x}\n")

print(f"\nResults saved to pytorch_results.txt")
