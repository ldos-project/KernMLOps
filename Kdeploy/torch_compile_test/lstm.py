import torch
import torch.nn as nn
from gen_kernel_module import build

device = 'cpu'

# Hyperparameters
seq_length = 5       # number of timesteps in each input sequence
input_size = 10      # number of features per timestep
hidden_size = 20     # LSTM hidden state size
num_layers = 1       # LSTM layers
output_size = 1      # regression output
batch_size = 3       # number of sequences in a batch

# Dummy input and target
# Shape: (seq_len, batch, input_size)
x = torch.randn(seq_length, batch_size, input_size).to(device)
y = torch.randn(seq_length, batch_size, output_size).to(device)

# Define the LSTM model
class LSTM10D(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM10D, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)         # out: (seq_len, batch, hidden_size)
        out = self.linear(out)        # out: (seq_len, batch, output_size)
        return out

# Initialize model
model = LSTM10D(input_size, hidden_size, num_layers, output_size).to(device)

#compiled = torch.compile(model, fullgraph=True, backend="inductor")
print(x, model(x))
#get_primals(model, data)

#print(compiled(data))
build(model, x, 'main.c')
