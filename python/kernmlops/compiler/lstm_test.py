import torch
from gen_kernel_module import build


class VectorizedLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices
        self.W_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.W_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.b_ih = torch.nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = torch.nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, hx=None):
        """
        x: (seq_len, batch, input_size)
        hx: Tuple(h0, c0), each (batch, hidden_size)
        """
        seq_len, batch, _ = x.shape
        if hx is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = hx

        # Compute all gates for all timesteps at once
        gates = torch.matmul(x, self.W_ih.T) + self.b_ih  # (seq_len, batch, 4*hidden)

        # Precompute h contribution sequentially
        outputs = []
        for t in range(seq_len):
            gates_t = gates[t] + torch.matmul(h, self.W_hh.T) + self.b_hh
            i, f, g, o = gates_t.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)

        return torch.stack(outputs, dim=0), (h, c)


# ---------------------------
# Compile with TorchInductor
# ---------------------------
if __name__ == "__main__":
    seq_len, batch, input_size, hidden_size = 10, 10, 10, 10
    x = torch.randn(seq_len, batch, input_size)

    lstm = VectorizedLSTM(input_size, hidden_size)
    build(lstm, x)
