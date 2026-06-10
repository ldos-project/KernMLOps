import pathlib
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch
import torch.nn as nn

try:
    from .gen_kernel_module import TorchKernelDeployer
except ImportError:
    from gen_kernel_module import TorchKernelDeployer


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x):
        seq_len, batch, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            gates = x[t] @ self.W_ih.T + h @ self.W_hh.T + self.b_ih + self.b_hh
            i, f, g, o = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)

        return torch.stack(outputs, dim=0)


def main():
    seq_len, batch, input_size, hidden_size = 4, 2, 3, 5
    model = SimpleLSTM(input_size, hidden_size).eval()
    input_shape = (seq_len, batch, input_size)
    output_dir = pathlib.Path("build_lstm")

    module = TorchKernelDeployer(model, input_shape)
    module.build(output_dir)

    print(f"Generated files in {output_dir}")
    print("Next: cd build_lstm && make user && ./user")


if __name__ == "__main__":
    main()
