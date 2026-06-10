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


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def main():
    model = SimpleNet().eval()
    input_shape = (16,)
    output_dir = pathlib.Path("build")

    module = TorchKernelDeployer(model, input_shape)
    module.build(output_dir)

    print(f"Generated files in {output_dir}")
    print("Next: cd build && make user && ./user")


if __name__ == "__main__":
    main()
