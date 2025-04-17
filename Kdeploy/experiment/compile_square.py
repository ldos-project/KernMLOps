
import torch

# A simple PyTorch function to compile
def fn(x):
    return x * x  # or use torch.square(x)

# Input tensor
x = torch.randn(1423763, device="cuda")  # must be CUDA

# Compile it using TorchInductor
compiled_fn = torch.compile(fn)
out = compiled_fn(x)
