"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

* The basic programming model of Triton.

* The `triton.jit` decorator, which is used to define Triton kernels.

* The best practices for validating and benchmarking your custom ops against native reference implementations.

"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl

GPU_BLOCK_SIZE = 1024
CPU_BLOCK_SIZE = 4096
# Single Thread Threshold
CPU_ST_THRESHOLD = 65536
USE_GPU = False

@triton.jit
def add_kernel_tiled(x_ptr,  # *Pointer* to first input vector.
                     y_ptr,  # *Pointer* to second input vector.
                     output_ptr,  # *Pointer* to output vector.
                     n_elements,  # Size of the vector.
                     BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                     TILE_SIZE: tl.constexpr,  # Number of elements each iteration should process.
                     # NOTE `constexpr` so it can be used as a shape value.
                     ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
        offsets = block_start + i * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

def add_tiled(x: torch.Tensor, y: torch.Tensor, output):
    if output is None:
        output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel_tiled[grid](x, y, output, n_elements, BLOCK_SIZE=CPU_BLOCK_SIZE, TILE_SIZE=16)

    return output
    
# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:
torch.manual_seed(0)
size = 98432

triton.runtime.driver.set_active_to_cpu()
x = torch.rand(size, device='cpu')
y = torch.rand(size, device='cpu')
output_triton_cpu = add_tiled(x, y, None)

print("x:", x)
print("y:", y)
print("z:", output_triton_cpu)

