import torch

import triton
import triton.language as tl

BLOCK_SIZE = 8 
BLOCK_SIZE_M = BLOCK_SIZE 
BLOCK_SIZE_N = BLOCK_SIZE 
"""
Kernel for computing Y = A @ X, where A is a dense matrix with
M rows and N columns.
- Input X has shape (N,)
- A has shape (M, N)
- Output has shape (M,)
"""


@triton.jit
def gemv_add_relu_kernel(
    Y,
    A,
    X,
    B,
    M,
    N,
    stride_am,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # RELU(A * X + B)
    # A: (M, N), X: (N,), B: (M,)

    # Each program block handles BLOCK_SIZE_M rows.
    pid = tl.program_id(0)
    rm = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    mask = rm < M 
    acc = tl.load(B + rm, mask=mask)
    #acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Compute the number of iterations required to cover the columns.
    num_iters = tl.cdiv(N, BLOCK_SIZE_N)
    for i in range(num_iters):
        col_offset = i * BLOCK_SIZE_N
        rn = tl.arange(0, BLOCK_SIZE_N)
        # Create a mask for columns that are within the valid range.
        mask1d = rn < (N - col_offset)
        valid = tl.broadcast_to(mask1d[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        #tl.device_print("N", N) 
        #tl.device_print("col", col_offset) 

        # Compute pointers for A and X for the current column block.
        a_ptr = A + (rm[:, None] * stride_am + (col_offset + rn)[None, :])
        x_ptr = X + col_offset + rn

        # Use masked loads so out-of-bound columns return 0.
        a = tl.load(a_ptr, mask=valid, other=0.0)
        x = tl.load(x_ptr, mask=mask1d, other=0.0)
        acc += tl.sum(a * x[None, :], axis=1)

    # Apply ReLU: set negative accumulated values to 0.
    result = tl.maximum(acc, 0.0)
    # Compute the output pointer.
    y_ptr = Y + rm
    # Mask for rows (in case M is not a multiple of BLOCK_SIZE_M)
    valid_rows = rm < M
    tl.store(y_ptr, result, mask=valid_rows)



def gemv(
    weight: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    num_threads=0,
):
    assert weight.shape[1] == x.shape[0], "Incompatible dimensions"
    assert weight.is_contiguous() and x.is_contiguous(), "Input and weight must be contiguous"
    assert x.dtype == weight.dtype, f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"

    M, N = weight.shape

    if output is None:
        # Allocates output.
        output = torch.empty((M, ), device=x.device, dtype=x.dtype)
    else:
        assert output.shape == (M, ) and output.dtype == x.dtype, "Incompatible output"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), )
    print(M, BLOCK_SIZE_M, grid({"BLOCK_SIZE_M": BLOCK_SIZE_M}))
    print("stride", weight.stride(0))

    gemv_add_relu_kernel[grid](output, weight, x, b, M, N, weight.stride(0), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                      num_threads=num_threads)

    return output


torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

weight = torch.randn((935, 572), device='cpu', dtype=torch.float32)
x = torch.randn((572), device='cpu', dtype=torch.float32)
bias = torch.randn((935), device='cpu', dtype=torch.float32)

triton_output = gemv(weight, x, bias, None)

# torch.matmul will select bf16 kernels on Linux Arm if x is 1-d, which has lower precision.
# So we reshape x to be 2-d, which will invoke different kernels.
torch_output = torch.nn.functional.relu(torch.matmul(weight, x[:, None]).reshape(-1) + bias)
#print(f"triton_cpu_output_with_{weight.dtype}_inputs={triton_output}")
#print(f"torch_cpu_output_with_{weight.dtype}_inputs={torch_output}")
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=rtol):
    print("✅ TritonCPU and TorchCPU match")
else:
    print("❌ TritonCPU and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
