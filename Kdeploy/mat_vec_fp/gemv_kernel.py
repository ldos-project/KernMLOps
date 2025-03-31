import torch

import triton
import triton.language as tl

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
USE_GPU = False
"""
Kernel for computing Y = A @ X, where A is a dense matrix with
M rows and N columns.
- Input X has shape (N,)
- A has shape (M, N)
- Output has shape (M,)
"""


@triton.jit
def gemv_kernel(
    Y,
    A,
    X,
    M,
    N,
    stride_am,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)

    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rn

    acc = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    for n in range(N, 0, -BLOCK_SIZE_N):
        a = tl.load(A)
        x = tl.load(X)
        acc += tl.sum(a * x[None, :], axis=1)
        A += BLOCK_SIZE_N
        X += BLOCK_SIZE_N

    Y = Y + rm
    tl.store(Y, acc)


def gemv(
    weight: torch.Tensor,
    x: torch.Tensor,
    output: torch.Tensor,
    num_threads=0,
):
    assert weight.shape[1] == x.shape[0], "Incompatible dimensions"
    assert weight.is_contiguous() and x.is_contiguous(), "Input and weight must be contiguous"
    assert x.dtype == weight.dtype, f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"

    M, N = weight.shape

    # TODO: Currently masked load is not supported yet.
    assert M % BLOCK_SIZE_M == 0 and N % BLOCK_SIZE_N == 0, "Masking currently not supported, Matrix dimensions must be multiples of block size"

    if output is None:
        # Allocates output.
        output = torch.empty((M, ), device=x.device, dtype=x.dtype)
    else:
        assert output.shape == (M, ) and output.dtype == x.dtype, "Incompatible output"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), )

    gemv_kernel[grid](output, weight, x, M, N, weight.stride(0), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                      num_threads=num_threads)

    return output


torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

weight = torch.randn((128, 128), device='cpu', dtype=torch.float32)
x = torch.randn((128), device='cpu', dtype=torch.float32)
triton_output = gemv(weight, x, None)
# torch.matmul will select bf16 kernels on Linux Arm if x is 1-d, which has lower precision.
# So we reshape x to be 2-d, which will invoke different kernels.
torch_output = torch.matmul(weight, x[:, None]).reshape(-1)
#print(f"triton_cpu_output_with_{weight.dtype}_inputs={triton_output}")
#print(f"torch_cpu_output_with_{weight.dtype}_inputs={torch_output}")
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=rtol):
    print("✅ TritonCPU and TorchCPU match")
else:
    print("❌ TritonCPU and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
