import torch
import triton
import triton.language as tl

# --- Tunable constants ---
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
GROUP_M = 8  # This is no longer used but kept for signature matching
# ------------------------------------------------

@triton.jit
def mm(M, N, K,
       a_ptr, b_ptr, c_ptr,
       stride_am, stride_ak,
       stride_bk, stride_bn,
       stride_cm, stride_cn,
       BLOCK_M: tl.constexpr,
       BLOCK_N: tl.constexpr,
       BLOCK_K: tl.constexpr,
       GROUP_M: tl.constexpr): # GROUP_M is unused here

    ALLOW_TF32 = False

    # --- Calculate total grid dimensions ---
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # --- Create loops to iterate over the grid ---
    for pid_m in range(0, grid_m):
        for pid_n in range(0, grid_n):

            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            mask_m = rm < M
            mask_n = rn < N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            num_k_tiles = tl.cdiv(K, BLOCK_K)
            for k_idx in range(0, num_k_tiles):
                offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
                k_mask = offs_k < K

                a = tl.load(
                    a_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=mask_m[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_ptr + offs_k[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=k_mask[:, None] & mask_n[None, :],  # <--- THIS LINE IS FIXD
                    other=0.0,
                )
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

            tl.store(
                c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
                acc,
                mask=mask_m[:, None] & mask_n[None, :],
            )


if __name__ == '__main__':
    M, N, K = 10, 20, 50

    A = torch.randn((M, K), dtype=torch.float32, device='cpu')
    B = torch.randn((K, N), dtype=torch.float32, device='cpu')
    C = torch.randn((M, N), dtype=torch.float32, device='cpu')

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    print(f"A stride: {A.stride()}, B stride: {B.stride()}, C stride: {C.stride()}")

    # --- Hardcode grid as requested ---
    grid = (1,)
    print(f"Running on a single grid: {grid}")

    mm[grid]( # <--- This is now (1,)
        M, N, K,
        A, B, C,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M
    )

    # check result
    ref = A @ B
    max_err = (C - ref).abs().max().item()
    print(f"max error: {max_err}")
