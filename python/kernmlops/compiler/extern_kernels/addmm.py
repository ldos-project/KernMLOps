import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=range(16))
def addmm(M, N, K,
          a_ptr, b_ptr, c_ptr, out_ptr,
          stride_am, stride_ak,
          stride_bk, stride_bn,
          stride_cm, stride_cn,
          stride_om, stride_on,
          alpha, beta):
    """
    Full addmm inside Triton:
        OUT = beta*C + alpha*(A@B)
    Single Triton program for entire matrix
    """
    for row in range(M):
        for col in range(N):
            acc = 0.0
            for k in range(K):
                a_val = tl.load(a_ptr + row*stride_am + k*stride_ak)
                b_val = tl.load(b_ptr + k*stride_bk + col*stride_bn)
                acc += a_val * b_val
            c_val = tl.load(c_ptr + row*stride_cm + col*stride_cn)
            tl.store(out_ptr + row*stride_om + col*stride_on,
                     beta * c_val + alpha * acc)

if __name__ == '__main__':
    M, N, K = 1, 20, 5

    A = torch.randn((M, K), dtype=torch.float32, device='cpu')
    B = torch.randn((K, N), dtype=torch.float32, device='cpu')
    C = torch.randn((M, N), dtype=torch.float32, device='cpu')
    D = torch.randn((M, N), dtype=torch.float32, device='cpu')

    alpha, beta = 1.1, 1.2

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    stride_dm, stride_dn = D.stride()
    print(A.stride(), B.stride(), C.stride(), D.stride())

    addmm[lambda meta: (1, )](
        M, N, K,
        A, B, C, D,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_dm, stride_dn,
        alpha, beta
    )

    # check result
    ref = beta * C + alpha * (A @ B)
    print("max error:", (D - ref).abs().max().item())
