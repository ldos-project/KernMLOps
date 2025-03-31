#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/vmalloc.h>
#include <linux/moduleparam.h>
#include <asm/fpu/api.h>
#include <asm/bug.h>

extern void matmul_kernel(
    float* A, float* B, float* C, 
    int M, int N, int K, 

    int stride_am, int stride_ak,
    int stride_bk, int stride_bn,
    int stride_cm, int stride_cn,

    int x, int y, int z, 
    int gridX, int gridY, int gridZ
);

extern void run_on_stack(void* new_stack, void(*f)(void));

MODULE_LICENSE("GPL");

int stack_size = 8388608; // 8 MiB
module_param(stack_size, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(stack_size, "Size of the Kernel Module Stack (in bytes)");

void* kernel_sp;

void work(void) {
    const int M = 16;
    const int N = 16;
    const int K = 16;

    float* A = vmalloc(M * K * sizeof (float));
    float* B = vmalloc(K * N * sizeof (float));
    float* C = vmalloc(M * N * sizeof (float));

    int i = 0;
    int j = 0;
    
    for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
            A[i * K + j] = i;
        }
    }

    for (i = 0; i < K; i++) {
        for (j = 0; j < N; j++) {
            B[i * N + j] = j;
        }
    }

    pr_info("Calling matmul kernel...\n");

    matmul_kernel(
        A, B, C,
        M, N, K,

        K, 1,
        N, 1,
        N, 1,

        0, 0, 0, 1, 1, 1
    );

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            if (C[i * N + j] != M * i * j) {
                pr_info("Matrix multiply failed at (%d, %d)\n", i, j);
            }
        }
    }
    pr_info("Matrix multiply done!");

    vfree(A);
    vfree(B);
    vfree(C);
}

int init_module(void) 
{ 
    char* new_stack = vmalloc(stack_size);
    char* new_sp = new_stack + stack_size;
    new_sp -= ((unsigned long)new_sp % 16);

    pr_info("Current Stack Start: %px\n", new_sp);
    pr_info("Current Stack End: %px\n", new_stack);


    kernel_fpu_begin();
    run_on_stack(new_sp, work);
    kernel_fpu_end(); 

    vfree(new_stack);

    /* A non 0 return means init_module failed; module can't be loaded. */ 
    return 0; 
}

void cleanup_module(void) 
{
    pr_info("kernel module goodbye\n");
} 
