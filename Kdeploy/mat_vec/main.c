#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/vmalloc.h>
#include <linux/moduleparam.h>

extern void gemv_kernel(int* Y, int* A, int* X, int M, int N, int stride_am,
                        int x, int y, int z, int gridX, int gridY, int gridZ);

MODULE_LICENSE("GPL");

int stack_size = 8388608; // 8 MiB
module_param(stack_size, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(stack_size, "Size of the Kernel Module Stack (in bytes)");

void* kernel_sp;

void work(void) {
    const int M = 128;
    const int N = 128;

    int* A = vmalloc(M * N * sizeof (int));
    int* X = vmalloc(N * sizeof (int));
    int* Y = vmalloc(M * sizeof (int));

    int i = 0;
    int j = 0;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            if (i == j) {
                A[j * N + i] = 2;
            }
            else {
                A[j * N + i] = 0;
            }
        }
        X[i] = i;
    }

    gemv_kernel(Y, A, X, M, N, N, 0, 0, 0, 1, 1, 1);

    for (i = 0; i < M; i++) {
        pr_info("%d %d\n", X[i], Y[i]);
    }

    vfree(A);
    vfree(X);
    vfree(Y);
}

int init_module(void) 
{ 
    pr_info("Hello!\n");
    char* new_stack = vmalloc(stack_size);
    char* new_sp = new_stack + stack_size;
    asm("mov %%rsp, %0" : "=r"(kernel_sp)); // save current stack pointer
    asm("mov %0, %%rsp" : : "r"(new_sp));   // switch stacks
    
    pr_info("Current Stack: %p\n", new_sp);
    work();
     
    asm("mov %0, %%rsp" : : "r"(kernel_sp)); // restore stack pointer
    vfree(new_stack);

    /* A non 0 return means init_module failed; module can't be loaded. */ 
    return 0; 
}

void cleanup_module(void) 
{
    pr_info("kernel module goodbye\n");
} 
