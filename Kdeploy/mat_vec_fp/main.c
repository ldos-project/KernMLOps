#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/vmalloc.h>
#include <linux/moduleparam.h>
#include <asm/fpu/api.h>
#include <asm/bug.h>

extern void gemv_kernel(float* Y, float* A, float* X, int M, int N, int stride_am,
                        int x, int y, int z, int gridX, int gridY, int gridZ);

extern void run_on_stack(void* new_stack, void(*f)(void));

MODULE_LICENSE("GPL");

int stack_size = 8388608; // 8 MiB
module_param(stack_size, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(stack_size, "Size of the Kernel Module Stack (in bytes)");

void* kernel_sp;

void work(void) {
    const int M = 128;
    const int N = 128;

    float* A = vmalloc(M * N * sizeof (float));
    float* X = vmalloc(N * sizeof (float));
    float* Y = vmalloc(M * sizeof (float));

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
        pr_info("%d %d\n", (int)X[i], (int)Y[i]);
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
    new_sp -= ((unsigned long)new_sp % 16);

    pr_info("Current Stack Start d: %lu\n", (unsigned long)new_sp);

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
