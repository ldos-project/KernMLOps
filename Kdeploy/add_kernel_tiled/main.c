#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/vmalloc.h>
#include <linux/moduleparam.h>

extern void add_kernel_tiled(int* a, int* b, int* c, int len, int x, int y, int z, int gridX, int gridY, int gridZ);

MODULE_LICENSE("GPL");

int stack_size = 0;
module_param(stack_size, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(stack_size, "Size of the Kernel Module Stack (in bytes)");

void* kernel_sp;


#define N 2000 
void work(void) {
    int a[N];
    int b[N];
    int c[N];

    int i = 0;

    for (i = 0; i < N; i++) {
        a[i] = i;
        b[i] = N - i;
    }

    add_kernel_tiled(a, b, c, N, 0, 0, 0, 1, 1, 1);

    for (i = 0; i < N; i++) {
        pr_info("%d + %d = %d\n", a[i], b[i], c[i]);
    }
}

int init_module(void) 
{ 
    if (stack_size == 0) {
        pr_info("Failed to load module, no stack size provided");
        return -1;
    }

    char* new_stack = vmalloc(stack_size);
    char* new_sp = new_stack + stack_size;
    asm("mov %%rsp, %0" : "=r"(kernel_sp)); // save current stack pointer
    asm("mov %0, %%rsp" : : "r"(new_sp));   // switch stacks
                                            
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
