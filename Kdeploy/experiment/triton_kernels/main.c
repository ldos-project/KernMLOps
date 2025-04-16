#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/vmalloc.h>
#include <linux/moduleparam.h>
#include <asm/fpu/api.h>
#include <asm/bug.h>
#include <linux/ktime.h>

extern void gemv_add_relu_kernel(float* Y, float* A, float* X, float* B, int M, int N, int stride_am,
                        int x, int y, int z, int gridX, int gridY, int gridZ);

void launch_gemv_add_relu(float* Y, float* A, float* X, float* B, int M, int N, int stride_am);
const int BLOCK_SIZE = 8;

extern void run_on_stack(void* new_stack, void(*f)(void));
int align(int size, int block_size);


MODULE_LICENSE("GPL");

const int INPUT_SIZE = 12;
const int OUTPUT_SIZE = 2;
int nn_size = 8;
module_param(nn_size, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(nn_size, "Size of the Neural Net");

int stack_size = 8388608; // 8 MiB
void* kernel_sp;

void* alloc(int size);

void work(void) {
    kernel_fpu_begin();

    int aligned_in_size = align(INPUT_SIZE, BLOCK_SIZE);
    int aligned_out_size = align(OUTPUT_SIZE, BLOCK_SIZE);
    int aligned_nn_size = align(nn_size, BLOCK_SIZE);
    float* in = alloc(aligned_in_size * sizeof(float)); 
    float* l1 = alloc(aligned_nn_size * sizeof(float)); 
    float* l2 = alloc(aligned_nn_size * sizeof(float)); 
    float* l3 = alloc(aligned_nn_size * sizeof(float)); 
    float* out = alloc(aligned_out_size * sizeof(float)); 

    float* weights0 = alloc(aligned_in_size * aligned_nn_size * sizeof(float));
    float* weights1 = alloc(aligned_nn_size * aligned_nn_size * sizeof(float));
    float* weights2 = alloc(aligned_nn_size * aligned_nn_size * sizeof(float));
    float* weights3 = alloc(aligned_nn_size * aligned_out_size * sizeof(float));

    float* bias0 = alloc(aligned_nn_size * sizeof(float)); 
    float* bias1 = alloc(aligned_nn_size * sizeof(float)); 
    float* bias2 = alloc(aligned_nn_size * sizeof(float)); 
    float* bias3 = alloc(aligned_out_size * sizeof(float)); 

    pr_info("Beginning inference...");
    struct timespec64 t0,t1;
    ktime_get_real_ts64(&t0); 

    launch_gemv_add_relu(l1, weights0, in, bias0, nn_size, INPUT_SIZE, aligned_in_size);
    launch_gemv_add_relu(l2, weights1, l1, bias1, nn_size, nn_size, aligned_nn_size);
    launch_gemv_add_relu(l3, weights2, l2, bias2, nn_size, nn_size, aligned_nn_size);
    launch_gemv_add_relu(out, weights3, l3, bias3, OUTPUT_SIZE, nn_size, aligned_nn_size);

    /*
    launch_gemv_add_relu(l1, weights0, in, aligned_nn_size, aligned_in_size, aligned_in_size);
    pr_info("done 1");

    pr_info("l2: %px", l2);
    pr_info("w1: %px", weights1);
    pr_info("w1 size: %d", aligned_nn_size * aligned_nn_size * sizeof(float));
    pr_info("l1: %px", l1);
    pr_info("nn: %d", aligned_nn_size);
    launch_gemv_add_relu(l2, weights1, l1, aligned_nn_size, aligned_nn_size, aligned_nn_size);
    pr_info("done 2");
    launch_gemv_add_relu(l3, weights2, l2, aligned_nn_size, aligned_nn_size, aligned_nn_size);
    pr_info("done 3");

    pr_info("out: %px", out);
    pr_info("w3: %px", weights3);
    pr_info("w3 size: %d", aligned_nn_size * aligned_out_size * sizeof(float));
    pr_info("l3: %px", l3);
    pr_info("nn: %d", aligned_nn_size);
 
    launch_gemv_add_relu(out, weights3, l3, aligned_out_size, aligned_nn_size, aligned_out_size);
    pr_info("done 4");
    */

    ktime_get_real_ts64(&t1); 
    long long us = (t1.tv_sec - t0.tv_sec) * 1000000ULL + (t1.tv_nsec - t0.tv_nsec) / 1000;
    pr_info("%lld us", us);

    kernel_fpu_end();


    vfree(in);
    vfree(l1);
    vfree(l2);
    vfree(l3);
    vfree(out);

    vfree(weights0);
    vfree(weights1);
    vfree(weights2);
    vfree(weights3);

    vfree(bias0);
    vfree(bias1);
    vfree(bias2);
    vfree(bias3);
}

int init_module(void) 
{ 
    pr_info("Hello!\n");
    char* new_stack = alloc(stack_size);
    char* new_sp = new_stack + stack_size - 16;
    new_sp -= ((unsigned long)new_sp % 16);

    pr_info("Current Stack Start d: %lu\n", (unsigned long)new_sp);

    pr_info("Current Stack Start: %px\n", new_sp);
    pr_info("Current Stack End: %px\n", new_stack);

    run_on_stack(new_sp, work);

    vfree(new_stack);

    /* A non 0 return means init_module failed; module can't be loaded. */ 
    return 0; 
}

void cleanup_module(void) 
{
    pr_info("kernel module goodbye\n");
} 

void launch_gemv_add_relu(float* Y, float* A, float* X, float* B, int M, int N, int stride_am) {
    int grid = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int i;

    for (i = 0; i < grid; i++) {
	gemv_add_relu_kernel(Y, A, X, B, M, N, stride_am,
			 i, 0, 0, grid, 1, 1);
    }
}

int align(int size, int block_size) {
    return ((size + block_size - 1) / block_size) * block_size;
}

void* alloc(int size) {
    size += BLOCK_SIZE;
    float* out = vmalloc(size);
    int i;
    for (i = 0; i < (size / sizeof (float)); i++) {
        out[i] = (i - 1.0) / i;
    }
    return out;
}
