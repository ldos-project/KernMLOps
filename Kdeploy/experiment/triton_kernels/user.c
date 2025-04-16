
#include <stdio.h>
#include <stdlib.h>

extern void gemv_relu_kernel(float* Y, float* A, float* X, int M, int N, int stride_am,
                        int x, int y, int z, int gridX, int gridY, int gridZ);

void launch_gemv_relu(float* Y, float* A, float* X, int M, int N, int stride_am);
const int BLOCK_SIZE = 16;

extern void run_on_stack(void* new_stack, void(*f)(void));
int align(int size, int block_size);


const int INPUT_SIZE = 12;
const int OUTPUT_SIZE = 2;
int nn_size = 4096;

int stack_size = 8388608; // 8 MiB
void* kernel_sp;

void work(void) {
    int aligned_in_size = align(INPUT_SIZE, BLOCK_SIZE);
    int aligned_out_size = align(OUTPUT_SIZE, BLOCK_SIZE);
    int aligned_nn_size = align(nn_size, BLOCK_SIZE);
    float* in = malloc(aligned_in_size * sizeof(float)); 
    float* l1 = malloc(aligned_nn_size * sizeof(float)); 
    float* l2 = malloc(aligned_nn_size * sizeof(float)); 
    float* l3 = malloc(aligned_nn_size * sizeof(float)); 
    float* out = malloc(aligned_out_size * sizeof(float)); 

    float* weights0 = malloc(aligned_in_size * aligned_nn_size * sizeof(float));
    float* weights1 = malloc(aligned_nn_size * aligned_nn_size * sizeof(float));
    float* weights2 = malloc(aligned_nn_size * aligned_nn_size * sizeof(float));
    float* weights3 = malloc(aligned_nn_size * aligned_out_size * sizeof(float));

    float* bias0 = malloc(aligned_nn_size * sizeof(float)); 
    float* bias1 = malloc(aligned_nn_size * sizeof(float)); 
    float* bias2 = malloc(aligned_nn_size * sizeof(float)); 
    float* bias3 = malloc(aligned_out_size * sizeof(float)); 

    printf("Beginning inference...\n");

    launch_gemv_relu(l1, weights0, in, nn_size, INPUT_SIZE, aligned_in_size);

    printf("1\n");
    printf("l2: %p\n", l2);
    printf("l1: %p\n", l1);
    printf("w1: %p\n", weights1);

    launch_gemv_relu(l2, weights1, l1, nn_size, nn_size, aligned_nn_size);

    printf("2\n");
    printf("l3: %p\n", l3);
    printf("l2: %p\n", l2);
    printf("w2: %p\n", weights2);


    launch_gemv_relu(l3, weights2, l2, nn_size, nn_size, aligned_nn_size);
    launch_gemv_relu(out, weights3, l3, OUTPUT_SIZE, nn_size, aligned_nn_size);

    printf("Done inference...\n");

    /*
    launch_gemv_relu(l1, weights0, in, aligned_nn_size, aligned_in_size, aligned_in_size);
    launch_gemv_relu(l2, weights1, l1, aligned_nn_size, aligned_nn_size, aligned_nn_size);
    launch_gemv_relu(l3, weights2, l2, aligned_nn_size, aligned_nn_size, aligned_nn_size);
    launch_gemv_relu(out, weights3, l3, aligned_out_size, aligned_nn_size, aligned_nn_size);
    */


    free(in);
    free(l1);
    free(l2);
    free(l3);
    free(out);
    
    printf("some free...\n");

    free(weights0);
    free(weights1);
    free(weights2);
    free(weights3);

    printf("weight free...\n");

    free(bias0);
    free(bias1);
    free(bias2);
    free(bias3);

    printf("bias free...\n");
}

int main(void) 
{ 
    printf("Hello!\n");
    char* new_stack = malloc(stack_size);
    char* new_sp = new_stack + stack_size - 16;
    new_sp -= ((unsigned long)new_sp % 16) + 8;

    printf("Current Stack Start d: %lu\n", (unsigned long)new_sp);

    printf("Current Stack Start: %p\n", new_sp);
    printf("Current Stack End: %p\n", new_stack);

    //run_on_stack(new_sp, work);
    work();

    printf("out of stack\n");
    printf("Current Stack End: %p\n", new_stack);

    free(new_stack);

    /* A non 0 return means init_module failed; module can't be loaded. */ 
    return 0; 
}

void launch_gemv_relu(float* Y, float* A, float* X, int M, int N, int stride_am) {
    int grid = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int i;
    for (i = 0; i < grid; i++) {
	if (i == 114) {
	    printf("hi\n");
	}

	gemv_relu_kernel(Y, A, X, M, N, stride_am,
			 i, 0, 0, grid, 1, 1);
    }
}

int align(int size, int block_size) {
    return ((size + block_size - 1) / block_size) * block_size;
}
