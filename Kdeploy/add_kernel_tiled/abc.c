
#include <stdio.h>
#include <stdlib.h>

extern void add_kernel_tiled(int* a, int* b, int* c, int len, int x, int y, int z, int gridX, int gridY, int gridZ);

int stack_size = 10000;
void* kernel_sp;

#define N 10 

void work() {
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
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
}



int main(void) 
{ 
    char* new_stack = malloc(stack_size);
    char* new_sp = new_stack + stack_size;
    printf("New SP: %p\n", new_sp);
    asm("mov %%rsp, %0" : "=r"(kernel_sp)); // save current stack pointer
    printf("Old SP: %p\n", kernel_sp);
    asm("mov %0, %%rsp" : : "r"(new_sp));   // switch stacks 
                                            
    work();
    
    asm("mov %0, %%rsp" : : "r"(kernel_sp)); // restore stack pointer
    printf("Restored SP: %p\n", kernel_sp);
    free(new_stack);

    /* A non 0 return means init_module failed; module can't be loaded. */ 
    return 0; 
}

