#include <stdio.h>
#include <stdlib.h>

extern void gemv_kernel(float* Y, float* A, float* X, int M, int N, int stride_am,
                        int x, int y, int z, int gridX, int gridY, int gridZ);

extern void run_on_stack(void* new_stack, void(*f)(void));

void work(void) {
    const int M = 128;
    const int N = 128;

    float* A = malloc(M * N * sizeof (float));
    float* X = malloc(N * sizeof (float));
    float* Y = malloc(M * sizeof (float));

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
        printf("%d %d\n", (int)X[i], (int)Y[i]);
    }

    free(A);
    free(X);
    free(Y);
}

int main(void) 
{ 
    const int stack_size = 8388608;
    char* new_stack = malloc(stack_size);
    char* new_sp = new_stack + stack_size;
    new_sp -= ((unsigned long)new_sp % 16);

    //work();
    run_on_stack(new_sp, work);

    free(new_stack);


    return 0; 
}
