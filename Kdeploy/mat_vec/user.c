#include <stdio.h>
#include <stdlib.h>

extern void gemv_kernel(int* Y, int* A, int* X, int M, int N, int stride_am,
                        int x, int y, int z, int gridX, int gridY, int gridZ);

void work(void) {
    const int M = 128;
    const int N = 128;

    int* A = malloc(M * N * sizeof (int));
    int* X = malloc(N * sizeof (int));
    int* Y = malloc(M * sizeof (int));

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
        printf("%d %d\n", X[i], Y[i]);
    }

    free(A);
    free(X);
    free(Y);
}

int main(void) 
{ 
    work();

    return 0; 
}
