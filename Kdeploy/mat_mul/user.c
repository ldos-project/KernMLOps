#include <stdio.h>
#include <stdlib.h>

extern void matmul_kernel(
    float* A, float* B, float* C, 
    int M, int N, int K, 

    int stride_am, int stride_ak,
    int stride_bk, int stride_bn,
    int stride_cm, int stride_cn,

    int x, int y, int z, 
    int gridX, int gridY, int gridZ
);
                        


void work(void) {
    const int M = 16;
    const int N = 16;
    const int K = 16;

    float* A = malloc(M * K * sizeof (float));
    float* B = malloc(K * N * sizeof (float));
    float* C = malloc(M * N * sizeof (float));

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
            printf("%.0f ", C[i * N + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
}

int main(void) 
{ 
    work();

    return 0; 
}
