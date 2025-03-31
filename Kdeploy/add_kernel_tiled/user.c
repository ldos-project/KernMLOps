#include <stdio.h>

extern void add_kernel_tiled(int* a, int* b, int* c, int len, int x, int y, int z, int gridX, int gridY, int gridZ);

#define N 100 

int main(void) 
{ 
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

    /* A non 0 return means init_module failed; module can't be loaded. */ 
    return 0; 
}
