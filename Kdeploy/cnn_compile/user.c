#include <stdio.h>
#include <stdlib.h>

extern void triton_poi_fused_0(float* a, float* b, int c, int d, int x, int y, int z, int gridx,
                               int gridy, int gridz);

extern void triton_poi_fused_1(float* a, float* b, int c, int d, int x, int y, int z, int gridx,
                               int gridy, int gridz);

extern void triton_poi_fused_2(float* a, float* b, int c, int d, int x, int y, int z, int gridx,
                               int gridy, int gridz);

extern void triton_poi_fused_3(float* a, float* b, int c, int d, int x, int y, int z, int gridx,
                               int gridy, int gridz);

extern void triton_poi_fused_relu_4(float* a, int numel, int x, int y, int z, int gridx, int gridy,
                                    int gridz);

extern void triton_poi_fused_max_pool2d_with_indices_5(float* a, float* b, float* c, int numel,
                                                       int x, int y, int z, int gridx, int gridy,
                                                       int gridz);

extern void triton_poi_fused_relu_6(float* a, int numel, int x, int y, int z, int gridx, int gridy,
                                    int gridz);

extern void triton_poi_fused_max_pool2d_with_indices_7(float* a, float* b, float* c, int numel,
                                                       int x, int y, int z, int gridx, int gridy,
                                                       int gridz);

extern void triton_poi_fused_relu_8(float* a, int numel, int x, int y, int z, int gridx, int gridy,
                                    int gridz);

extern void triton_poi_fused_max_pool2d_with_indices_9(float* a, float* b, float* c, int numel,
                                                       int x, int y, int z, int gridx, int gridy,
                                                       int gridz);

extern void triton_poi_fused_relu_10(float* a, int numel, int x, int y, int z, int gridx, int gridy,
                                     int gridz);

float** allocate_primals();

float* call(float** primals) {
  float* buf0 = malloc(32 * 3 * 3 * 3 * sizeof(float));
  triton_poi_fused_0(primals[0], buf0, 96, 9, 0, 0, 0, 1, 1, 1);
  float* buf1 = malloc(1 * 3 * 64 * 64 * sizeof(float));
  triton_poi_fused_1(primals[2], buf1, 3, 4096, 0, 0, 0, 1, 1, 1);
  float* buf2 = malloc(64 * 32 * 3 * 3 * sizeof(float));
  triton_poi_fused_2(primals[3], buf2, 2048, 9, 0, 0, 0, 1, 1, 1);
  float* buf3 = malloc(128 * 64 * 3 * 3 * sizeof(float));
  triton_poi_fused_3(primals[5], buf3, 8192, 9, 0, 0, 0, 1, 1, 1);

  float* buf4 = extern_kernels.convolution(buf1, buf0, primals_2, stride = (1, 1), padding = (1, 1),
                                           dilation = (1, 1), transposed = False,
                                           output_padding = (0, 0), groups = 1);
  float* buf5 = buf4;
  triton_poi_fused_relu_4(buf5, 131072, 0, 0, 0, 1, 1, 1);
  float* buf6 = malloc(1 * 32 * 32 * 32 * sizeof(float));
  float* buf7 = malloc(1 * 32 * 32 * 32 * sizeof(char));
  triton_poi_fused_max_pool2d_with_indices_5(buf5, buf6, buf7, 32768, 0, 0, 0, 1, 1, 1);
  float* buf8 = extern_kernels.convolution(buf6, buf2, primals_5, stride = (1, 1), padding = (1, 1),
                                           dilation = (1, 1), transposed = False,
                                           output_padding = (0, 0), groups = 1);
  float* buf9 = buf8;
  triton_poi_fused_relu_6(buf9, 65536, 0, 0, 0, 1, 1, 1);
  float* buf10 = malloc(1 * 64 * 16 * 16 * sizeof(float));
  float* buf10 = malloc(1 * 64 * 16 * 16 * sizeof(char));
  triton_poi_fused_max_pool2d_with_indices_7(buf9, buf10, buf11, 16384, 0, 0, 0, 1, 1, 1);
  buf12 = extern_kernels.convolution(buf10, buf3, primals_7, stride = (1, 1), padding = (1, 1),
                                     dilation = (1, 1), transposed = False, output_padding = (0, 0),
                                     groups = 1);
  float* buf13 = buf12;
  triton_poi_fused_relu_8(buf13, 32768, 0, 0, 0, 1, 1, 1);
  float* buf14 = malloc(1 * 128 * 8 * 8 * sizeof(float));
  float* buf15 = malloc(1 * 128 * 8 * 8 * sizeof(float));
  triton_poi_fused_max_pool2d_with_indices_9(buf13, buf14, buf15, 64, 128, 0, 0, 0, 1, 1, 1);
  float* buf16 = malloc(1 * 128 * sizeof(float));
  extern_kernels.addmm(primals_9, reinterpret_tensor(buf15, (1, 8192), (0, 1), 0),
                       reinterpret_tensor(primals_8, (8192, 128), (1, 8192), 0), alpha = 1,
                       beta = 1, out = buf16);
  float* buf17 = buf16;
  triton_poi_fused_relu_10(buf17, 128, 0, 0, 0, 1, 1, 1);
  float* buf18 = malloc(1 * 10 * sizeof(float));
  extern_kernels.addmm(primals_11, buf17, reinterpret_tensor(primals_10, (128, 10), (1, 128), 0),
                       alpha = 1, beta = 1, out = buf18);
  return buf18;
}

int main(void) {
  /*
  const int stack_size = 8388608;
  char *new_stack = malloc(stack_size);
  char *new_sp = new_stack + stack_size;
  new_sp -= ((unsigned long)new_sp % 16);

  // work();
  run_on_stack(new_sp, work);

  free(new_stack);
  */
  float** primals = allocate_primals();
  float* out = call(primals);
  for (int i = 0; i < 10; i++) {
    printf("%.4f ", out[i]);
  }
  printf("\n");

  return 0;
}
