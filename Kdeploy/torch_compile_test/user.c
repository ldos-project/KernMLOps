#include <stdio.h>
#include <stdlib.h>

extern void triton_poi_fused_relu_0(float* a, float* b, float* c, float* out, int numel, int x,
                                    int y, int z, int gridx, int gridy, int gridz);

extern void triton_poi_fused_relu_1(float* a, float* b, float* c, float* out, int numel, int x,
                                    int y, int z, int gridx, int gridy, int gridz);

extern void triton_poi_fused_addmm_2(float* a, float* b, float* c, float* out, int numel, int x,
                                     int y, int z, int gridx, int gridy, int gridz);

float* call(float** primals) {
  float* buf0 = malloc(2 * sizeof(float));
  float* buf1 = malloc(2 * sizeof(float));
  float* buf2;
  float* buf3;

  triton_poi_fused_relu_0(primals[2], primals[0], primals[1], buf0, 2, 0, 0, 0, 1, 1, 1);
  triton_poi_fused_relu_1(buf0, primals[3], primals[4], buf1, 2, 0, 0, 0, 1, 1, 1);
  buf2 = buf0;

  triton_poi_fused_relu_1(buf1, primals[5], primals[6], buf2, 2, 0, 0, 0, 1, 1, 1);
  buf3 = buf1;
  triton_poi_fused_addmm_2(buf2, primals[7], primals[8], buf3, 2, 0, 0, 0, 1, 1, 1);

  free(buf0);
  return buf3;
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

  float* primals[9];
  primals[0] = malloc(2 * 2 * sizeof(float));
  primals[1] = malloc(2 * 1 * sizeof(float));
  primals[2] = malloc(2 * 1 * sizeof(float));
  primals[3] = malloc(2 * 2 * sizeof(float));
  primals[4] = malloc(2 * 1 * sizeof(float));
  primals[5] = malloc(2 * 2 * sizeof(float));
  primals[6] = malloc(2 * 1 * sizeof(float));
  primals[7] = malloc(2 * 2 * sizeof(float));
  primals[8] = malloc(2 * 1 * sizeof(float));

  /*
  [Parameter containing:
  tensor([[ 0.1505,  0.4821],
          [ 0.3405, -0.0710]], requires_grad=True), Parameter containing:
  tensor([0.0394, 0.5562], requires_grad=True), tensor([1., 1.]), Parameter
  containing: tensor([[-0.6640,  0.6090], [ 0.1042, -0.4600]],
  requires_grad=True), Parameter containing: tensor([ 0.5770, -0.5517],
  requires_grad=True), Parameter containing: tensor([[-0.4564, -0.6216], [
  0.3483, -0.4782]], requires_grad=True), Parameter containing: tensor([-0.0818,
  -0.3487], requires_grad=True), Parameter containing: tensor([[-0.5258,
  -0.4167], [ 0.4412,  0.2115]], requires_grad=True), Parameter containing:
  tensor([0.5190, 0.5110], requires_grad=True)]

  out:
  tensor([0.5190, 0.5110], grad_fn=<CompiledFunctionBackward>)
  */
  primals[0][0] = -0.2609;
  primals[0][1] = 0.3745;
  primals[0][2] = -0.0779;
  primals[0][3] = 0.4720;
  primals[1][0] = -0.0760;
  primals[1][1] = 0.2335;
  primals[2][0] = 1.0000;
  primals[2][1] = 1.0000;
  primals[3][0] = -0.5081;
  primals[3][1] = -0.0108;
  primals[3][2] = -0.4129;
  primals[3][3] = 0.3129;
  primals[4][0] = 0.2924;
  primals[4][1] = -0.0366;
  primals[5][0] = 0.2120;
  primals[5][1] = 0.2538;
  primals[5][2] = -0.4555;
  primals[5][3] = -0.4630;
  primals[6][0] = 0.5233;
  primals[6][1] = -0.3107;
  primals[7][0] = 0.5437;
  primals[7][1] = 0.6694;
  primals[7][2] = 0.0324;
  primals[7][3] = -0.2079;
  primals[8][0] = -0.1707;
  primals[8][1] = 0.0361;

  float* out = call(primals);
  printf("%.4f %.4f\n", out[0], out[1]);

  return 0;
}
