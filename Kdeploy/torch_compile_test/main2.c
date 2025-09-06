#include <asm/bug.h>
#include <asm/fpu/api.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

extern void run_on_stack(void* new_stack, void (*f)(void));

MODULE_LICENSE("GPL");

extern void triton_poi_fused_relu_0(float* a, float* b, float* c, float* out, int numel, int x,
                                    int y, int z, int gridx, int gridy, int gridz);

extern void triton_poi_fused_relu_1(float* a, float* b, float* c, float* out, int numel, int x,
                                    int y, int z, int gridx, int gridy, int gridz);

extern void triton_poi_fused_addmm_2(float* a, float* b, float* c, float* out, int numel, int x,
                                     int y, int z, int gridx, int gridy, int gridz);

float* call(float** primals);
void work(void);

float* call(float** primals) {
  float* buf0 = (float*)kmalloc(2 * sizeof(float), GFP_KERNEL);
  float* buf1 = (float*)kmalloc(2 * sizeof(float), GFP_KERNEL);
  float* buf2;
  float* buf3;

  triton_poi_fused_relu_0(primals[2], primals[0], primals[1], buf0, 2, 0, 0, 0, 1, 1, 1);
  triton_poi_fused_relu_1(buf0, primals[3], primals[4], buf1, 2, 0, 0, 0, 1, 1, 1);
  buf2 = buf0;

  triton_poi_fused_relu_1(buf1, primals[5], primals[6], buf2, 2, 0, 0, 0, 1, 1, 1);
  buf3 = buf1;
  triton_poi_fused_addmm_2(buf2, primals[7], primals[8], buf3, 2, 0, 0, 0, 1, 1, 1);

  kfree(buf0);
  return buf3;
}

int stack_size = 8388608; // 8 MiB
module_param(stack_size, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(stack_size, "Size of the Kernel Module Stack (in bytes)");

void* kernel_sp;

void work(void) {
  pr_info("in work\n");
  float* primals[9];

  primals[0] = (float*)kmalloc(2 * 2 * sizeof(float), GFP_KERNEL);
  primals[1] = (float*)kmalloc(2 * 1 * sizeof(float), GFP_KERNEL);
  primals[2] = (float*)kmalloc(2 * 1 * sizeof(float), GFP_KERNEL);
  primals[3] = (float*)kmalloc(2 * 2 * sizeof(float), GFP_KERNEL);
  primals[4] = (float*)kmalloc(2 * 1 * sizeof(float), GFP_KERNEL);
  primals[5] = (float*)kmalloc(2 * 2 * sizeof(float), GFP_KERNEL);
  primals[6] = (float*)kmalloc(2 * 1 * sizeof(float), GFP_KERNEL);
  primals[7] = (float*)kmalloc(2 * 2 * sizeof(float), GFP_KERNEL);
  primals[8] = (float*)kmalloc(2 * 1 * sizeof(float), GFP_KERNEL);

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
  pr_info("%d/10000 %d/10000\n", (int)(out[0] * 10000), (int)(out[1] * 10000));
  kfree(out);
  for (int i = 0; i < 9; i++) {
    kfree(primals[i]);
  }
}

int init(void) {
  pr_info("Hello!\n");

  kernel_fpu_begin();
  work();
  kernel_fpu_end();

  /* A non 0 return means init_module failed; module can't be loaded. */
  return 0;
}

void cleanup(void) {
  pr_info("kernel module goodbye\n");
}

module_init(init);
module_exit(cleanup);
