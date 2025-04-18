#include "../../fstore/fstore.h"
#include "bloat.h"
#include <asm/fpu/api.h>
#include <linux/delay.h>
#include <linux/mm.h>
#include <linux/module.h> /* Needed by all modules */
#include <linux/printk.h> /* Needed for pr_info() */
#include <linux/timex.h>

#define MAX_LAYER_SIZE 16
#define INPUT_SIZE     20
#define OUTPUT_SIZE    2

void sort(void* base, size_t num, size_t size, cmp_func_t cmp_func, swap_func_t swap_func);

static const float weights0[] = {
    0.017325,  -0.102724, -0.076493, -0.103416, 0.142636,  -0.235354, -0.065574, 0.003554,
    -0.051728, -0.074628, 0.069469,  0.119377,  -0.052321, 0.052125,  0.095846,  -0.045546,
    -0.059484, -0.125362, -0.095823, 0.113942,  0.170176,  0.071576,  -0.063629, 0.007492,
    -0.081567, -0.248377, 0.215028,  0.026375,  0.203714,  0.009150,  -0.051810, 0.104339,
    -0.027414, -0.104355, 0.221644,  -0.276541, 0.176998,  -0.140427, 0.002099,  -0.183965,
    -0.080983, -0.109257, 0.132868,  -0.044274, -0.057402, -0.051081, 0.183269,  0.096902,
    0.203611,  0.221048,  -0.195776, -0.147638, -0.196374, 0.058230,  0.034689,  0.194912,
    0.216228,  -0.050254, 0.233330,  0.015611,  -0.154697, 0.164024,  -0.029023, -0.001119,
    -0.135541, 0.095664,  0.032383,  0.202640,  0.072061,  0.070040,  -0.020493, 0.251334,
    -0.003692, 0.117550,  0.063758,  -0.152416, 0.114160,  0.025027,  -0.002080, 0.166572,
    -0.149022, -0.011197, -0.150522, 0.083043,  0.037524,  -0.226303, -0.112720, 0.082830,
    -0.153484, 0.153823,  0.231671,  -0.019949, -0.102960, 0.001183,  -0.148051, -0.037754,
    0.049897,  -0.151921, 0.137840,  0.029518,  -0.120744, 0.097058,  -0.198699, 0.220296,
    0.095354,  0.007681,  -0.022395, 0.161488,  0.146623,  0.089465,  -0.171654, -0.142592,
    -0.059466, -0.004346, -0.140429, 0.150123,  -0.082976, -0.213062, 0.137893,  0.000128,
    -0.095390, -0.169232, -0.059250, -0.018229, -0.056491, 0.128476,  0.132140,  -0.197357,
    -0.015069, -0.051106, 0.107817,  -0.203441, 0.144158,  -0.036114, -0.198031, 0.118987,
    -0.100851, 0.149531,  -0.215394, -0.083902, 0.177955,  0.126650,  -0.199499, -0.117919,
    -0.061364, 0.146142,  -0.103262, -0.048126, 0.191299,  -0.178151, 0.018307,  -0.040574,
    0.152989,  0.204866,  -0.118217, 0.167389,  -0.081665, -0.163189, -0.181980, -0.179768,
    0.103266,  0.151930,  -0.007809, -0.090062, 0.229411,  -0.222419, -0.159179, -0.059077,
    0.234066,  -0.229914, 0.007487,  0.072391,  -0.047921, 0.081953,  0.213922,  -0.119830,
    0.003960,  -0.152037, 0.038695,  -0.135718, 0.178728,  0.206901,  -0.084066, 0.082162,
    -0.189501, -0.029745, 0.038857,  -0.199305, -0.186916, 0.057414,  -0.006124, 0.019704,
    -0.135509, -0.199854, 0.026849,  0.028416,  0.027837,  -0.106240, 0.196848,  -0.068449,
    -0.053273, -0.016166, -0.047121, 0.124150,  0.156286,  0.199528,  -0.151590, -0.084502,
    -0.196979, 0.090990,  0.053486,  0.324889,  0.048343,  0.346585,  -0.055407, 0.286789,
    0.039010,  0.038935,  -0.064017, -0.008765, -0.157123, -0.029921, 0.045248,  0.049677,
    0.076329,  0.122900,  -0.069966, -0.051365, 0.122078,  -0.050632, -0.114028, 0.068309,
    -0.004509, 0.079061,  -0.098978, 0.029059,  -0.114549, -0.173451, -0.021340, -0.068924,
    -0.270386, 0.304900,  0.111735,  0.045502,  -0.092129, 0.257978,  0.053905,  -0.095624,
    -0.057199, 0.152048,  0.099010,  0.073063,  0.107083,  0.239143,  -0.094318, -0.052115,
    -0.293936, 0.294086,  -0.223895, 0.306932,  0.052599,  -0.036166, -0.078489, 0.153006,
    -0.128027, -0.205476, -0.063026, -0.113722, -0.182814, 0.180879,  -0.022653, -0.007706,
    0.133821,  -0.031178, -0.022177, -0.236991, 0.154846,  -0.128677, 0.176170,  -0.100327,
    0.107721,  -0.009164, 0.219649,  -0.235176, 0.136191,  0.085705,  0.167790,  0.016841,
    -0.076055, -0.237041, -0.178977, -0.015074, 0.135615,  -0.068872, 0.003842,  -0.223231,
    -0.062013, -0.105094, 0.164920,  0.007180,  0.091236,  -0.134489, 0.227723,  -0.082237,
    0.265106,  -0.113050, 0.072470,  0.148267,  -0.025969, -0.219409, -0.132832, -0.165648,
    -0.004911, 0.012397,  0.071075,  0.147492,  0.106987,  -0.119764, -0.130215, 0.005708};

static const float bias0[] = {0.047214,  -0.082733, -0.371399, -0.242038, 0.391561,  -0.024686,
                              0.240204,  -0.127172, 0.013863,  -0.003516, -0.086109, 0.358476,
                              -0.150033, -0.049151, 0.052743,  -0.041167};

static const float weights1[] = {
    -0.197524, 0.185109,  0.350741, 0.194102,  -0.485247, -0.044340, -0.334681, -0.083691,
    0.149598,  -0.151096, 0.127873, -0.701373, 0.079485,  -0.145900, 0.115464,  0.237253,
    0.146962,  -0.177476, 0.131313, 0.258514,  -0.345678, 0.154409,  -0.467941, 0.242716,
    -0.185698, -0.161476, 0.159626, -0.386010, 0.317017,  0.246859,  -0.189613, -0.194924};

static const float bias1[] = {0.578542, 0.408592};

static void forward(const float* input, float* output, float* temp1, float* temp2) {
  float* temp;
  float *curr = temp1, *next = temp2;

  // Layer 0
  for (int i = 0; i < 16; i++) {
    float sum = 0.0f;
    for (int j = 0; j < 20; j++) {
      sum += input[j] * weights0[i * 20 + j];
    }
    curr[i] = sum + bias0[i];
    curr[i] = curr[i] > 0.0f ? curr[i] : 0.0f;
  }

  // Output Layer
  for (int i = 0; i < 2; i++) {
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
      sum += curr[j] * weights1[i * 16 + j];
    }
    output[i] = sum + bias1[i];
  }
}

// Make a cache for vmfault
u32* target_tgid;
static int is_blocked = 0;

static void block_cache() {
  if (*target_tgid != 0)
    is_blocked = 1;
}

static void unblock_cache() {
  if (target_tgid != 0)
    is_blocked = 0;
}

struct data {
  __u64 ts;
  __u64 count;
  __u64 valid;
} data_t;

u64* rss_head;
u64* tlb_head;
data_t* rss_data;
data_t* tlb_data;

struct to_sort {
  __u64 ts;
  __u64 count;
} sortable_t;

void read_data(data_t* data_buffer, u64* head_ptr, sortable_t* buffer, int len) {
  __u64 start = READ_ONCE(head_ptr);
  __u64 trials = BUFFER_SIZE >> 1;
  for (int c = len - 1; c >= 0 && trials >= 0; trials--) {
    start = (start - 1) % BUFFER_ENTRIES;
    data_t* point = data_buffer + start;
    cmpxchg_acquire(&point->valid, 1, 2);
    buffer[c]->count = point->count;
    buffer[c]->ts = point->ts;
    if (cmpxchg_release(&point->valid, 2, 1) == 1) {
      c--;
    }
  }
}

int sort_data(const void* a, const void* b) {
  sortable_t sa = *(const sortable_t*)a;
  sortable_t sb = *(const sortable_t*)b;
  return sa.ts - sb.ts;
}

void get_data_into_float(float* input) {
  sortable_t rss[10];
  sortable_t dtlb[10];
  read_data(rss_data, rss_head, rss, 10);
  read_data(dtlb_data, dtlb_head, dtlb, 10);
  sort(rss, 10, sizeof(sortable_t), sort_data);
  sort(dtlb_data, 10, sizeof(sortable_t), sort_data);
  for (i = 0; i < 10; i++) {
    input[i * 2] = (float)(rss[i]);
    input[i * 2 + 1] = (float)(dtlb[i]);
  }
}

int infer(struct mm_struct* mm, int unmapped, int referenced) {
  if (mm->owner->tgid != *target_tgid) {
    return unmapped && (!referenced || referenced < HPAGE_PMD_NR / 2);
  }
  pr_info("Found you");
  kernel_fpu_begin();
  float input[20];
  float temp1[MAX_LAYER_SIZE];
  float temp2[MAX_LAYER_SIZE];
  float output[2];
  get_data_into_float(input);
  forward(input, output, temp1, temp2);
  // First output is now, second output is 5 timesteps forwards
  if (output[1] >= 0.8)
    block_cache();
  int cmp = (output[0] >= 0.8) || (output[1] >= 0.8);
  kernel_fpu_end();
  if (cmp)
    return unmapped && (!reference || referenced < HPAGE_PMD_NR - 2) return unmapped &&
           (!referenced || referenced < HPAGE_PMD_NR / 2);
}

int (*ml_throttle_hugepage_faults)(struct vm_fault* vmf);

int (*ml_referenced_page_limit_collapse)(struct mm_struct* mm, int unmapped, int referenced);

static int should_throttle_fault(struct vm_fault* vmf) {
  if (target_tgid == -1)
    return 0;
  if (vmf->vma && vmf->vma->vm_mm && vmf->vma->vm_mm->owner &&
      vmf->vma->vm_mm->owner->tgid == target_tgid && is_blocked) {
    return 1;
  }
  return 0;
}

__u64 map_id[5]

    // Example usage function
    int __init
    init_module(void) {
  pr_info("Kernel Module Init \n");
  // get the cache
  if (convert8byteStringHash(CACHE, &map_id[0]))
    return -EINVAL;
  int err = fstore_get_map_array_start(map_id, 4, 4, 1, (void**)target_tgid);
  if (err != 0)
    return err;

  if (convert8byteStringHash(RHEAD, &map_id[1]))
    return -EINVAL;
  int err = fstore_get_map_array_start(map_id, 4, 8, 1, (void**)rss_head);
  if (err != 0)
    return err;

  if (convert8byteStringHash(RDATA, &map_id[2]))
    return -EINVAL;
  int err = fstore_get_map_array_start(map_id, 4, 24, BUFFER_ENTRIES, (void**)rss_data);
  if (err != 0)
    return err;

  if (convert8byteStringHash(THEAD, &map_id[3]))
    return -EINVAL;
  int err = fstore_get_map_array_start(map_id, 4, 8, 1, (void**)tlb_head);
  if (err != 0)
    return err;

  if (convert8byteStringHash(TDATA, &map_id[4]))
    return -EINVAL;
  int err = fstore_get_map_array_start(map_id, 4, 24, BUFFER_ENTRIES, (void**)tlb_data);
  if (err != 0)
    return err;

  ml_referenced_page_limit_collapse = infer;
  ml_throttle_hugepage_faults = should_throttle_fault;
  return 0;
}

void __exit cleanup_module(void) {
  pr_info("Goodbye \n");
  ml_referenced_page_limit_collapse = NULL;
  ml_throttle_hugepage_faults = NULL;
  for (int i = 0; i < 5; i++) {
    fstore_put_map_array(map_id[i]);
  }
}

MODULE_LICENSE("GPL");
