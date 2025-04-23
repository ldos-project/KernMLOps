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

int fstore_get_map_array_start(u64 map_name, size_t key_size, size_t value_size,
                               size_t num_elements, void** map_ptr);

int fstore_put_map_array(u64 map_name);

static const float weights0[] = {
    0.178033,  0.038148,  0.099444,  0.036364,  -0.160153, -0.075194, -0.126246, -0.226075,
    0.153429,  0.133899,  0.150711,  0.048391,  0.021579,  0.097926,  -0.070673, 0.018658,
    -0.198925, -0.137760, -0.182743, -0.195182, 0.334344,  -0.017734, 0.271336,  -0.190007,
    0.030002,  -0.163793, 0.259454,  -0.120839, 0.002649,  -0.000251, 0.106311,  -0.160719,
    0.184267,  -0.119682, 0.332457,  -0.077655, -0.055606, -0.148486, 0.232296,  -0.054855,
    0.211621,  -0.081351, -0.150886, -0.196385, 0.091806,  -0.034066, 0.123060,  -0.077836,
    0.003997,  -0.162068, -0.119170, -0.064253, -0.010321, 0.027278,  -0.158671, 0.198628,
    -0.187937, -0.118153, -0.089465, 0.053907,  0.323950,  -0.314762, 0.283536,  -0.067208,
    0.030712,  -0.144154, 0.072557,  -0.287397, 0.344002,  0.005217,  0.336285,  -0.101161,
    0.042839,  -0.130990, 0.332344,  -0.431950, 0.363924,  -0.082129, -0.018558, -0.036571,
    0.457233,  -0.738167, 0.419327,  -0.942683, 0.512830,  -0.937987, 0.139067,  -0.913436,
    0.265727,  -0.801001, 0.129399,  -0.976539, 0.347130,  -0.779699, 0.175187,  -0.924818,
    0.227195,  -0.538436, 0.457084,  -0.711639, 0.139559,  0.267731,  0.006681,  0.029993,
    -0.077023, 0.381986,  -0.170493, 0.362476,  -0.229690, 0.302425,  -0.172531, 0.319241,
    0.054444,  0.000165,  0.098337,  0.366388,  -0.278352, 0.184557,  -0.002039, 0.351657,
    -0.157690, -0.046121, -0.109871, -0.130623, -0.075321, -0.202225, 0.001697,  0.230375,
    0.101444,  -0.036116, -0.219901, 0.062651,  -0.214830, -0.125824, -0.129481, 0.056328,
    -0.010594, -0.127939, 0.091536,  0.134021,  -0.181193, -0.087087, -0.063471, -0.025273,
    -0.060876, 0.143099,  -0.257194, 0.026051,  -0.014171, -0.045064, -0.170943, -0.214569,
    0.118643,  0.001298,  -0.263814, -0.243686, 0.077255,  0.149358,  -0.075657, 0.117012,
    0.022273,  0.277675,  -0.207790, 0.363912,  0.068607,  0.407131,  -0.015478, 0.131992,
    -0.100504, 0.418791,  0.090103,  0.137245,  0.120426,  0.477812,  -0.251830, 0.257998,
    -0.212033, 0.188537,  -0.184474, 0.166584,  -0.003142, 0.290613,  -0.135700, 0.415466,
    0.098144,  0.261629,  0.140273,  0.415649,  -0.009028, 0.248123,  0.023865,  0.333535,
    -0.189503, 0.089323,  -0.249279, 0.269664,  -0.092108, 0.000055,  -0.214359, 0.344458,
    -0.154336, -0.276464, -0.247086, -0.147176, 0.042336,  -0.105201, -0.261022, 0.138924,
    -0.186150, 0.032197,  0.149026,  -0.100968, -0.094057, -0.070239, 0.026418,  0.205526,
    0.020930,  0.141361,  -0.157635, 0.119028,  0.319441,  0.019742,  0.327302,  -0.104659,
    0.025228,  -0.091795, 0.295893,  -0.008358, 0.132057,  -0.340027, -0.032008, -0.289136,
    0.242306,  -0.258571, 0.262166,  -0.166733, 0.097093,  -0.031031, 0.312557,  -0.144707,
    -0.164918, 0.376457,  0.012830,  0.181835,  0.046803,  0.378769,  -0.194202, 0.420031,
    -0.113695, 0.241684,  -0.247357, 0.140201,  -0.050821, 0.169001,  -0.251842, 0.039851,
    -0.055710, 0.220574,  -0.101786, 0.221622,  0.143994,  0.389004,  -0.111574, 0.343650,
    0.019503,  0.403121,  -0.164048, 0.032883,  -0.171645, 0.068576,  -0.190360, 0.237394,
    -0.231022, 0.162106,  0.144215,  0.299709,  0.199992,  0.229089,  -0.035012, 0.206093,
    -0.258080, 0.067270,  0.059015,  0.077675,  -0.240619, -0.151476, -0.242577, -0.096093,
    -0.149992, 0.076261,  -0.013509, -0.022812, -0.251647, -0.176574, 0.088124,  0.032340,
    0.059224,  -0.124037, 0.097668,  0.088834,  0.163887,  0.055991,  0.066231,  -0.032226,
    0.261828,  -0.047049, 0.322636,  -0.285392, 0.148589,  -0.187672, 0.189803,  -0.397100,
    0.051610,  -0.101658, 0.124104,  -0.017289, 0.225205,  -0.151052, -0.079548, 0.009200};

static const float bias0[] = {-0.200041, -0.626553, -0.188323, -0.528633, 0.191974, -0.376013,
                              0.816932,  0.971796,  -0.405513, -0.385715, 0.892988, -0.611413,
                              -0.287291, -0.456491, 0.987371,  -0.376859};

static const float weights1[] = {
    -0.195442, 0.506364, 0.125460,  0.659023, 2.103589, 0.558912, -0.982338, -1.238313,
    0.305176,  0.541860, -1.224363, 0.474777, 0.576836, 0.365266, -1.070154, 0.320466,
    0.101335,  0.417710, -0.186058, 0.498497, 2.153835, 0.392865, -0.883285, -1.129130,
    0.429418,  0.390499, -1.418123, 0.506423, 0.665251, 0.399101, -1.410942, 0.240761};

static const float bias1[] = {-0.570929, -0.354454};

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
static int is_blocked;

static void block_page_fault_special(void) {
  if (*target_tgid != 0)
    is_blocked = 1;
}

static void unblock_page_fault_special(void) {
  if (target_tgid != 0)
    is_blocked = 0;
}

typedef struct data {
  __u64 ts;
  __u64 count;
  __u64 valid;
} data_t;

u64* rss_head;
u64* dtlb_head;
data_t* rss_data;
data_t* dtlb_data;

typedef struct to_sort {
  __u64 ts;
  __u64 count;
} sortable_t;

static void read_data(data_t* data_buffer, u64* head_ptr, sortable_t* buffer, int len) {
  __u64 start = READ_ONCE(*head_ptr);
  __u64 trials = BUFFER_ENTRIES >> 1;
  for (int c = len - 1; c >= 0 && trials >= 0; trials--) {
    start = (start - 1) % BUFFER_ENTRIES;
    data_t* point = data_buffer + start;
    cmpxchg_acquire(&point->valid, 1, 2);
    buffer[c].count = point->count;
    buffer[c].ts = point->ts;
    if (cmpxchg_release(&point->valid, 2, 1) == 1) {
      c--;
    }
  }
}

static int sort_data(const void* a, const void* b) {
  sortable_t sa = *(const sortable_t*)a;
  sortable_t sb = *(const sortable_t*)b;
  return sa.ts - sb.ts;
}

static void get_data_into_float(float* input) {
  sortable_t rss[10];
  sortable_t dtlb[10];
  read_data(rss_data, rss_head, rss, 10);
  read_data(dtlb_data, dtlb_head, dtlb, 10);
  sort(rss, 10, sizeof(sortable_t), sort_data, NULL);
  sort(dtlb_data, 10, sizeof(sortable_t), sort_data, NULL);
  for (int i = 0; i < 10; i++) {
    input[i * 2] = (float)(rss[i].count);
    input[i * 2 + 1] = (float)(dtlb[i].count);
  }
}

static int infer(struct mm_struct* mm, int unmapped, int referenced) {
  pr_info("Got this tgid %lu\n", mm->owner->tgid);
  if (mm->owner->tgid != *target_tgid) {
    return unmapped && (!referenced || referenced < HPAGE_PMD_NR / 2);
  }
  pr_info("Found you\n");
  float input[20];
  float temp1[MAX_LAYER_SIZE];
  float temp2[MAX_LAYER_SIZE];
  float output[2];
  get_data_into_float(input);
  kernel_fpu_begin();
  forward(input, output, temp1, temp2);
  // First output is now, second output is 5 timesteps forwards
  if (output[1] >= 0.8)
    block_page_fault_special();
  int cmp = (output[0] >= 0.8) || (output[1] >= 0.8);
  if (output[1] < 0.5 && output[2] < 0.5)
    unblock_page_fault_special();
  kernel_fpu_end();
  if (cmp)
    return unmapped && (!referenced || referenced < HPAGE_PMD_NR - 2);
  return unmapped && (!referenced || referenced < HPAGE_PMD_NR / 2);
}

int (*ml_throttle_hugepage_faults)(struct vm_fault* vmf);

int (*ml_referenced_page_limit_collapse)(struct mm_struct* mm, int unmapped, int referenced);

static int should_throttle_fault(struct vm_fault* vmf) {
  pr_info("Throttling\n");
  if (*target_tgid == 0)
    return 0;
  if (vmf->vma && vmf->vma->vm_mm && vmf->vma->vm_mm->owner &&
      vmf->vma->vm_mm->owner->tgid == *target_tgid && is_blocked) {
    return 1;
  }
  return 0;
}

__u64 map_id[5];

// Example usage function
int __init init_module(void) {
  pr_info("Kernel Module Init \n");
  // get the cache
  if (convert8byteStringHash(CACHE, &map_id[0]))
    return -EINVAL;
  int err = fstore_get_map_array_start(map_id[0], 4, 4, 1, (void**)&target_tgid);
  if (err != 0) {
    pr_info("Cache error\n");
    return err;
  }

  pr_info("The pid we are targeting is %ld\n", *target_tgid);

  if (convert8byteStringHash(RHEAD, &map_id[1]))
    return -EINVAL;
  err = fstore_get_map_array_start(map_id[1], 4, 8, 1, (void**)&rss_head);
  if (err != 0) {
    pr_info("RHEAD error\n");
    return err;
  }

  if (convert8byteStringHash(RDATA, &map_id[2]))
    return -EINVAL;
  err = fstore_get_map_array_start(map_id[2], 4, 24, BUFFER_ENTRIES, (void**)&rss_data);
  if (err != 0) {
    pr_info("RDATA error\n");
    return err;
  }

  if (convert8byteStringHash(THEAD, &map_id[3]))
    return -EINVAL;
  err = fstore_get_map_array_start(map_id[3], 4, 8, 1, (void**)&dtlb_head);
  if (err != 0) {
    pr_info("THEAD error\n");
    return err;
  }

  if (convert8byteStringHash(TDATA, &map_id[4]))
    return -EINVAL;
  err = fstore_get_map_array_start(map_id[4], 4, 24, BUFFER_ENTRIES, (void**)&dtlb_data);
  if (err != 0) {
    pr_info("TDATA error\n");
    return err;
  }

  is_blocked = 1;
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
