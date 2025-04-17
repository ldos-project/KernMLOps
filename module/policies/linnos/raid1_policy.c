#include "../../fstore/fstore.h"
#include "../../linux/drivers/md/md.h"
#include "../../linux/drivers/md/raid1.h"
#include "linnos-inference-capture.h"
#include <asm/fpu/api.h>
#include <linux/bio.h>
#include <linux/blk_types.h>
#include <linux/blkdev.h>
#include <linux/delay.h>
#include <linux/module.h>
#include <linux/module.h> /* Needed by all modules */
#include <linux/printk.h> /* Needed for pr_info() */
#include <linux/timex.h>
#define MAX_LAYER_SIZE 8
#define INPUT_SIZE     4
#define OUTPUT_SIZE    2

#define IO_BLOCKED     ((struct bio*)1)

static u32 select(u32 device) {
  switch (device) {
    case 271581184:
      return 0;
    case 271581185:
      return 1;
    case 271581186:
      return 2;
  }
  return 3;
}

int (*ml_choose_best_rdev_raid1)(struct r1conf* conf, struct r1bio* r1_bio);

bool rdev_readable(struct md_rdev* rdev, struct r1bio* r1_bio);

static const float weights0[] = {
    -0.193689, -0.332377, 0.399570, 0.136509,  0.093448,  -0.007472, 0.116340,  0.360244,
    -0.179984, 0.001694,  0.194476, 0.101892,  -0.426636, -0.216380, -0.393331, 0.430029,
    -0.026020, 0.059531,  0.368225, 0.164477,  -0.213492, 0.356735,  -0.134896, -0.408591,
    0.086748,  0.246952,  0.077820, -0.363860, 0.197074,  0.221929,  -0.302366, -0.298428};

static const float bias0[] = {-0.545327, -0.010195, 0.031252,  0.502827,
                              0.492101,  -0.376440, -0.243972, 0.438466};

static const float weights1[] = {
    -0.300106, 0.354516,  0.230332,  0.232282,  -0.352630, 0.255562,  0.286923,  -0.311827,
    0.285720,  -0.261634, 0.134430,  -0.115013, -0.038138, -0.037236, 0.369536,  -0.305061,
    -0.208890, 0.268854,  -0.248951, 0.105301,  0.045945,  -0.149808, -0.057232, -0.059342,
    -0.366640, -0.205238, 0.206782,  0.118410,  -0.034608, -0.014186, -0.160369, 0.057799,
    -0.135360, -0.202417, 0.249197,  -0.198105, 0.149266,  -0.082429, -0.027583, -0.095676,
    -0.012737, -0.259226, 0.410471,  -0.216811, -0.132738, -0.011571, -0.336452, 0.043018,
    0.037711,  -0.379006, 0.252055,  -0.311444, -0.010601, -0.070340, 0.202527,  -0.253710,
    -0.360516, -0.078323, 0.112554,  0.286938,  0.015296,  0.199878,  0.042413,  -0.030386};

static const float bias1[] = {-0.118379, -0.243972, -0.116336, -0.193931,
                              -0.269090, 0.105460,  -0.197530, -0.200359};

static const float weights2[] = {
    -0.232786, -0.228923, -0.153763, -0.283201, -0.087366, 0.153033,  -0.069204, 0.013007,
    0.025398,  0.067460,  -0.297556, -0.040284, 0.285558,  -0.205139, 0.141009,  0.097410,
    -0.006487, -0.335677, 0.088062,  -0.286287, 0.116932,  -0.210068, -0.267364, -0.110440,
    0.256830,  -0.018506, 0.178209,  0.140168,  -0.012694, 0.221618,  0.305810,  -0.321790,
    -0.079187, 0.123309,  0.070792,  0.235180,  0.056213,  -0.241868, -0.171703, 0.142501,
    0.217905,  0.069851,  -0.242289, 0.341904,  0.049348,  0.290618,  0.246277,  -0.321463,
    0.138742,  -0.111137, 0.206621,  -0.157411, 0.302911,  -0.264413, 0.271527,  -0.223631,
    0.167721,  0.149767,  -0.304719, -0.222625, -0.024201, 0.205862,  -0.195138, -0.229373};

static const float bias2[] = {-0.261172, -0.181218, 0.333258,  -0.154462,
                              -0.227940, 0.111932,  -0.205809, -0.075186};

static const float weights3[] = {-0.280009, -0.034768, 0.160847,  -0.294175, -0.239241, -0.051094,
                                 0.154589,  -0.327220, -0.031476, 0.214947,  -0.016209, 0.261974,
                                 -0.072670, 0.161183,  0.355928,  -0.194063};

static const float bias3[] = {-0.102024, 0.178842};

static void forward(const float* input, float* output, float* arr1, float* arr2) {
  float* temp;
  float *curr = arr1, *next = arr2;

  // Layer 0
  for (int i = 0; i < 8; i++) {
    float sum = 0.0f;
    for (int j = 0; j < 4; j++) {
      sum += input[j] * weights0[i * 4 + j];
    }
    curr[i] = sum + bias0[i];
  }

  // Layer 1
  for (int i = 0; i < 8; i++) {
    float sum = 0.0f;
    for (int j = 0; j < 8; j++) {
      sum += curr[j] * weights1[i * 8 + j];
    }
    next[i] = sum + bias1[i];
  }
  temp = curr;
  curr = next;
  next = temp;

  // Layer 2
  for (int i = 0; i < 8; i++) {
    float sum = 0.0f;
    for (int j = 0; j < 8; j++) {
      sum += curr[j] * weights2[i * 8 + j];
    }
    next[i] = sum + bias2[i];
  }
  temp = curr;
  curr = next;
  next = temp;

  // Output Layer
  for (int i = 0; i < 2; i++) {
    float sum = 0.0f;
    for (int j = 0; j < 8; j++) {
      sum += curr[j] * weights3[i * 8 + j];
    }
    output[i] = sum + bias3[i];
  }
}

atomic_t start_loc = ATOMIC_INIT(0);

typedef struct lengths {
  u64 segments;
  u64 ios_4k;
} queue_lengths;

queue_lengths* qs;

static int infer(struct r1conf* conf, struct r1bio* r1_bio) {
  unsigned int raid_disks = conf->raid_disks;
  unsigned int start = ((unsigned int)atomic_inc_return(&start_loc)) % raid_disks;

  struct md_rdev* rdev;

  unsigned int curr = start;
  kernel_fpu_begin();
  float input[INPUT_SIZE];
  float sync_flag = (r1_bio->master_bio->bi_opf & REQ_SYNC) ? 1 : 0;
  float nomerge_flag = (r1_bio->master_bio->bi_opf & REQ_NOMERGE) ? 1 : 0;
  input[2] = sync_flag;
  input[3] = nomerge_flag;
  for (unsigned int i = 0; i < raid_disks; i++) {
    float output[OUTPUT_SIZE];
    float arr1[MAX_LAYER_SIZE];
    float arr2[MAX_LAYER_SIZE];
    curr = (start + i) % raid_disks;

    if (r1_bio->bios[curr] == IO_BLOCKED) {
      continue;
    }

    rdev = conf->mirrors[curr].rdev;
    if (!rdev_readable(rdev, r1_bio)) {
      continue;
    }
    if (rdev->bdev == NULL) {
      continue;
    }
    u32 index = select(disk_devt(rdev->bdev->bd_disk));
    if (index > 2) {
      pr_err("Extremely bad error has occurred\n");
      return curr;
    }

    input[0] = READ_ONCE(qs[index].segments);
    input[1] = READ_ONCE(qs[index].ios_4k);
    forward(input, output, arr1, arr2);
    if (output[0] - output[1]) {
      break;
    }
  }
  kernel_fpu_end();
  return curr;
}

int fstore_get_map_array_start(u64 map_name, size_t key_size, size_t value_size,
                               size_t num_elements, void** map_ptr);

int fstore_put_map_array(u64 map_name);

__u64 map_id;

int __init init_module(void) {
  if (convert8byteStringHash(NAME, &map_id))
    return -EINVAL;
  int err = fstore_get_map_array_start(map_id, 4, sizeof(queue_lengths), 3, (void**)&qs);
  if (err != 0)
    return err;
  ml_choose_best_rdev_raid1 = infer;
  return 0;
}

void __exit cleanup_module(void) {
  fstore_put_map_array(map_id);
  ml_choose_best_rdev_raid1 = NULL;
  pr_info("Goodbye \n");
}

MODULE_LICENSE("GPL");
