#include "../vmlinux.h"
#include "linnos_policy.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

#define MPMO_QUEUE_DEFINE(x, data)     \
  struct {                             \
    __uint(type, BPF_MAP_TYPE_ARRAY);  \
    __uint(max_entries, 1);            \
    __type(key, u32);                  \
    __type(value, u64);                \
    __uint(map_flags, BPF_F_MMAPABLE); \
  } mpmo_head_##x SEC(".maps");        \
                                       \
  struct {                             \
    __uint(type, BPF_MAP_TYPE_ARRAY);  \
    __uint(max_entries, BUFFER_SIZE);  \
    __type(key, u32);                  \
    __type(value, data);               \
    __uint(map_flags, BPF_F_MMAPABLE); \
  } mpmo_buff_##x SEC(".maps")

// ENUM FOR MPSO
u64 writing_val = 3;
u64 reading_val = 2;
u64 readable_val = 1;
u64 invalid_val = 0;

#define RESERVE_BUFFER_SPACE(tag, type, reserve_attempt)                                    \
  ({                                                                                        \
    u64* head_idx_ptr = bpf_map_lookup_elem(&mpmo_head_##tag, &zero_key);                   \
    type* location = NULL;                                                                  \
    bool ready = false;                                                                     \
    if (head_idx_ptr) {                                                                     \
      u32 index = (u32)__atomic_fetch_add(head_idx_ptr, 1, __ATOMIC_RELAXED);               \
      index = index % BUFFER_SIZE;                                                          \
      location = bpf_map_lookup_elem(&mpmo_buff_##tag, &index);                             \
      if (location) {                                                                       \
        for (int i = 0; i < reserve_attempt && !ready; i++) {                               \
          u64 valid = location->valid;                                                      \
          if (valid != writing_val) {                                                       \
            ready = __atomic_compare_exchange(&location->valid, &valid, &writing_val, true, \
                                              __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);          \
          }                                                                                 \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
    location;                                                                               \
  })

#define SUBMIT_DATA(ptr)                                                    \
  ({                                                                        \
    u64 blah;                                                               \
    __atomic_exchange(&ptr->valid, &readable_val, &blah, __ATOMIC_RELEASE); \
  })

#define REQ_IDLE            (1 << __REQ_IDLE)
#define REQ_FUA             (1 << __REQ_FUA)
#define REQ_META            (1 << __REQ_META)
#define RQF_SPECIAL_PAYLOAD (1 << __RQF_SPECIAL_PAYLOAD)
#define REQ_READ            (1 << __REQ_READ)

#define K_READ(name, where) bpf_probe_read_kernel(&name, sizeof(name), &(where))

typedef struct lengths {
  u64 ios_4k;
} queue_lengths;

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 3);
  __type(key, u32);
  __type(value, queue_lengths);
  __uint(map_flags, BPF_F_MMAPABLE);
} segment_array SEC(".maps");

static int block_4k_ios(u32 bytes) {
  return (bytes + 4095) >> 12;
}

static dev_t ddevt(struct request* rq) {
  struct request_queue* q;
  struct gendisk* disk_ptr;
  int major;
  int minor;
  if (bpf_probe_read_kernel(&q, sizeof(struct request_queue*), &rq->q))
    return 0;
  if (bpf_probe_read_kernel(&disk_ptr, sizeof(struct gendisk*), &q->disk))
    return 0;
  if (bpf_probe_read_kernel(&major, sizeof(int), &disk_ptr->major))
    return 0;
  if (bpf_probe_read_kernel(&minor, sizeof(int), &disk_ptr->first_minor))
    return 0;
  return (major << 20) | minor;
}

u32 select(u32 device) {
  switch (device) {
    case 271581184:
      return 0;
    case 271581186:
      return 1;
    case 271581188:
      return 2;
  }
  return 3;
}

typedef struct io_data {
  u64 ns;
  u64 ios_4k;
} io_data_t;

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 32768);
  __type(key, struct request*);
  __type(value, io_data_t);
} req_hash SEC(".maps");

SEC("raw_tracepoint/block_rq_issue")
int trace_block_rq_issue(struct bpf_raw_tracepoint_args* ctx) {
  u32 device = 0;
  struct request* rq = (void*)ctx->args[0];
  device = ddevt(rq);
  u32 index = select(device);
  queue_lengths* val = bpf_map_lookup_elem(&segment_array, &index);
  u32 bytes;
  if (K_READ(bytes, rq->__data_len))
    return 0;

  u32 blocks_4k = block_4k_ios(bytes);
  if (val) {
    __atomic_fetch_add(&val->ios_4k, blocks_4k, __ATOMIC_RELAXED);
  }
  u64 flags;
  K_READ(flags, rq->cmd_flags);
  if (flags == 0) {
    io_data_t data = {bpf_ktime_get_ns(), bytes};
    bpf_map_update_elem(&req_hash, &rq, &data, BPF_ANY);
  }
  return 0;
}

typedef struct io_history_data {
  u64 ts;
  u64 ios_4k;
  u64 latency;
  u64 valid;
} io_history_data_t;

MPMO_QUEUE_DEFINE(dev0, io_history_data_t);
MPMO_QUEUE_DEFINE(dev1, io_history_data_t);
MPMO_QUEUE_DEFINE(dev2, io_history_data_t);
const u32 zero_key = 0;
const u64 zero_val = 0;
const u64 one_val = 1;

#define IO_HISTORY_COMMIT(tag)                                                       \
  do {                                                                               \
    io_history_data_t* hist_data = RESERVE_BUFFER_SPACE(dev0, io_history_data_t, 1); \
    if (hist_data) {                                                                 \
      hist_data->ts = bpf_ktime_get_ns();                                            \
      hist_data->latency = hist_data->ts - data->ns;                                 \
      hist_data->ios_4k = blocks_4k;                                                 \
      SUBMIT_DATA(hist_data);                                                        \
    }                                                                                \
  } while (0)

SEC("raw_tracepoint/block_rq_complete")
int trace_block_rq_complete(struct bpf_raw_tracepoint_args* ctx) {
  u32 device = 0;
  struct request* rq = (void*)ctx->args[0];
  device = ddevt(rq);
  u32 index = select(device);
  u32 bytes;
  if (K_READ(bytes, rq->__data_len))
    return 0;
  u32 blocks_4k = block_4k_ios(bytes);
  queue_lengths* val = bpf_map_lookup_elem(&segment_array, &index);
  if (val) {
    __atomic_fetch_add(&val->ios_4k, -blocks_4k, __ATOMIC_RELAXED);
  }
  io_data_t* data = bpf_map_lookup_elem(&req_hash, &rq);
  if (data) {
    if (index == 0)
      IO_HISTORY_COMMIT(dev0);
    if (index == 1)
      IO_HISTORY_COMMIT(dev1);
    if (index == 2)
      IO_HISTORY_COMMIT(dev2);
  }
  return 0;
}
