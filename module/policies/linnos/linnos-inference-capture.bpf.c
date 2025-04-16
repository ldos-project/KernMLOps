#include "vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

typedef struct lengths {
  u64 segments;
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

#define REQ_IDLE            (1 << __REQ_IDLE)
#define REQ_FUA             (1 << __REQ_FUA)
#define REQ_META            (1 << __REQ_META)
#define RQF_SPECIAL_PAYLOAD (1 << __RQF_SPECIAL_PAYLOAD)

#define K_READ_DEF(type, name, where) \
  { (type name; bpf_probe_read_kernel(&name, sizeof(type), &(where));) }

#define K_READ(name, where) bpf_probe_read_kernel(&name, sizeof(name), &(where))

// copied from linux kernel
static inline unsigned short blk_rq_nr_phys_segments_dup(struct request* rq) {
  blk_opf_t cmd_flags;
  if (!bpf_probe_read_kernel(&cmd_flags, sizeof(cmd_flags), &rq->cmd_flags) &&
      (cmd_flags & REQ_IDLE) && (cmd_flags & REQ_FUA) && (cmd_flags & REQ_META))
    return 0;

  req_flags_t rq_flags;
  if (!bpf_probe_read_kernel(&rq_flags, sizeof(req_flags_t), &rq->rq_flags) &&
      rq_flags & RQF_SPECIAL_PAYLOAD)
    return 1;

  short unsigned int nr_phys_segments;
  if (K_READ(nr_phys_segments, rq->nr_phys_segments))
    return 0;
  return nr_phys_segments;
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
    case 271581185:
      return 1;
    case 271581186:
      return 2;
  }
  return 3;
}

SEC("raw_tracepoint/block_rq_issue")
int trace_block_rq_issue(struct bpf_raw_tracepoint_args* ctx) {
  u32 device = 0;
  struct request* rq = (void*)ctx->args[0];
  device = ddevt(rq);
  u32 index = select(device);
  queue_lengths* val = bpf_map_lookup_elem(&segment_array, &index);
  if (val) {
    u32 bytes;
    if (K_READ(bytes, rq->__data_len))
      return 0;

    u32 segment = blk_rq_nr_phys_segments_dup(rq);
    u32 blocks_4k = block_4k_ios(bytes);
    __atomic_fetch_add(&val->segments, segment, __ATOMIC_RELAXED);
    __atomic_fetch_add(&val->ios_4k, blocks_4k, __ATOMIC_RELAXED);
  }
  return 0;
}

SEC("raw_tracepoint/block_rq_complete")
int trace_block_rq_complete(struct bpf_raw_tracepoint_args* ctx) {
  u32 device = 0;
  struct request* rq = (void*)ctx->args[0];
  device = ddevt(rq);
  u32 index = select(device);
  queue_lengths* val = bpf_map_lookup_elem(&segment_array, &index);
  if (val) {
    u32 bytes;
    if (K_READ(bytes, rq->__data_len))
      return 0;

    u32 segment = blk_rq_nr_phys_segments_dup(rq);
    u32 blocks_4k = block_4k_ios(bytes);
    __atomic_fetch_add(&val->segments, -segment, __ATOMIC_RELAXED);
    __atomic_fetch_add(&val->ios_4k, -blocks_4k, __ATOMIC_RELAXED);
  }
  return 0;
}
