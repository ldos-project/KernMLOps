// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2015 Brendan Gregg.
// Licensed under the Apache License, Version 2.0 (the "License")
// 20-Sep-2015   Brendan Gregg   Created this.
// 31-Mar-2022   Rocky Xing      Added disk filter support.
// 01-Aug-2023   Jerome Marchand Added support for block tracepoints
// modified version of: https://github.com/iovisor/bcc/blob/master/tools/biolatency.py

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#ifndef POLICY_REPLAY
#include <bpf/bpf_core_read.h>
#endif

#include "block_io.h"

_Static_assert(sizeof(block_io_start_perf_event_t) == 48, "original issue layout");
_Static_assert(sizeof(block_io_end_perf_event_t) == 56, "original completion layout");

#ifndef POLICY_REPLAY
#define BLOCK_READ(obj, field) BPF_CORE_READ(obj, field)
#else
// Keep the real probe-read helper ID in replay, but omit CO-RE intrinsics.
#define BLOCK_READ(obj, field)                                                 \
  ({                                                                         \
    typeof((obj)->field) tmp = {0};                                           \
    bpf_probe_read_kernel(&tmp, sizeof(tmp), &(obj)->field);                    \
    tmp;                                                                     \
  })
#endif

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 10240);
  __type(key, u32);
  __type(value, struct queue_lengths);
} device_queue SEC(".maps");

// Track membership separately for IO already outstanding at probe installation.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 1024);
  __type(key, struct start_key);
  __type(value, u32);
} started_4k_ios SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 18);
} block_io_starts SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 18);
} block_io_ends SEC(".maps");

static __always_inline u32 ddevt(struct gendisk *disk) {
  return (BLOCK_READ(disk, major) << 20) | BLOCK_READ(disk, first_minor);
}

static __always_inline int block_4k_ios(u32 bytes) {
  return (bytes + 4095) >> 12;
}

// Copied from the Linux kernel; preserve the three separate cmd_flags reads.
static __always_inline unsigned short
blk_rq_nr_phys_segments_dup(struct request *rq) {
  if ((BLOCK_READ(rq, cmd_flags) & BLOCK_REQ_IDLE) &&
      (BLOCK_READ(rq, cmd_flags) & BLOCK_REQ_FUA) &&
      (BLOCK_READ(rq, cmd_flags) & BLOCK_REQ_META))
    return 0;
  if (BLOCK_READ(rq, rq_flags) & BLOCK_REQ_SPECIAL_PAYLOAD)
    return 1;
  return BLOCK_READ(rq, nr_phys_segments);
}

// Number of scatter-gather DMA address/length pairs after physical coalescing.
// https://docs.kernel.org/block/blk-mq.html
SEC("raw_tp/block_rq_issue")
int block_rq_issue(struct bpf_raw_tracepoint_args *ctx) {
  struct request *req = (void *)ctx->args[0];
  struct request_queue *queue = BLOCK_READ(req, q);
  struct gendisk *disk = BLOCK_READ(queue, disk);
  u32 device = ddevt(disk);
  u64 sector = BLOCK_READ(req, __sector);
  u64 flags = BLOCK_READ(req, cmd_flags);
  u32 segments = blk_rq_nr_phys_segments_dup(req);
  u32 bytes = BLOCK_READ(req, __data_len);

  u64 ts = bpf_ktime_get_ns();

  struct queue_lengths init_entry = {0};
  // Match lookup_or_try_init, including the lookup after a raced/failed insert.
  struct queue_lengths *q_lengths = bpf_map_lookup_elem(&device_queue, &device);
  if (!q_lengths) {
    bpf_map_update_elem(&device_queue, &device, &init_entry, BPF_NOEXIST);
    q_lengths = bpf_map_lookup_elem(&device_queue, &device);
  }
  if (!q_lengths) {
    return 0;
  }
  int block_4ks = block_4k_ios(bytes);
  struct start_key start = {0};
  start.dev = device;
  start.sector = sector;
  bpf_map_update_elem(&started_4k_ios, &start, &block_4ks, BPF_ANY);

  __sync_fetch_and_add(&q_lengths->queue_length_4ks, block_4ks);
  __sync_fetch_and_add(&q_lengths->queue_length_segments, segments);
  // These snapshots may be noisy; do not use the atomic return values.
  // https://github.com/llvm/llvm-project/issues/91888
  int queue_length_4ks = q_lengths->queue_length_4ks;
  int queue_length_segments = q_lengths->queue_length_segments;

  block_io_start_perf_event_t data = {0};
  data.device = device;
  data.sector = sector;
  data.segments = segments;
  data.block_io_bytes = bytes;
  data.block_io_start_uptime_us = ts / 1000;
  data.block_io_flags = flags;
  data.queue_length_4ks = queue_length_4ks;
  data.queue_length_segments = queue_length_segments;

  bpf_ringbuf_output(&block_io_starts, &data, sizeof(data), 0);
  return 0;
}

SEC("raw_tp/block_rq_complete")
int block_rq_complete(struct bpf_raw_tracepoint_args *ctx) {
  struct request *req = (void *)ctx->args[0];
  struct request_queue *queue = BLOCK_READ(req, q);
  struct gendisk *disk = BLOCK_READ(queue, disk);
  u32 device = ddevt(disk);
  u64 sector = BLOCK_READ(req, __sector);
  u64 flags = BLOCK_READ(req, cmd_flags);
  u32 segments = blk_rq_nr_phys_segments_dup(req);
  u32 bytes = BLOCK_READ(req, __data_len);

  u64 start_time_ns = BLOCK_READ(req, start_time_ns);
  u64 io_start_time_ns = BLOCK_READ(req, io_start_time_ns);
  u64 ts = bpf_ktime_get_ns();
  u64 io_delta = ts - io_start_time_ns;
  u64 delta = ts - start_time_ns;

  block_io_end_perf_event_t data = {0};
  data.device = device;
  data.sector = sector;
  data.segments = segments;
  data.block_io_bytes = bytes;
  data.block_io_end_uptime_us = ts / 1000;
  data.block_latency_us = delta / 1000;
  data.block_io_latency_us = io_delta / 1000;
  data.block_io_flags = flags;

  bpf_ringbuf_output(&block_io_ends, &data, sizeof(data), 0);

  // Queue cleanup must run after emission, even if the ring buffer is full.
  struct queue_lengths *q_lengths = bpf_map_lookup_elem(&device_queue, &device);
  if (!q_lengths) {
    return 0;
  }
  struct start_key start = {0};
  start.dev = device;
  start.sector = sector;
  if (!bpf_map_lookup_elem(&started_4k_ios, &start)) {
    return 0;
  }
  bpf_map_delete_elem(&started_4k_ios, &start);

  // Membership, not the saved value, determines whether to subtract these bytes.
  int block_4ks = block_4k_ios(bytes);
  __sync_fetch_and_add(&q_lengths->queue_length_4ks, -block_4ks);
  __sync_fetch_and_add(&q_lengths->queue_length_segments, -segments);

  return 0;
}

char LICENSE[] SEC("license") = "GPL";
