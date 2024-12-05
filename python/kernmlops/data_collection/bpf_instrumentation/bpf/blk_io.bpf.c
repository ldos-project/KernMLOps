// Copyright (c) 2015 Brendan Gregg.
// Licensed under the Apache License, Version 2.0 (the "License")
// 20-Sep-2015   Brendan Gregg   Created this.
// 31-Mar-2022   Rocky Xing      Added disk filter support.
// 01-Aug-2023   Jerome Marchand Added support for block tracepoints
// modified version of: https://github.com/iovisor/bcc/blob/master/tools/biolatency.py

#include <linux/blk-mq.h>
#include <uapi/linux/ptrace.h>

typedef struct block_io_start_perf_event {
  u32 device;
  u64 sector;
  u32 segments;
  u32 block_io_bytes;
  u64 block_io_start_uptime_us;
  u64 block_io_flags;
  int queue_length_segments;
  int queue_length_4ks;
} block_io_start_perf_event_t;

typedef struct block_io_end_perf_event {
  u32 device;
  u64 sector;
  u32 segments;
  u32 block_io_bytes;
  u64 block_io_end_uptime_us;
  u64 block_latency_us;
  u64 block_io_latency_us;
  u64 block_io_flags;
} block_io_end_perf_event_t;

struct queue_lengths {
  int queue_length_segments;
  int queue_length_4ks;
};

struct start_key {
  u32 dev;
  u64 sector;
};

BPF_HASH(device_queue, u32, struct queue_lengths);
// we maintain started_4k_ios separately so we can manage scenarios where there is
// existing outstanding io for a device when this BPF program is installed
BPF_HASH(started_4k_ios, struct start_key, u32, 1024);
BPF_PERF_OUTPUT(block_io_starts);
BPF_PERF_OUTPUT(block_io_ends);

static dev_t ddevt(struct gendisk* disk) {
  return (disk->major << 20) | disk->first_minor;
}

static int block_4k_ios(u32 bytes) {
  return (bytes + 4095) >> 12;
}

// copied from linux kernel
static inline unsigned short blk_rq_nr_phys_segments_dup(struct request* rq) {
  if (rq->rq_flags & RQF_SPECIAL_PAYLOAD)
    return 1;
  return rq->nr_phys_segments;
}

// Linux defines number of segments as:
// Number of scatter-gather DMA addr+len pairs after
// physical address coalescing is performed.

// see https://docs.kernel.org/block/blk-mq.html
// inserts are rare and only occur when the hardware cannot accept new requests due to load
// block_rq_insert: https://elixir.bootlin.com/linux/v5.6/source/include/trace/events/block.h#L192

// when a block request is issued to the hardware driver
// block_rq_issue: https://elixir.bootlin.com/linux/v5.6/source/include/trace/events/block.h#L207
RAW_TRACEPOINT_PROBE(block_rq_issue) {
  struct request* req = (void*)ctx->args[0];
  u32 device = ddevt(req->__RQ_DISK__);
  u64 sector = req->__sector;
  u64 flags = req->cmd_flags;
  u32 segments = blk_rq_nr_phys_segments_dup(req);
  u32 bytes = req->__data_len;

  u64 ts = bpf_ktime_get_ns();

  struct queue_lengths init_entry;
  __builtin_memset(&init_entry, 0, sizeof(init_entry));
  struct queue_lengths* q_lengths = device_queue.lookup_or_try_init(&device, &init_entry);
  if (!q_lengths) {
    return 0;
  }
  int block_4ks = block_4k_ios(bytes);
  struct start_key start;
  __builtin_memset(&start, 0, sizeof(start));
  start.dev = device;
  start.sector = sector;
  started_4k_ios.update(&start, &block_4ks);

  __sync_fetch_and_add(&q_lengths->queue_length_4ks, block_4ks);
  __sync_fetch_and_add(&q_lengths->queue_length_segments, segments);
  // may be noisy by the atomic function cannot return a value
  // https://github.com/llvm/llvm-project/issues/91888
  int queue_length_4ks = q_lengths->queue_length_4ks;
  int queue_length_segments = q_lengths->queue_length_segments;

  // store io data
  struct block_io_start_perf_event data;
  __builtin_memset(&data, 0, sizeof(data));
  data.device = device;
  data.sector = sector;
  data.segments = segments;
  data.block_io_bytes = bytes;
  // TODO(Patrick): avoid division and multiplication
  data.block_io_start_uptime_us = ts / 1000;
  data.block_io_flags = flags;
  data.queue_length_4ks = queue_length_4ks;
  data.queue_length_segments = queue_length_segments;

  block_io_starts.perf_submit(ctx, &data, sizeof(data));

  return 0;
}
// fin: 172972899196. issue: 172972899111
// https://elixir.bootlin.com/linux/v5.6/source/include/trace/events/block.h#L116
RAW_TRACEPOINT_PROBE(block_rq_complete) {
  struct request* req = (void*)ctx->args[0];
  u32 device = ddevt(req->__RQ_DISK__);
  u64 sector = req->__sector;
  u64 flags = req->cmd_flags;
  u32 segments = blk_rq_nr_phys_segments_dup(req);
  u32 bytes = req->__data_len;

  u64 start_time_ns = req->start_time_ns;
  u64 io_start_time_ns = req->io_start_time_ns;
  u64 ts = bpf_ktime_get_ns();
  u64 io_delta = ts - io_start_time_ns;
  u64 delta = ts - start_time_ns;

  // store io data
  struct block_io_end_perf_event data;
  __builtin_memset(&data, 0, sizeof(data));
  data.device = device;
  data.sector = sector;
  data.segments = segments;
  data.block_io_bytes = bytes;
  // TODO(Patrick): avoid division and multiplication
  data.block_io_end_uptime_us = ts / 1000;
  data.block_latency_us = delta / 1000;
  data.block_io_latency_us = io_delta / 1000;
  data.block_io_flags = flags;

  block_io_ends.perf_submit(ctx, &data, sizeof(data));

  // get device queue for request that just finished
  struct queue_lengths* q_lengths = device_queue.lookup(&device);
  // if the queue has not been initialized, finish early
  if (!q_lengths) {
    return 0;
  }
  // ensure that the finished block io was tracked when it was inserted
  struct start_key start;
  __builtin_memset(&start, 0, sizeof(start));
  start.dev = device;
  start.sector = sector;
  if (!started_4k_ios.lookup(&start)) {
    return 0;
  }
  // clear the inserted state
  started_4k_ios.delete(&start);

  // update device queue length to not include this finished io
  int block_4ks = block_4k_ios(bytes);
  __sync_fetch_and_add(&q_lengths->queue_length_4ks, -block_4ks);
  __sync_fetch_and_add(&q_lengths->queue_length_segments, -segments);

  return 0;
}
