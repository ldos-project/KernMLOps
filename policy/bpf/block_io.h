// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2015 Brendan Gregg.
// Licensed under the Apache License, Version 2.0 (the "License")
// 20-Sep-2015   Brendan Gregg   Created this.
// 31-Mar-2022   Rocky Xing      Added disk filter support.
// 01-Aug-2023   Jerome Marchand Added support for block tracepoints
// modified version of: https://github.com/iovisor/bcc/blob/master/tools/biolatency.py

#ifndef POLICY_BPF_BLOCK_IO_H
#define POLICY_BPF_BLOCK_IO_H

#include <linux/types.h>

typedef __u16 u16;
typedef __u32 u32;
typedef __u64 u64;

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

// REQ_* bits: https://github.com/torvalds/linux/blob/v6.12/include/linux/blk_types.h
#define BLOCK_REQ_META (1U << 12)
#define BLOCK_REQ_IDLE (1U << 15)
#define BLOCK_REQ_FUA (1U << 17)

// RQF_SPECIAL_PAYLOAD, anonymous enum index 12 (not the older bit 18):
// https://github.com/torvalds/linux/blob/v6.12/include/linux/blk-mq.h
#define BLOCK_REQ_SPECIAL_PAYLOAD (1U << 12)

/*
 * Minimal types for Linux v6.12's request->q->disk path only, with no rq_disk
 * fallback. Live field offsets are relocated by CO-RE using kernel BTF.
 * POLICY_REPLAY uses these layouts directly for deterministic native replay.
 */
#ifndef POLICY_REPLAY
#define BLOCK_PRESERVE_ACCESS_INDEX __attribute__((preserve_access_index))
#else
#define BLOCK_PRESERVE_ACCESS_INDEX
#endif

struct gendisk {
  int major;
  int first_minor;
} BLOCK_PRESERVE_ACCESS_INDEX;

struct request_queue {
  struct gendisk *disk;
} BLOCK_PRESERVE_ACCESS_INDEX;

struct request {
  struct request_queue *q;
  u32 cmd_flags;
  u32 rq_flags;
  u32 __data_len;
  u64 __sector;
  u64 start_time_ns;
  u64 io_start_time_ns;
  u16 nr_phys_segments;
} BLOCK_PRESERVE_ACCESS_INDEX;

#undef BLOCK_PRESERVE_ACCESS_INDEX

#endif
