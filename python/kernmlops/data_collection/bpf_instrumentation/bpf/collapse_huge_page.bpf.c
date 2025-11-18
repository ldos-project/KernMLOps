
// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2020 Wenbo Zhang
// modified version of
// https://github.com/iovisor/bcc/blob/master/tools/runqlat.py

#include <asm-generic/vmlinux.lds.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <uapi/linux/ptrace.h>

typedef struct basic_info {
  u32 pid;
  u32 tgid;
  u64 ts_start;
  u64 ts_end;
} basic_info_t;

typedef struct trace_mm_khugepaged_scan_pmd_struct {
  u32 pid;
  u32 tgid;
  u64 start_ts_ns;
  u64 end_ts_ns;
  u64 mm;
  u64 page;
  u32 writeable;
  u32 referenced;
  u32 none_or_zero;
  u32 status;
  u32 unmapped;
} trace_mm_khugepaged_scan_pmd_t;

BPF_PERF_OUTPUT(trace_mm_khugepaged_scan_pmds);

RAW_TRACEPOINT_PROBE(mm_khugepaged_scan_pmd) {
  u64 start = bpf_ktime_get_ns();
  trace_mm_khugepaged_scan_pmd_t data;
  __builtin_memset(&data, 0, sizeof(data));
  struct mm_struct* mm = (struct mm_struct*)ctx->args[0];
  data.start_ts_ns = start;
  data.mm = (u64)ctx->args[0];
  data.tgid = mm->owner->tgid;
  data.pid = mm->owner->pid;
  data.page = (u64)ctx->args[1];
  data.writeable = ctx->args[2];
  data.referenced = ctx->args[3];
  data.none_or_zero = ctx->args[4];
  data.status = ctx->args[5];
  data.unmapped = ctx->args[6];
  data.end_ts_ns = bpf_ktime_get_ns();
  trace_mm_khugepaged_scan_pmds.perf_submit(ctx, &data, sizeof(data));
  return 0;
}

BPF_PERF_OUTPUT(collapse_huge_pages);

typedef struct collapse_huge_page_struct {
  u32 pid;
  u32 tgid;
  u64 start_ts_ns;
  u64 end_ts_ns;
  u64 mm;
  u64 address;
  u32 referenced;
  u32 unmapped;
  u64 cc;
} collapse_huge_page_t;

int kprobe_collapse_huge_page(struct pt_regs* ctx, struct mm_struct* mm, u64 address,
                              int referenced, int unmapped, struct collapse_control* cc) {
  u64 start = bpf_ktime_get_ns();
  collapse_huge_page_t data;
  __builtin_memset(&data, 0, sizeof(data));
  data.mm = (u64)mm;
  data.address = address;
  data.referenced = referenced;
  data.unmapped = unmapped;
  data.pid = mm->owner->pid;
  data.tgid = mm->owner->tgid;
  data.cc = (u64)cc;
  data.start_ts_ns = start;
  data.end_ts_ns = bpf_ktime_get_ns();
  collapse_huge_pages.perf_submit(ctx, &data, sizeof(data));
  return 0;
}

typedef struct trace_mm_collapse_huge_page_struct {
  u32 pid;
  u32 tgid;
  u64 start_ts_ns;
  u64 end_ts_ns;
  u64 mm;
  u32 isolated;
  u32 status;
} trace_mm_collapse_huge_page_t;

BPF_PERF_OUTPUT(trace_mm_collapse_huge_pages);
// If this succeeds a folio was allocated meaning there was space

RAW_TRACEPOINT_PROBE(mm_collapse_huge_page) {
  u64 start = bpf_ktime_get_ns();
  trace_mm_collapse_huge_page_t data;
  __builtin_memset(&data, 0, sizeof(data));
  struct mm_struct* mm = (struct mm_struct*)ctx->args[0];
  data.isolated = (u32)ctx->args[1];
  data.status = (u32)ctx->args[2];
  data.pid = mm->owner->pid;
  data.tgid = mm->owner->tgid;
  data.start_ts_ns = start;
  data.end_ts_ns = bpf_ktime_get_ns();
  trace_mm_collapse_huge_pages.perf_submit(ctx, &data, sizeof(data));
  return 0;
}
