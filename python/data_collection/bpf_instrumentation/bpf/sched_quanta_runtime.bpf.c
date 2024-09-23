
// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2020 Wenbo Zhang
// modified version of
// https://github.com/iovisor/bcc/blob/master/tools/runqlat.py

#include <linux/init_task.h>
#include <linux/sched.h>
#include <uapi/linux/ptrace.h>

typedef struct quanta_runtime_perf_event {
  u32 pid;
  u32 tgid;
  u64 quanta_end_uptime_us;
  u32 quanta_run_length_us;
} quanta_runtime_perf_event_t;

BPF_HASH(start, u32);
BPF_PERF_OUTPUT(quanta_runtimes);

#if USE_TRACEPOINT
RAW_TRACEPOINT_PROBE(sched_switch) {
  // TP_PROTO(bool preempt, struct task_struct *prev, struct task_struct *next)
  struct task_struct* prev = (struct task_struct*)ctx->args[1];
  struct task_struct* next = (struct task_struct*)ctx->args[2];
  u32 next_pid = next->pid;
#else
int trace_run(struct pt_regs* ctx, struct task_struct* prev) {
  u32 next_pid = bpf_get_current_pid_tgid();
#endif

  // ivcsw: treat next task like an enqueue event and store timestamp
  if (!(FILTER || next_pid == 0)) {
    u64 ts = bpf_ktime_get_ns();
    start.update(&next_pid, &ts);
  }

  u32 tgid = prev->tgid;
  u32 pid = prev->pid;
  if (FILTER || pid == 0)
    return 0;
  u64 *tsp, delta;

  // fetch timestamp and calculate delta
  tsp = start.lookup(&pid);
  if (tsp == 0) {
    return 0; // missed enqueue
  }
  u64 ts = bpf_ktime_get_ns();
  delta = ts - *tsp;

  struct quanta_runtime_perf_event data;
  __builtin_memset(&data, 0, sizeof(data));
  data.pid = pid;
  data.tgid = tgid;
  data.quanta_end_uptime_us = ts / 1000;
  data.quanta_run_length_us = delta / 1000;

  // store as histogram
  quanta_runtimes.perf_submit(ctx, &data, sizeof(data));
  start.delete(&pid);
  return 0;
}
