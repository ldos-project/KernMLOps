
// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2020 Wenbo Zhang
// modified version of
// https://github.com/iovisor/bcc/blob/master/tools/runqlat.py

#include <asm-generic/vmlinux.lds.h>
#include <linux/init_task.h>
#include <linux/sched.h>
#include <uapi/linux/ptrace.h>

typedef struct quanta_runtime_perf_event {
  u32 pid;
  u32 tgid;
  u64 quanta_end_uptime_us;
  u32 quanta_run_length_us;
} quanta_runtime_perf_event_t;

BPF_HASH(run_start, u32);
BPF_HASH(queue_start, u32);
BPF_PERF_OUTPUT(quanta_runtimes);
BPF_PERF_OUTPUT(quanta_queue_times);

#if USE_TRACEPOINT
RAW_TRACEPOINT_PROBE(sched_wakeup_new) {
  struct task_struct* p = (struct task_struct*)ctx->args[0];
#else
int trace_wake_up_new_task(struct pt_regs* ctx, struct task_struct* p) {
#endif
  u64 ts = bpf_ktime_get_ns();
  u32 pid = p->pid;
  queue_start.update(&pid, &ts);
  return 0;
}

#if USE_TRACEPOINT
RAW_TRACEPOINT_PROBE(sched_wakeup) {
  struct task_struct* p = (struct task_struct*)ctx->args[0];
#else
int trace_ttwu_do_wakeup(struct pt_regs* ctx, struct rq* rq, struct task_struct* p,
                         int wake_flags) {
#endif
  u64 ts = bpf_ktime_get_ns();

  u32 tgid = p->tgid;
  u32 pid = p->pid;
  if (FILTER || pid == 0)
    return 0;
  // update queue time to now since initial time spent blocked
  queue_start.update(&pid, &ts);
  return 0;
}

#if USE_TRACEPOINT
RAW_TRACEPOINT_PROBE(sched_switch) {
  // TP_PROTO(bool preempt, struct task_struct *prev, struct task_struct *next)
  struct task_struct* prev = (struct task_struct*)ctx->args[1];
  struct task_struct* next = (struct task_struct*)ctx->args[2];
  u32 next_pid = next->pid;
  u32 next_tgid = next->tgid;
#else
int trace_run(struct pt_regs* ctx, struct task_struct* prev) {
  u32 next_pid = bpf_get_current_pid_tgid();
  u32 next_tgid = bpf_get_current_pid_tgid() >> 32;
#endif

  u64 ts = bpf_ktime_get_ns();

  // ivcsw: treat next task like an enqueue event and store timestamp
  if (!(FILTER || next_pid == 0)) {
    // fetch timestamp and calculate delta
    u64 *tsp, delta;
    tsp = queue_start.lookup(&next_pid);
    if (tsp != 0) {
      delta = ts - *tsp;

      struct quanta_runtime_perf_event data;
      __builtin_memset(&data, 0, sizeof(data));
      data.pid = next_pid;
      data.tgid = next_tgid;
      // TODO(Patrick): avoid division and multiplication
      data.quanta_end_uptime_us = ts / 1000;
      data.quanta_run_length_us = delta / 1000;
      // TODO(Patrick): consider only submitting if greater than 10us or so
      quanta_queue_times.perf_submit(ctx, &data, sizeof(data));
      queue_start.delete(&next_pid);
    }
    run_start.update(&next_pid, &ts);
  }

  u32 tgid = prev->tgid;
  u32 pid = prev->pid;
  if (FILTER || pid == 0)
    return 0;
  u64 *tsp, delta;

  // fetch timestamp and calculate delta
  tsp = run_start.lookup(&pid);
  if (tsp == 0) {
    return 0; // missed enqueue
  }
  delta = ts - *tsp;

  struct quanta_runtime_perf_event data;
  __builtin_memset(&data, 0, sizeof(data));
  data.pid = pid;
  data.tgid = tgid;
  // TODO(Patrick): avoid division and multiplication
  data.quanta_end_uptime_us = ts / 1000;
  data.quanta_run_length_us = delta / 1000;

  quanta_runtimes.perf_submit(ctx, &data, sizeof(data));
  run_start.delete(&pid);
  queue_start.update(&pid, &ts);
  return 0;
}
