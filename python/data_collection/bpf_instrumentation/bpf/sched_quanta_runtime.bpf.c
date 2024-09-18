
// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2020 Wenbo Zhang
// modified version of
// https://github.com/iovisor/bcc/blob/master/tools/runqlat.py

#include <linux/init_task.h>
#include <linux/nsproxy.h>
#include <linux/pid_namespace.h>
#include <linux/sched.h>
#include <uapi/linux/ptrace.h>

typedef struct pid_key {
  u32 id;
  u64 slot;
} pid_key_t;

typedef struct pidns_key {
  u32 id;
  u64 slot;
} pidns_key_t;

typedef struct quanta_runtime_perf_event {
  u32 pid;
  u32 tgid;
  u64 quanta_end_uptime_us;
  u32 quanta_run_length_us;
} quanta_runtime_perf_event_t;

BPF_HASH(start, u32);
BPF_PERF_OUTPUT(quanta_runtimes);

// record enqueue timestamp
static int trace_enqueue(u32 tgid, u32 pid) {
  if (FILTER || pid == 0)
    return 0;
  u64 ts = bpf_ktime_get_ns();
  start.update(&pid, &ts);
  return 0;
}

static __always_inline unsigned int pid_namespace(struct task_struct* task) {
/* pids[] was removed from task_struct since commit
 * 2c4704756cab7cfa031ada4dab361562f0e357c0 Using the macro INIT_PID_LINK as a
 * conditional judgment.
 */
#ifdef INIT_PID_LINK
  struct pid_link pids;
  unsigned int level;
  struct upid upid;
  struct ns_common ns;

  /*  get the pid namespace by following task_active_pid_ns(),
   *  pid->numbers[pid->level].ns
   */
  bpf_probe_read_kernel(&pids, sizeof(pids), &task->pids[PIDTYPE_PID]);
  bpf_probe_read_kernel(&level, sizeof(level), &pids.pid->level);
  bpf_probe_read_kernel(&upid, sizeof(upid), &pids.pid->numbers[level]);
  bpf_probe_read_kernel(&ns, sizeof(ns), &upid.ns->ns);

  return ns.inum;
#else
  struct pid* pid;
  unsigned int level;
  struct upid upid;
  struct ns_common ns;

  /*  get the pid namespace by following task_active_pid_ns(),
   *  pid->numbers[pid->level].ns
   */
  bpf_probe_read_kernel(&pid, sizeof(pid), &task->thread_pid);
  bpf_probe_read_kernel(&level, sizeof(level), &pid->level);
  bpf_probe_read_kernel(&upid, sizeof(upid), &pid->numbers[level]);
  bpf_probe_read_kernel(&ns, sizeof(ns), &upid.ns->ns);

  return ns.inum;
#endif
}

// These tracepoints track when a process "wakes" up from an event like IO
// and can be rescheduled

// RAW_TRACEPOINT_PROBE(sched_wakeup)
// {
//  // TP_PROTO(struct task_struct *p)
//  struct task_struct *p = (struct task_struct *)ctx->args[0];
//  return trace_enqueue(p->tgid, p->pid);
// }

// RAW_TRACEPOINT_PROBE(sched_wakeup_new)
// {
//  // TP_PROTO(struct task_struct *p)
//  struct task_struct *p = (struct task_struct *)ctx->args[0];
//  return trace_enqueue(p->tgid, p->pid);
// }

RAW_TRACEPOINT_PROBE(sched_switch) {
  // TP_PROTO(bool preempt, struct task_struct *prev, struct task_struct *next)
  struct task_struct* prev = (struct task_struct*)ctx->args[1];
  struct task_struct* next = (struct task_struct*)ctx->args[2];
  u32 pid, tgid;

  // ivcsw: treat next task like an enqueue event and store timestamp
  u32 next_tgid = next->tgid;
  u32 next_pid = next->pid;
  if (!(FILTER || next_pid == 0)) {
    u64 ts = bpf_ktime_get_ns();
    start.update(&next_pid, &ts);
  }

  tgid = prev->tgid;
  pid = prev->pid;
  if (FILTER || pid == 0)
    return 0;
  struct quanta_runtime_perf_event data;
  u64 *tsp, delta;

  // fetch timestamp and calculate delta
  tsp = start.lookup(&pid);
  if (tsp == 0) {
    return 0; // missed enqueue
  }
  u64 ts = bpf_ktime_get_ns();
  delta = ts - *tsp;

  data.pid = pid;
  data.tgid = tgid;
  data.quanta_end_uptime_us = ts / 1000;
  data.quanta_run_length_us = delta / 1000;

  // store as histogram
  quanta_runtimes.perf_submit(ctx, &data, sizeof(data));
  start.delete(&pid);
  return 0;
}
