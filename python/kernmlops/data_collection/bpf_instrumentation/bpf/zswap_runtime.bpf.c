#include <linux/sched.h>
#include <linux/time.h>

typedef struct zswap_event {
  u32 pid;
  u32 tgid;
  u64 ts;
  u64 duration;
  char buff[16]; // opname
} zswap_event_t;

BPF_PERF_OUTPUT(zswap_store_events);
BPF_PERF_OUTPUT(zswap_load_events);
BPF_PERF_OUTPUT(zswap_invalidate_events);

BPF_HASH(stores, u64, u64);
BPF_HASH(loads, u64, u64);
BPF_HASH(invalidates, u64, u64);

int trace_zswap_store_entry(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64 ts = bpf_ktime_get_ns();
  stores.update(&id, &ts);
  return 0;
}

int trace_zswap_store_return(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64* start_ts = stores.lookup(&id);
  if (start_ts == 0)
    return 0;
  struct task_struct* task;
  if (IS_ERR(task = (struct task_struct*)PT_REGS_RC(ctx)))
    return 0;
  zswap_event_t event;
  event.ts = bpf_ktime_get_ns();
  event.duration = event.ts - *start_ts;
  event.pid = task->pid;
  event.tgid = task->tgid;
  char opname[16] = "store";
  bpf_probe_read_str(&event.buff, sizeof(event.buff), opname);
  zswap_store_events.perf_submit(ctx, &event, sizeof(event));
  stores.delete(&id);
  return 0;
}

int trace_zswap_load_entry(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64 ts = bpf_ktime_get_ns();
  loads.update(&id, &ts);
  return 0;
}

int trace_zswap_load_return(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64* start_ts = loads.lookup(&id);
  if (start_ts == 0)
    return 0;
  struct task_struct* task;
  if (IS_ERR(task = (struct task_struct*)PT_REGS_RC(ctx)))
    return 0;
  zswap_event_t event;
  event.ts = bpf_ktime_get_ns();
  event.duration = event.ts - *start_ts;
  event.pid = task->pid;
  event.tgid = task->tgid;
  char opname[16] = "load";
  bpf_probe_read_str(&event.buff, sizeof(event.buff), opname);
  zswap_load_events.perf_submit(ctx, &event, sizeof(event));
  loads.delete(&id);
  return 0;
}

int trace_zswap_invalidate_entry(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64 ts = bpf_ktime_get_ns();
  invalidates.update(&id, &ts);
  return 0;
}

int trace_zswap_invalidate_return(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64* start_ts = invalidates.lookup(&id);
  if (start_ts == 0)
    return 0;
  struct task_struct* task;
  if (IS_ERR(task = (struct task_struct*)PT_REGS_RC(ctx)))
    return 0;
  zswap_event_t event;
  event.ts = bpf_ktime_get_ns();
  event.duration = event.ts - *start_ts;
  event.pid = task->pid;
  event.tgid = task->tgid;
  char opname[16] = "invalidate";
  bpf_probe_read_str(&event.buff, sizeof(event.buff), opname);
  zswap_invalidate_events.perf_submit(ctx, &event, sizeof(event));
  invalidates.delete(&id);
  return 0;
}
