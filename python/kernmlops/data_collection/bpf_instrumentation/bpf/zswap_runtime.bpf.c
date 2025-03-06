#include <linux/sched.h>
#include <linux/time.h>

typedef struct zswap_event {
  u32 pid;
  u32 tgid;
  u64 start_ts;
  u64 end_ts;
} zswap_event_t;

BPF_PERF_OUTPUT(zswap_store_events);
BPF_PERF_OUTPUT(zswap_load_events);
BPF_PERF_OUTPUT(zswap_invalidate_events);

BPF_HASH(stores, u64, u64);
BPF_HASH(loads, u64, u64);
BPF_HASH(invalidates, u64, u64);

int trace_zswap_store_entry(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64 start_ts = bpf_ktime_get_ns();
  stores.update(&id, &start_ts);
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
  event.pid = (u32)(id);
  event.tgid = (u32)(id >> 32);
  event.start_ts = *start_ts;
  event.end_ts = bpf_ktime_get_ns();
  zswap_store_events.perf_submit(ctx, &event, sizeof(event));
  stores.delete(&id);
  return 0;
}

int trace_zswap_load_entry(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64 start_ts = bpf_ktime_get_ns();
  loads.update(&id, &start_ts);
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
  event.pid = (u32)(id);
  event.tgid = (u32)(id >> 32);
  event.start_ts = *start_ts;
  event.end_ts = bpf_ktime_get_ns();
  zswap_load_events.perf_submit(ctx, &event, sizeof(event));
  loads.delete(&id);
  return 0;
}

int trace_zswap_invalidate_entry(struct pt_regs* ctx) {
  u64 id = bpf_get_current_pid_tgid();
  u64 start_ts = bpf_ktime_get_ns();
  invalidates.update(&id, &start_ts);
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
  event.pid = (u32)(id);
  event.tgid = (u32)(id >> 32);
  event.start_ts = *start_ts;
  event.end_ts = bpf_ktime_get_ns();
  zswap_invalidate_events.perf_submit(ctx, &event, sizeof(event));
  invalidates.delete(&id);
  return 0;
}
