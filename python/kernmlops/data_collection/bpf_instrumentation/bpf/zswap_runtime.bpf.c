#include <linux/sched.h>
#include <linux/time.h>

#include "kermit.h"

typedef struct zswap_event {
  u32 pid;
  u32 tgid;
  u64 start_ts;
  u64 end_ts;
} zswap_event_t;

KERMIT_OUTPUT(zswap_store_events, zswap_event_t);
KERMIT_OUTPUT(zswap_load_events, zswap_event_t);
KERMIT_OUTPUT(zswap_invalidate_events, zswap_event_t);

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
  zswap_event_t* data = KERMIT_RESERVE(zswap_store_events, zswap_event_t, 1);
  if (data) {
    data->pid = (u32)(id);
    data->tgid = (u32)(id >> 32);
    data->start_ts = *start_ts;
    data->end_ts = bpf_ktime_get_ns();
    KERMIT_SUBMIT(zswap_store_events, ctx, data);
  }
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
  zswap_event_t* data = KERMIT_RESERVE(zswap_load_events, zswap_event_t, 1);
  if (data) {
    data->pid = (u32)(id);
    data->tgid = (u32)(id >> 32);
    data->start_ts = *start_ts;
    data->end_ts = bpf_ktime_get_ns();
    KERMIT_SUBMIT(zswap_load_events, ctx, data);
  }
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
  zswap_event_t* data = KERMIT_RESERVE(zswap_invalidate_events, zswap_event_t, 1);
  if (data) {
    data->pid = (u32)(id);
    data->tgid = (u32)(id >> 32);
    data->start_ts = *start_ts;
    data->end_ts = bpf_ktime_get_ns();
    KERMIT_SUBMIT(zswap_invalidate_events, ctx, data);
  }
  invalidates.delete(&id);
  return 0;
}
