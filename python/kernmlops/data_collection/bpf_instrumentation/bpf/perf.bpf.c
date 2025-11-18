#include <linux/ptrace.h>
#include <uapi/linux/bpf_perf_event.h>

typedef struct perf_event_data {
  u32 pid;
  u32 tgid;
  u64 ts_uptime_us;
  u32 count;
  u64 enabled_time_us;
  u64 running_time_us;
} tlb_perf_event_data_t;

// For handler code see file "python/kernmlops/data_collection/bpf_instrumentation/tlb_perf_hook.py"
