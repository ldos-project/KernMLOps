#include <linux/ptrace.h>
#include <uapi/linux/bpf_perf_event.h>

typedef struct tlb_perf_event {
  u64 ts_uptime_us;
  u32 count;
  u64 enabled_time_us;
  u64 running_time_us;
} tlb_perf_event_t;

BPF_PERF_OUTPUT(dtlb_misses);
BPF_PERF_OUTPUT(itlb_misses);

static inline __attribute__((always_inline)) void fill_counter_data(
    struct tlb_perf_event* data, struct bpf_perf_event_value* value_buf) {
  u64 ts = bpf_ktime_get_ns();
  data->ts_uptime_us = ts / 1000;
  data->count = value_buf->counter;
  data->enabled_time_us = value_buf->enabled / 1000;
  data->running_time_us = value_buf->running / 1000;
}

int on_dtlb_cache_miss(struct bpf_perf_event_data* ctx) {
  struct bpf_perf_event_value value_buf;
  if (bpf_perf_prog_read_value(ctx, (void*)&value_buf, sizeof(struct bpf_perf_event_value))) {
    return 0;
  }

  struct tlb_perf_event data;
  __builtin_memset(&data, 0, sizeof(data));
  fill_counter_data(&data, &value_buf);
  dtlb_misses.perf_submit(ctx, &data, sizeof(data));

  return 0;
}

int on_itlb_cache_miss(struct bpf_perf_event_data* ctx) {
  struct bpf_perf_event_value value_buf;
  if (bpf_perf_prog_read_value(ctx, (void*)&value_buf, sizeof(struct bpf_perf_event_value))) {
    return 0;
  }

  struct tlb_perf_event data;
  __builtin_memset(&data, 0, sizeof(data));
  fill_counter_data(&data, &value_buf);
  itlb_misses.perf_submit(ctx, &data, sizeof(data));

  return 0;
}
