#include "bloat.h"
#include "../vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, u32);
  __uint(map_flags, BPF_F_MMAPABLE);
} singular_cache_array SEC(".maps");

const u32 zero_index = 0;

#define CHECK_TGID_RETURN(x)                                                  \
  do {                                                                        \
    u32* val = (u32*)bpf_map_lookup_elem(&singular_cache_array, &zero_index); \
    if (!val)                                                                 \
      return 0;                                                               \
    u32 useful = *val;                                                        \
    if (useful == 0 || useful == (x))                                         \
      return 0;                                                               \
  } while (0)

typedef struct rss_data {
  u64 ts;
  u64 count;
  u64 valid;
} rss_data_t;

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, u64);
  __uint(map_flags, BPF_F_MMAPABLE);
} rss_head_idx SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, BUFFER_ENTRIES);
  __type(key, u32);
  __type(value, rss_data_t);
  __uint(map_flags, BPF_F_MMAPABLE);
} rss_buffer SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 32768);
  __type(key, u32);
  __type(value, u64);
} rss_hash SEC(".maps");

SEC("raw_tracepoint/rss_stat")
int raw_trace_rss_stat(struct bpf_raw_tracepoint_args* ctx) {
  u64 ts = bpf_ktime_get_ns();
  u32 pid = bpf_get_current_pid_tgid();
  struct mm_struct* mm = (struct mm_struct*)ctx->args[0];
  struct task_struct* task;
  if (!bpf_probe_read_kernel(&task, sizeof(task), &mm->owner))
    return 0;
  u32 other_tgid = 0;
  if (bpf_probe_read_kernel(&other_tgid, sizeof(u32), &task->tgid))
    return 0;

  CHECK_TGID_RETURN(other_tgid);

  bpf_map_update_elem(&rss_hash, &pid, &ts, BPF_ANY);
  return 0;
}

static rss_data_t* rss_reserve_buffer_space(u32 base_offset, u32 header_offset, u32 buff_size) {
  u64* head_idx_ptr = bpf_map_lookup_elem(&rss_head_idx, &header_offset);
  if (!head_idx_ptr)
    return NULL;
  u32 index = (u32)__atomic_fetch_add(head_idx_ptr, 1, __ATOMIC_ACQUIRE);
  index = (u32)(index % buff_size) + base_offset;
  rss_data_t* data = bpf_map_lookup_elem(&rss_buffer, &index);
  if (!data)
    return data;
  u64 blah;
  u64 val = 0;
  __atomic_exchange(&data->valid, &val, &blah, __ATOMIC_ACQUIRE);
  return data;
}

static void submit_data(rss_data_t* data) {
  u64 blah;
  u64 val = 1;
  __atomic_exchange(&data->valid, &val, &blah, __ATOMIC_RELEASE);
}

SEC("tracepoint/kmem/rss_stat")
int trace_rss_stat(struct trace_event_raw_rss_stat* args) {
  u32 pid = bpf_get_current_pid_tgid();
  u64 ts;
  u64* tsptr = (u64*)bpf_map_lookup_elem(&rss_hash, &pid);
  if (tsptr == NULL)
    return 0;
  u64 count = args->size >> 12;
  rss_data_t* data = rss_reserve_buffer_space(0, 0, BUFFER_ENTRIES);
  if (!data)
    return 0;
  data->ts = *tsptr;
  data->count = count;
  submit_data(data);
  return 0;
}

#define CPUS 64

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, rss_data_t);
  __uint(map_flags, BPF_F_MMAPABLE);
} dtlb_header SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, BUFFER_ENTRIES);
  __type(key, u32);
  __type(value, rss_data_t);
  __uint(map_flags, BPF_F_MMAPABLE);
} dtlb_buffer SEC(".maps");

static rss_data_t* perf_reserve_buffer_space(u32 base_offset, u32 header_offset, u32 buff_size) {
  u64* head_idx_ptr = bpf_map_lookup_elem(&dtlb_header, &header_offset);
  if (!head_idx_ptr)
    return NULL;
  u32 index = (u32)__atomic_fetch_add(head_idx_ptr, 1, __ATOMIC_ACQUIRE);
  index = (u32)(index % buff_size) + base_offset;
  rss_data_t* data = bpf_map_lookup_elem(&dtlb_buffer, &index);
  if (!data)
    return data;
  u64 blah;
  u64 val = 0;
  __atomic_exchange(&data->valid, &val, &blah, __ATOMIC_ACQUIRE);
  return data;
}

SEC("perf_event")
int dtlb_miss_rate(struct bpf_perf_event_data* ctx) {
  u32 pid = bpf_get_current_pid_tgid();
  CHECK_TGID_RETURN(pid);
  struct bpf_perf_event_value value_buf;
  u64 ts = bpf_ktime_get_ns();
  if (bpf_perf_prog_read_value(ctx, (void*)&value_buf, sizeof(struct bpf_perf_event_value))) {
    return 0;
  }
  rss_data_t* data = perf_reserve_buffer_space(0, 0, BUFFER_ENTRIES);
  if (!data)
    return 0;
  data->ts = ts;
  data->count = value_buf.counter;
  submit_data(data);
  return 0;
}
