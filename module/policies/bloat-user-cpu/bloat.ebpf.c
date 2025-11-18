#include "bloat.h"
#include "../vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

#define MPMO_QUEUE_DEFINE(x, data)     \
  struct {                             \
    __uint(type, BPF_MAP_TYPE_ARRAY);  \
    __uint(max_entries, 1);            \
    __type(key, u32);                  \
    __type(value, u64);                \
    __uint(map_flags, BPF_F_MMAPABLE); \
  } mpmo_head_##x SEC(".maps");        \
                                       \
  struct {                             \
    __uint(type, BPF_MAP_TYPE_ARRAY);  \
    __uint(max_entries, BUFFER_SIZE);  \
    __type(key, u32);                  \
    __type(value, data);               \
    __uint(map_flags, BPF_F_MMAPABLE); \
  } mpmo_buff_##x SEC(".maps")

// ENUM FOR MPSO
u64 writing_val = 3;
u64 reading_val = 2;
u64 readable_val = 1;
u64 invalid_val = 0;
const u32 zero_index = 0;

#define RESERVE_BUFFER_SPACE(tag, type, reserve_attempt)                                    \
  ({                                                                                        \
    u64* head_idx_ptr = bpf_map_lookup_elem(&mpmo_head_##tag, &zero_index);                 \
    type* location = NULL;                                                                  \
    bool ready = false;                                                                     \
    if (head_idx_ptr) {                                                                     \
      u32 index = (u32)__atomic_fetch_add(head_idx_ptr, 1, __ATOMIC_RELAXED);               \
      index = index % BUFFER_SIZE;                                                          \
      location = bpf_map_lookup_elem(&mpmo_buff_##tag, &index);                             \
      if (location) {                                                                       \
        for (int i = 0; i < reserve_attempt && !ready; i++) {                               \
          u64 valid = location->valid;                                                      \
          if (valid != writing_val) {                                                       \
            ready = __atomic_compare_exchange(&location->valid, &valid, &writing_val, true, \
                                              __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);          \
          }                                                                                 \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
    location;                                                                               \
  })

#define SUBMIT_DATA(ptr)                                                    \
  ({                                                                        \
    u64 blah;                                                               \
    __atomic_exchange(&ptr->valid, &readable_val, &blah, __ATOMIC_RELEASE); \
  })

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, u32);
  __uint(map_flags, BPF_F_MMAPABLE);
} singular_cache_array SEC(".maps");

#define CHECK_TGID_RETURN(x)                                                  \
  do {                                                                        \
    u32* val = (u32*)bpf_map_lookup_elem(&singular_cache_array, &zero_index); \
    if (!val)                                                                 \
      return 0;                                                               \
    u32 useful = *val;                                                        \
    if (useful != (x))                                                        \
      return 0;                                                               \
  } while (0)

typedef struct rss_data {
  u64 ts;
  u64 count;
  u64 valid;
} rss_data_t;

MPMO_QUEUE_DEFINE(rss, rss_data_t);

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 32768);
  __type(key, u32);
  __type(value, u64);
} rss_hash SEC(".maps");

SEC("raw_tracepoint/rss_stat")
int raw_trace_rss_stat(struct bpf_raw_tracepoint_args* ctx) {
  static const char fmt[] = "raw_trace: %ld\n";
  u64 ts = bpf_ktime_get_ns();
  u32 pid = bpf_get_current_pid_tgid();
  struct mm_struct* mm = (struct mm_struct*)ctx->args[0];
  struct task_struct* task;
  if (bpf_probe_read_kernel(&task, sizeof(struct task_struct*), &mm->owner)) {
    return 0;
  }
  s32 other_tgid = 0;
  if (bpf_probe_read_kernel(&other_tgid, sizeof(u32), &task->tgid))
    return 0;

  CHECK_TGID_RETURN(other_tgid);
  u64* countptr = (u64*)bpf_map_lookup_elem(&rss_hash, &pid);
  if (countptr) {
    rss_data_t* data = RESERVE_BUFFER_SPACE(rss, rss_data_t, 1);
    if (!data)
      return 0;
    data->ts = ts;
    data->count = *countptr;
    SUBMIT_DATA(data);
  }
  bpf_map_delete_elem(&rss_hash, &pid);
  return 0;
}

SEC("tracepoint/kmem/rss_stat")
int trace_rss_stat(struct trace_event_raw_rss_stat* args) {
  u32 pid = bpf_get_current_pid_tgid();
  u64 count = args->size >> 12;
  bpf_map_update_elem(&rss_hash, &pid, &count, BPF_ANY);
  return 0;
}

MPMO_QUEUE_DEFINE(dtlb, rss_data_t);

SEC("perf_event")
int dtlb_miss_rate(struct bpf_perf_event_data* ctx) {
  struct bpf_perf_event_value value_buf;
  u64 ts = bpf_ktime_get_ns();
  int cmp = bpf_perf_prog_read_value(ctx, (void*)&value_buf, sizeof(struct bpf_perf_event_value));
  if (cmp < 0)
    return 0;
  rss_data_t* data = RESERVE_BUFFER_SPACE(dtlb, rss_data_t, 1);
  if (!data)
    return 0;
  data->ts = ts;
  data->count = value_buf.counter;
  SUBMIT_DATA(data);
  return 0;
}
