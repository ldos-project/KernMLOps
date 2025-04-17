#include "../vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

const char redis_comm[] = "redis-server";
const char pr_comm[] = "pr";

typedef struct singular_cache {
  u32 valid;
  u32 tgid;
} singular_cache_t;

struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, singular_cache_t);
  __uint(map_flags, BPF_F_MMAPABLE);
} singular_cache_array SEC(".maps");

bool is_expected_comm(const char* comm) {
  return (!bpf_strncmp(redis_comm, sizeof(redis_comm), comm)) ||
         (!bpf_strncmp(pr_comm, sizeof(pr_comm), comm));
}

const int zero_index = 0;

bool check_singular_array(u32 tgid) {
  singular_cache_t* val = bpf_map_lookup_elem(&singular_cache_array, &zero_index);
  if (val) {
    if (__atomic_fetch(&val->valid, __ATOMIC_ACQUIRE)) {
      return val->tgid == tgid;
    } else {
      char comm[13];
      bpf_get_current_comm(comm, 13);
      if (is_expected_comm(comm)) {
        val->tgid = tgid;
        __atomic_set(&val->valid, 1, __ATOMIC_RELEASE);
        return true;
      }
      return false;
    }
  }
}

typedef struct rss_data {
  u64 ts;
  u64 convert_data;
} rss_data_t;

SEC("tracepoint/kmem/rss_stat")
