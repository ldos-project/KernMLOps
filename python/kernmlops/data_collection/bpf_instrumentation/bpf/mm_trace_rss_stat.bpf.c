#include <linux/mm_types.h>
#include <linux/sched.h>

#include "kermit.h"

typedef struct rss_stat_output {
  u32 pid;
  u32 tgid;
  u64 ts;
  int member;
  u64 counter_value;
} rss_stat_output_t;

KERMIT_OUTPUT(rss_stat_output, rss_stat_output_t);

BPF_HASH(rss_stat_hash, u32, rss_stat_output_t, 32768);

// ASSUMPTION, raw runs before non-raw (should be correct)

#define PAGE_SZ 12

RAW_TRACEPOINT_PROBE(rss_stat) {
  rss_stat_output_t stack_data;
  u32 pid = bpf_get_current_pid_tgid();
  memset((void*)&stack_data, 0, sizeof(stack_data));

  struct mm_struct* mm = (struct mm_struct*)ctx->args[0];
  stack_data.pid = mm->owner->pid;
  stack_data.tgid = mm->owner->tgid;

  rss_stat_hash.insert(&pid, &stack_data);
  return 0;
}

TRACEPOINT_PROBE(kmem, rss_stat) {
  rss_stat_output_t* data;
  u32 pid = bpf_get_current_pid_tgid();
  if ((data = rss_stat_hash.lookup(&pid)) == NULL) {
    return 0;
  }

  rss_stat_output_t* output = KERMIT_RESERVE(rss_stat_output, rss_stat_output_t, 1);
  if (output) {
    output->pid = data->pid;
    output->tgid = data->tgid;
    output->member = args->member;
    output->counter_value = (args->size) >> PAGE_SZ;
    output->ts = bpf_ktime_get_ns();
    KERMIT_SUBMIT(rss_stat_output, args, output);
  }
  rss_stat_hash.delete(&pid);
  return 0;
}
