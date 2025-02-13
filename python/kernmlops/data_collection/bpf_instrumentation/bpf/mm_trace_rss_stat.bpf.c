#include <linux/minmax.h>
#include <linux/mm_types.h>

typedef struct rss_stat_output {
  u32 pid;
  u32 tgid;
  u64 ts;
  s64 file;
  s64 anon;
  s64 swap;
  s64 shmem;
} rss_stat_output_t;

BPF_PERF_OUTPUT(rss_stat_output);

RAW_TRACEPOINT_PROBE(rss_stat) {
  struct mm_struct* mm = (struct mm_struct*)ctx->args[0];
  int member = (int)ctx->args[0];
  rss_stat_output_t data;
  memset((void*)&data, 0, sizeof(data));
  data.pid = mm->owner->pid;
  data.tgid = mm->owner->tgid;
  data.ts = bpf_ktime_get_ns();
  data.file = mm->rss_stat[MM_FILEPAGES].count;
  data.anon = mm->rss_stat[MM_ANONPAGES].count;
  data.swap = mm->rss_stat[MM_SWAPENTS].count;
  data.shmem = mm->rss_stat[MM_SHMEMPAGES].count;
  if (data.file < 0)
    data.file = 0;
  if (data.anon < 0)
    data.anon = 0;
  if (data.swap < 0)
    data.swap = 0;
  if (data.shmem < 0)
    data.shmem = 0;
  rss_stat_output.perf_submit(ctx, &data, sizeof(data));
  return 0;
}
