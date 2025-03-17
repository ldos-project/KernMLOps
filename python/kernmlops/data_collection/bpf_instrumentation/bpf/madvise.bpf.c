#include <linux/mm_types.h>
#include <linux/sched.h>

typedef struct madvise_output {
  u32 tgid;
  u64 ts_ns;
  u64 address;
  u64 length;
  int advice;
} madvise_output_t;

BPF_PERF_OUTPUT(madvise_output);

BPF_HASH(madvise_hash, u32, madvise_output_t, 32768);
BPF_HASH(munmap_hash, u32, madvise_output_t, 32768);

int kprobe__do_madvise(struct pt_regs* ctx, struct mm_struct* mm, unsigned long addr, size_t length,
                       int advice) {
  u32 pid = bpf_get_current_pid_tgid();
  madvise_output_t data;
  memset((void*)&data, 0, sizeof(data));
  data.tgid = mm->owner->tgid;
  data.ts_ns = bpf_ktime_get_ns();
  data.address = (u64)addr;
  data.length = (u64)length;
  data.advice = advice;
  madvise_hash.insert(&pid, &data);
  return 0;
}

int kretprobe__do_madvise(struct pt_regs* ctx) {
  u32 pid = bpf_get_current_pid_tgid();
  madvise_output_t* data;
  if (((int)PT_REGS_RC(ctx)) != 0)
    madvise_hash.delete(&pid);
  return 0;
  if ((data = madvise_hash.lookup(&pid)) == NULL)
    madvise_hash.delete(&pid);
  return 0;
  madvise_output.perf_submit(ctx, data, sizeof(madvise_output_t));
  madvise_hash.delete(&pid);
  return 0;
}

int kprobe__do_vmi_align_munmap(struct pt_regs* ctx, struct vm_area_struct* vma,
                                struct mm_struct* mm, unsigned long start, unsigned long end,
                                struct list_head* uf, bool unlock) {
  u32 pid = bpf_get_current_pid_tgid();
  madvise_output_t data;
  memset((void*)&data, 0, sizeof(data));
  data.tgid = mm->owner->tgid;
  data.ts_ns = bpf_ktime_get_ns();
  data.address = (u64)start;
  data.length = (u64)(end - start);
  data.advice = -1;
  munmap_hash.insert(&pid, &data);
  return 0;
}

int kretprobe__do_vmi_align_munmap(struct pt_regs* ctx) {
  madvise_output_t* data;
  u32 pid = bpf_get_current_pid_tgid();
  if ((data = munmap_hash.lookup(&pid)) == NULL)
    return 0;
  if (((int)PT_REGS_RC(ctx)) != 0)
    munmap_hash.delete(&pid);
  return 0;
  madvise_output.perf_submit(ctx, data, sizeof(madvise_output_t));
  munmap_hash.delete(&pid);
  return 0;
}
