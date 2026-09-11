#include <linux/mm_types.h>
#include <linux/sched.h>

#include "kermit.h"

typedef struct unmap_range_output {
  u32 tgid;
  u64 ts_ns;
  u64 start;
  u64 end;
  int huge;
} unmap_range_output_t;

KERMIT_OUTPUT(unmap_range_output, unmap_range_output_t);

int kprobe__unmap_page_range(struct pt_regs* ctx, struct mm_gather* tlb, struct vm_area_struct* vma,
                             unsigned long start, unsigned long end, struct zap_details* details) {
  unmap_range_output_t* data = KERMIT_RESERVE(unmap_range_output, unmap_range_output_t, 1);
  if (!data)
    return 0;
  data->tgid = vma->vm_mm->owner->tgid;
  data->ts_ns = bpf_ktime_get_ns();
  data->start = start;
  data->end = end;
  data->huge = false;
  KERMIT_SUBMIT(unmap_range_output, ctx, data);
  return 0;
}

int kprobe__unmap_hugepage_range(struct pt_regs* ctx, struct mm_gather* tlb,
                                 struct vm_area_struct* vma, unsigned long start, unsigned long end,
                                 struct page* ref_page, zap_flags_t zap_flags) {
  unmap_range_output_t* data = KERMIT_RESERVE(unmap_range_output, unmap_range_output_t, 1);
  if (!data)
    return 0;
  data->tgid = vma->vm_mm->owner->tgid;
  data->ts_ns = bpf_ktime_get_ns();
  data->start = start;
  data->end = end;
  data->huge = true;
  KERMIT_SUBMIT(unmap_range_output, ctx, data);
  return 0;
}
