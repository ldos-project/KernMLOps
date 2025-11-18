#include <asm-generic/vmlinux.lds.h>
#include <linux/mm.h>
#include <linux/mm_econ.h>
#include <linux/sched.h>
#include <linux/sched/loadavg.h>
#include <uapi/linux/ptrace.h>

struct cbmm_eager_paging {
  /* Cost related info */
  struct mm_cost_delta* cost;
  u64 freq_cycles;

  /* Benefit Related info */
  u64 greatest_range_benefit;
};

struct cbmm_eager_paging_inputs {
  u64 freq_cycles;
  u64 greatest_range_benefit;
  int decision;
};

struct cbmm_async_prezeroing {
  struct mm_cost_delta* cost;
  struct mm_action* action;
  unsigned long* load;
  u64 load_info;
  u64 daemon_cost;
  u64 prezero_n;
  u64 nfree;
  u64 critical_section_cost;
  u64 zeroing_per_page_cost;
  u64 recent_used;
};

struct cbmm_async_prezeroing_inputs {
  u64 load;
  u64 daemon_cost;
  u64 prezero_n;
  u64 nfree;
  u64 critical_section_cost;
  u64 zeroing_per_page_cost;
  u64 recent_used;
  int decision;
};

struct cbmm_action {
  int action;
  struct cbmm_eager_paging eager;
  struct cbmm_async_prezeroing prezero;
};

BPF_HASH(cbmm_action_hash, u64, struct cbmm_action, 1024);

BPF_PERF_OUTPUT(cbmm_eager);
BPF_PERF_OUTPUT(cbmm_prezero);

static void insert_mm_estimate_changes(int action) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action stored_action;
  memset(&stored_action, 0, sizeof(struct cbmm_action));
  stored_action.action = action;
  cbmm_action_hash.lookup_or_try_init(&tgid_pid, &stored_action);
}

int kprobe__mm_estimate_changes(struct pt_regs* ctx, struct mm_action* action,
                                struct mm_cost_delta* cost) {
  switch (action->action) {
    case MM_ACTION_EAGER_PAGING:
    case MM_ACTION_RUN_PREZEROING:
      insert_mm_estimate_changes(action->action);
      break;
    default:
      break;
  }
  return 0;
}

static void mm_decide_push_eager(struct pt_regs* ctx, int decision, struct cbmm_action action) {
  struct cbmm_eager_paging_inputs inputs;
  memset(&inputs, 0, sizeof(inputs));
  inputs.decision = decision;
  inputs.freq_cycles = action.eager.freq_cycles;
  inputs.greatest_range_benefit = action.eager.greatest_range_benefit;
  cbmm_eager.perf_submit(ctx, &inputs, sizeof(inputs));
}

static void mm_decide_push_prezero(struct pt_regs* ctx, int decision, struct cbmm_action action) {
  struct cbmm_async_prezeroing_inputs inputs;
  memset(&inputs, 0, sizeof(inputs));
  inputs.decision = decision;
  inputs.load = action.prezero.load_info;
  inputs.daemon_cost = action.prezero.daemon_cost;
  inputs.prezero_n = action.prezero.prezero_n;
  // inputs.nfree = action.prezero.nfree;
  // inputs.critical_section_cost = action.prezero.critical_section_cost;
  inputs.critical_section_cost = 150 * 2;
  inputs.nfree = 10 * 3000 * 1000 / inputs.critical_section_cost;
  inputs.zeroing_per_page_cost = action.prezero.zeroing_per_page_cost;
  inputs.recent_used = action.prezero.recent_used;
  cbmm_prezero.perf_submit(ctx, &inputs, sizeof(inputs));
}

int kretprobe__mm_decide(struct pt_regs* ctx) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action* storage_action = NULL;
  if (!(storage_action = cbmm_action_hash.lookup(&tgid_pid)))
    return 0;
  switch (storage_action->action) {
    case MM_ACTION_EAGER_PAGING:
      mm_decide_push_eager(ctx, PT_REGS_RC(ctx), *storage_action);
      break;
    case MM_ACTION_RUN_PREZEROING:
      mm_decide_push_prezero(ctx, PT_REGS_RC(ctx), *storage_action);
      break;
    default:
      break;
  }
  cbmm_action_hash.delete(&tgid_pid);
  return 0;
}

int kprobe__mm_estimate_eager_page_cost_benefit(struct pt_regs* ctx, struct mm_action* action,
                                                struct mm_cost_delta* cost) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action* storage_action = NULL;
  if (!(storage_action = cbmm_action_hash.lookup(&tgid_pid)))
    return 0;
  storage_action->eager.cost = cost;
  return 0;
}

int kretprobe__mm_estimate_eager_page_cost_benefit(struct pt_regs* ctx) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action* storage_action = NULL;
  if (!(storage_action = cbmm_action_hash.lookup(&tgid_pid)))
    return 0;
  storage_action->eager.freq_cycles = storage_action->eager.cost->cost;
  storage_action->eager.greatest_range_benefit = storage_action->eager.cost->benefit;
  return 0;
}

int kprobe__mm_estimate_daemon_cost(struct pt_regs* ctx, struct mm_action* action,
                                    struct mm_cost_delta* cost) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action* storage_action = NULL;
  if (!(storage_action = cbmm_action_hash.lookup(&tgid_pid)))
    return 0;
  storage_action->prezero.prezero_n = action->prezero_n;
  storage_action->prezero.daemon_cost = 1000000;
  return 0;
}

int kprobe__get_avenrun(struct pt_regs* ctx, unsigned long* loads, unsigned long offset,
                        int shift) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action* storage_action = NULL;
  if (!(storage_action = cbmm_action_hash.lookup(&tgid_pid)))
    return 0;
  storage_action->prezero.load = loads;
  return 0;
}

int kretprobe__get_avenrun(struct pt_regs* ctx) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action* storage_action = NULL;
  if (!(storage_action = cbmm_action_hash.lookup(&tgid_pid)))
    return 0;
  unsigned long blah = storage_action->prezero.load[0];
  unsigned long load = LOAD_INT(blah);
  storage_action->prezero.load_info = (u64)load;
  return 0;
}

/**
int kprobe__mm_estimate_async_prezeroing_lock_contention_cost(
    struct pt_regs* ctx,
    struct mm_action* action,
    struct mm_cost_delta* cost) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action* storage_action = NULL;
  if( ! (storage_action = cbmm_action_hash.lookup(&tgid_pid))) return 0;
  u64 critical_section_cost = 150 * 2;
  storage_action->prezero.critical_section_cost = critical_section_cost;
  storage_action->prezero.nfree = 10 * 3000 * 1000 /
    critical_section_cost;
  return 0;
}
**/

int kretprobe__mm_estimated_prezeroed_used(struct pt_regs* ctx) {
  u64 tgid_pid = bpf_get_current_pid_tgid();
  struct cbmm_action* storage_action = NULL;
  if (!(storage_action = cbmm_action_hash.lookup(&tgid_pid)))
    return 0;
  storage_action->prezero.zeroing_per_page_cost = 1000000;
  storage_action->prezero.recent_used = PT_REGS_RC(ctx);
  return 0;
}
