#include <linux/sched.h>

typedef struct start_data {
  u32 pid;
  u32 tgid;
  u64 ts;
  char buff[TASK_COMM_LEN];
} start_data_t;

typedef struct stop_data {
  u32 pid;
  u32 tgid;
  u64 ts;
} stop_data_t;

BPF_PERF_OUTPUT(copy_task_events);
BPF_PERF_OUTPUT(release_task_events);
BPF_PERF_OUTPUT(exec_events);

int kretprobe_copy_process(struct pt_regs* ctx) {
  struct task_struct* task;
  if (IS_ERR(task = (struct task_struct*)PT_REGS_RC(ctx)))
    return 0;
  start_data_t data;
  bpf_get_current_comm(&data.buff, sizeof(data.buff));
  data.ts = bpf_ktime_get_ns();
  data.pid = task->pid;
  data.tgid = task->tgid;
  copy_task_events.perf_submit(ctx, &data, sizeof(data));
  return 0;
}

int kprobe_release_task(struct pt_regs* ctx, struct task_struct* task) {
  stop_data_t data;
  data.ts = bpf_ktime_get_ns();
  data.pid = task->pid;
  data.tgid = task->tgid;
  release_task_events.perf_submit(ctx, &data, sizeof(data));
  return 0;
}

typedef start_data_t exec_data_t;

int kretprobe_exec(struct pt_regs* ctx) {
  if (PT_REGS_RC(ctx) != 0)
    return 0;

  exec_data_t data;
  bpf_get_current_comm(data.buff, sizeof(data.buff));
  data.pid = bpf_get_current_pid_tgid() >> 32;
  data.tgid = (u32)bpf_get_current_pid_tgid();
  data.ts = bpf_ktime_get_ns();
  exec_events.perf_submit(ctx, &data, sizeof(data));

  return 0;
}
