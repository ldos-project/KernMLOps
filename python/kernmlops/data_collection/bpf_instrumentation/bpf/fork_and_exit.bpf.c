#include <linux/sched.h>

#include "kermit.h"

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

KERMIT_OUTPUT(copy_task_events, start_data_t);
KERMIT_OUTPUT(release_task_events, stop_data_t);

int kretprobe_copy_process(struct pt_regs* ctx) {
  struct task_struct* task;
  if (IS_ERR(task = (struct task_struct*)PT_REGS_RC(ctx)))
    return 0;
  start_data_t* data = KERMIT_RESERVE(copy_task_events, start_data_t, 1);
  if (!data)
    return 0;
  __builtin_memset(data->buff, 0, sizeof(data->buff));
  bpf_get_current_comm(&data->buff, sizeof(data->buff));
  data->ts = bpf_ktime_get_ns();
  data->pid = task->pid;
  data->tgid = task->tgid;
  KERMIT_SUBMIT(copy_task_events, ctx, data);
  return 0;
}

int kprobe_do_exit(struct pt_regs* ctx, long code) {
  struct task_struct* task = (struct task_struct*)bpf_get_current_task();
  stop_data_t* data = KERMIT_RESERVE(release_task_events, stop_data_t, 1);
  if (!data)
    return 0;
  data->ts = bpf_ktime_get_ns();
  data->pid = task->pid;
  data->tgid = task->tgid;
  KERMIT_SUBMIT(release_task_events, ctx, data);
  return 0;
}

typedef start_data_t exec_data_t;

KERMIT_OUTPUT(exec_events, exec_data_t);

int kretprobe_exec(struct pt_regs* ctx) {
  if (PT_REGS_RC(ctx) != 0)
    return 0;

  exec_data_t* data = KERMIT_RESERVE(exec_events, exec_data_t, 1);
  if (!data)
    return 0;
  __builtin_memset(data->buff, 0, sizeof(data->buff));
  bpf_get_current_comm(data->buff, sizeof(data->buff));
  data->pid = bpf_get_current_pid_tgid() >> 32;
  data->tgid = (u32)bpf_get_current_pid_tgid();
  data->ts = bpf_ktime_get_ns();
  KERMIT_SUBMIT(exec_events, ctx, data);

  return 0;
}
