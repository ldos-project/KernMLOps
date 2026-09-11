#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>

struct task_struct {
    int pid;
    int tgid;
} __attribute__((preserve_access_index));

struct policy_event {
    __u32 pid;
    __u32 tgid;
    __u64 timestamp_ns;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4096);
} events SEC(".maps");

SEC("raw_tp/sys_enter")
int capture_syscall(struct bpf_raw_tracepoint_args *ctx)
{
    struct policy_event event = {0};
    struct task_struct *task = bpf_get_current_task_btf();

    event.pid = BPF_CORE_READ(task, pid);
    event.tgid = BPF_CORE_READ(task, tgid);
    event.timestamp_ns = bpf_ktime_get_ns();
    bpf_ringbuf_output(&events, &event, sizeof(event), 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
