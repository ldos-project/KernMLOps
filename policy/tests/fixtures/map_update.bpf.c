#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct policy_event {
    __u32 syscall_nr;
    __u64 pid_tgid;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4096);
} events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} counts SEC(".maps");

SEC("raw_tp/sys_enter")
int capture_syscall(struct bpf_raw_tracepoint_args *ctx)
{
    struct policy_event event = {0};
    __u32 key = 0;

    event.syscall_nr = (__u32)ctx->args[1];
    event.pid_tgid = bpf_get_current_pid_tgid();
    /* The omitted field is still needed by this externally visible effect. */
    bpf_map_update_elem(&counts, &key, &event.pid_tgid, BPF_ANY);
    bpf_ringbuf_output(&events, &event, sizeof(event), 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
