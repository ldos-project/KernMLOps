#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct policy_event {
    __u32 syscall_nr;
    __u8 payload[4];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4096);
} events SEC(".maps");

SEC("raw_tp/sys_enter")
int capture_syscall(struct bpf_raw_tracepoint_args *ctx)
{
    struct policy_event event = {0};

    event.syscall_nr = (__u32)ctx->args[1];
    event.payload[0] = (__u8)ctx->args[0];
    bpf_ringbuf_output(&events, &event, sizeof(event), 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
