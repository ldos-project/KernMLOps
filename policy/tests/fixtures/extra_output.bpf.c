#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct policy_event {
    __u64 timestamp_ns;
    __u32 syscall_nr;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4096);
} events SEC(".maps");

#ifdef NOINLINE
static __noinline void emit_extra(struct policy_event *event)
#else
static __always_inline void emit_extra(struct policy_event *event)
#endif
{
    bpf_ringbuf_output(&events, event, sizeof(*event), 0);
}

SEC("raw_tp/sys_enter")
int capture_syscall(struct bpf_raw_tracepoint_args *ctx)
{
    struct policy_event event = {0};

    event.timestamp_ns = bpf_ktime_get_ns();
    event.syscall_nr = (__u32)ctx->args[1];
    emit_extra(&event);
    bpf_ringbuf_output(&events, &event, sizeof(event), 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
