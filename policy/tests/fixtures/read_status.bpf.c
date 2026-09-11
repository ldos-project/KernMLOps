#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct policy_event {
    __u64 timestamp_ns;
    __s8 tag;
    __u16 cpu;
    __s32 syscall_nr;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4096);
} events SEC(".maps");

SEC("raw_tp/sys_enter")
int capture_syscall(struct bpf_raw_tracepoint_args *ctx)
{
    struct policy_event event = {0};

    event.timestamp_ns = bpf_ktime_get_ns();
    event.tag = -7;
    event.cpu = (__u16)bpf_get_smp_processor_id();
    /* The status controls submission even when syscall_nr is not selected. */
    if (bpf_probe_read_kernel(&event.syscall_nr, sizeof(event.syscall_nr),
                              (const void *)ctx->args[0]) != 0)
        return 0;
    bpf_ringbuf_output(&events, &event, sizeof(event), 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
