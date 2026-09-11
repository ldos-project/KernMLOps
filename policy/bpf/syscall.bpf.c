// SPDX-License-Identifier: GPL-2.0-only
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct policy_event {
    __u64 timestamp_ns;
    __u64 pid_tgid;
    __u32 syscall_nr;
    __u32 doubled_syscall_nr;
    __u64 checksum;
};

_Static_assert(sizeof(struct policy_event) == 32, "policy_event must be 32 bytes");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 18);
} events SEC(".maps");

SEC("raw_tp/sys_enter")
int capture_syscall(struct bpf_raw_tracepoint_args *ctx)
{
    struct policy_event event = {0};
    __u64 checksum;

    event.timestamp_ns = bpf_ktime_get_ns();
    event.pid_tgid = bpf_get_current_pid_tgid();
    bpf_probe_read_kernel(&event.syscall_nr, sizeof(event.syscall_nr),
                          &ctx->args[1]);
    event.doubled_syscall_nr = event.syscall_nr * 2;

    checksum = bpf_get_current_uid_gid();
    /* Keep a loop-carried dependence confined to the unselected checksum. */
#pragma clang loop unroll(disable)
    for (__u32 i = 0; i < 8; i++)
        checksum = (checksum ^ (checksum >> 13)) * 0x9e3779b185ebca87ULL + i;
    event.checksum = checksum;

    bpf_ringbuf_output(&events, &event, sizeof(event), 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
