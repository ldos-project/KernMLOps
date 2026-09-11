// SPDX-License-Identifier: GPL-2.0-only
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>

struct task_struct {
    int pid;
    int tgid;
} __attribute__((preserve_access_index));

struct mm_struct {
    struct task_struct *owner;
} __attribute__((preserve_access_index));

struct trace_event_raw_rss_stat {
    int member;
    long size;
} __attribute__((preserve_access_index));

struct rss_owner {
    __u32 pid;
    __u32 tgid;
};

struct rss_event {
    __u64 timestamp_ns;
    __u64 count_pages;
    __u32 pid;
    __u32 tgid;
    __s32 member;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 32768);
    __type(key, __u32);
    __type(value, struct rss_owner);
} rss_owners SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} rss_events SEC(".maps");

/* 0: ring output failure; 1: broken bridge; 2: unavailable owner. */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 3);
    __type(key, __u32);
    __type(value, __u64);
} event_errors SEC(".maps");

static __always_inline void count_event_error(__u32 index)
{
    __u64 *counter = bpf_map_lookup_elem(&event_errors, &index);

    if (counter)
        __sync_fetch_and_add(counter, 1);
}

/* Paired delivery requires the raw tracepoint to run before rss_emit. */
SEC("raw_tp/rss_stat")
int rss_owner_capture(struct bpf_raw_tracepoint_args *ctx)
{
    struct mm_struct *mm = (struct mm_struct *)ctx->args[0];
    struct task_struct *owner = NULL;
    struct rss_owner entry = {0};
    __u32 tid = (__u32)bpf_get_current_pid_tgid();

    if (!BPF_CORE_READ_INTO(&owner, mm, owner) && owner &&
        !BPF_CORE_READ_INTO(&entry.tgid, owner, tgid)) {
        entry.pid = BPF_CORE_READ(owner, pid);
    }

    /* Refresh any unpaired startup capture instead of reusing its owner. */
    if (bpf_map_update_elem(&rss_owners, &tid, &entry, BPF_ANY)) {
        bpf_map_delete_elem(&rss_owners, &tid);
        count_event_error(1);
    }
    return 0;
}

SEC("tp/kmem/rss_stat")
int rss_emit(struct trace_event_raw_rss_stat *ctx)
{
    __u32 tid = (__u32)bpf_get_current_pid_tgid();
    struct rss_owner *owner = bpf_map_lookup_elem(&rss_owners, &tid);
    struct rss_event event;

    if (!owner) {
        count_event_error(1);
        return 0;
    }
    if (!owner->tgid) {
        count_event_error(2);
        bpf_map_delete_elem(&rss_owners, &tid);
        return 0;
    }

    __builtin_memset(&event, 0, sizeof(event));
    event.timestamp_ns = bpf_ktime_get_ns();
    event.count_pages = BPF_CORE_READ(ctx, size) >> 12;
    event.pid = owner->pid;
    event.tgid = owner->tgid;
    event.member = BPF_CORE_READ(ctx, member);

    if (bpf_ringbuf_output(&rss_events, &event, sizeof(event), 0))
        count_event_error(0);

    bpf_map_delete_elem(&rss_owners, &tid);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
