// SPDX-License-Identifier: GPL-2.0-only
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct bpf_perf_event_data;

struct dtlb_event {
    __u64 timestamp_ns;
    __u64 misses;
    __u64 accesses;
    __u64 miss_enabled_ns;
    __u64 miss_running_ns;
    __u64 access_enabled_ns;
    __u64 access_running_ns;
    __u32 cpu;
    __u32 pid;
    __u32 tgid;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    /* The loader must resize this to the number of possible CPUs before load. */
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} miss_counters SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} dtlb_events SEC(".maps");

/* 0: ring output failure; 1: perf counter read failure. */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 2);
    __type(key, __u32);
    __type(value, __u64);
} event_errors SEC(".maps");

static __always_inline void count_event_error(__u32 index)
{
    __u64 *counter = bpf_map_lookup_elem(&event_errors, &index);

    if (counter)
        __sync_fetch_and_add(counter, 1);
}

/* The access group leader triggers samples; the map holds its miss members. */
SEC("perf_event")
int dtlb_sample(struct bpf_perf_event_data *ctx)
{
    struct bpf_perf_event_value a = {0};
    struct bpf_perf_event_value m = {0};
    struct dtlb_event event;
    __u32 cpu = bpf_get_smp_processor_id();
    __u64 pid_tgid;

    if (bpf_perf_prog_read_value(ctx, &a, sizeof(a)) ||
        bpf_perf_event_read_value(&miss_counters, cpu, &m, sizeof(m))) {
        count_event_error(1);
        return 0;
    }

    __builtin_memset(&event, 0, sizeof(event));
    event.timestamp_ns = bpf_ktime_get_ns();
    event.misses = m.counter;
    event.accesses = a.counter;
    event.miss_enabled_ns = m.enabled;
    event.miss_running_ns = m.running;
    event.access_enabled_ns = a.enabled;
    event.access_running_ns = a.running;
    event.cpu = cpu;
    pid_tgid = bpf_get_current_pid_tgid();
    event.pid = (__u32)pid_tgid;
    event.tgid = (__u32)(pid_tgid >> 32);

    if (bpf_ringbuf_output(&dtlb_events, &event, sizeof(event), 0))
        count_event_error(0);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
