#!/usr/bin/env python3
import argparse
import signal
import sys
import time

from bcc import BPF

parser = argparse.ArgumentParser(description='Measure time spent in zswap functions')
parser.add_argument('pid', type=int, help='Process ID to monitor')
parser.add_argument('-t', '--time', type=int, default=0,
                    help='Duration to run the trace in seconds (0 means until Ctrl+C)')
args = parser.parse_args()

bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/time.h>

struct timing_data_t {
    u64 count;
    u64 total_ns;
    u64 max_ns;
};

struct start_data_t {
    u64 ts;
    u32 pid;
};

BPF_HASH(starts, u64, struct start_data_t);
BPF_HASH(zswap_store_stats, u32, struct timing_data_t);
BPF_HASH(zswap_load_stats, u32, struct timing_data_t);
BPF_HASH(zswap_invalidate_stats, u32, struct timing_data_t);

static inline bool filter_pid(u32 pid) {
    u32 target_pid = FILTER_PID;

    if (target_pid == 0)
        return true;

    return (pid == target_pid);
}

int trace_zswap_store_entry(struct pt_regs *ctx) {
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;

    if (!filter_pid(pid))
        return 0;

    struct start_data_t start = {};
    start.ts = bpf_ktime_get_ns();
    start.pid = pid;

    starts.update(&id, &start);
    return 0;
}

int trace_zswap_store_return(struct pt_regs *ctx) {
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;

    struct start_data_t *startp = starts.lookup(&id);
    if (startp == 0)
        return 0;

    if (!filter_pid(pid))
        return 0;

    u64 delta = bpf_ktime_get_ns() - startp->ts;

    struct timing_data_t *timing, zero = {};
    timing = zswap_store_stats.lookup_or_init(&pid, &zero);
    timing->count++;
    timing->total_ns += delta;
    if (delta > timing->max_ns)
        timing->max_ns = delta;

    starts.delete(&id);
    return 0;
}

int trace_zswap_load_entry(struct pt_regs *ctx) {
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;

    if (!filter_pid(pid))
        return 0;

    struct start_data_t start = {};
    start.ts = bpf_ktime_get_ns();
    start.pid = pid;

    starts.update(&id, &start);
    return 0;
}

int trace_zswap_load_return(struct pt_regs *ctx) {
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;

    struct start_data_t *startp = starts.lookup(&id);
    if (startp == 0)
        return 0;

    if (!filter_pid(pid))
        return 0;

    u64 delta = bpf_ktime_get_ns() - startp->ts;

    struct timing_data_t *timing, zero = {};
    timing = zswap_load_stats.lookup_or_init(&pid, &zero);
    timing->count++;
    timing->total_ns += delta;
    if (delta > timing->max_ns)
        timing->max_ns = delta;

    starts.delete(&id);
    return 0;
}

int trace_zswap_invalidate_entry(struct pt_regs *ctx) {
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;

    if (!filter_pid(pid))
        return 0;

    struct start_data_t start = {};
    start.ts = bpf_ktime_get_ns();
    start.pid = pid;

    starts.update(&id, &start);
    return 0;
}

int trace_zswap_invalidate_return(struct pt_regs *ctx) {
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;

    struct start_data_t *startp = starts.lookup(&id);
    if (startp == 0)
        return 0;

    if (!filter_pid(pid))
        return 0;

    u64 delta = bpf_ktime_get_ns() - startp->ts;

    struct timing_data_t *timing, zero = {};
    timing = zswap_invalidate_stats.lookup_or_init(&pid, &zero);
    timing->count++;
    timing->total_ns += delta;
    if (delta > timing->max_ns)
        timing->max_ns = delta;

    starts.delete(&id);
    return 0;
}
"""

bpf_text = bpf_text.replace('FILTER_PID', str(args.pid))

b = BPF(text=bpf_text)

b.attach_kprobe(event="zswap_store", fn_name="trace_zswap_store_entry")
b.attach_kretprobe(event="zswap_store", fn_name="trace_zswap_store_return")

b.attach_kprobe(event="zswap_load", fn_name="trace_zswap_load_entry")
b.attach_kretprobe(event="zswap_load", fn_name="trace_zswap_load_return")

b.attach_kprobe(event="zswap_invalidate", fn_name="trace_zswap_invalidate_entry")
b.attach_kretprobe(event="zswap_invalidate", fn_name="trace_zswap_invalidate_return")

print("Tracing zswap functions for PID %d... Press Ctrl+C to end." % args.pid)
print("%-8s %-20s %-10s %-16s %-16s" % ("PID", "FUNCTION", "CALLS", "TOTAL TIME (ms)", "AVG TIME (μs)"))

def handle_interrupt(signal, frame):
    print_stats()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

def print_stats():
    for k, v in b["zswap_store_stats"].items():
        pid = k.value
        calls = v.count
        total_ns = v.total_ns
        total_ms = total_ns / 1000000.0
        avg_us = 0 if calls == 0 else (total_ns / calls) / 1000.0
        print("%-8d %-20s %-10d %-16.2f %-16.2f" %
              (pid, "zswap_store", calls, total_ms, avg_us))

    for k, v in b["zswap_load_stats"].items():
        pid = k.value
        calls = v.count
        total_ns = v.total_ns
        total_ms = total_ns / 1000000.0
        avg_us = 0 if calls == 0 else (total_ns / calls) / 1000.0
        print("%-8d %-20s %-10d %-16.2f %-16.2f" %
              (pid, "zswap_load", calls, total_ms, avg_us))

    for k, v in b["zswap_invalidate_stats"].items():
        pid = k.value
        calls = v.count
        total_ns = v.total_ns
        total_ms = total_ns / 1000000.0
        avg_us = 0 if calls == 0 else (total_ns / calls) / 1000.0
        print("%-8d %-20s %-10d %-16.2f %-16.2f" %
              (pid, "zswap_invalidate", calls, total_ms, avg_us))

    total_time_ms = 0
    total_calls = 0

    for map_name in ["zswap_store_stats", "zswap_load_stats", "zswap_invalidate_stats"]:
        for k, v in b[map_name].items():
            if k.value == args.pid:
                total_time_ms += v.total_ns / 1000000.0
                total_calls += v.count

    print("\nSUMMARY FOR PID %d:" % args.pid)
    print("Total calls to zswap functions: %d" % total_calls)
    print("Total time spent in zswap: %.2f ms" % total_time_ms)

if args.time > 0:
    try:
        time.sleep(args.time)
        print_stats()
    except KeyboardInterrupt:
        print_stats()
else:
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print_stats()
