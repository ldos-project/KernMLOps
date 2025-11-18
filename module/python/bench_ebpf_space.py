import argparse
import os
import time
from ctypes import c_int, c_uint32

from bcc import BPF

parser = argparse.ArgumentParser()
parser.add_argument('-n', action='store', type=int, dest='number', help="Number of iterations", default=1000)
parser.add_argument('-s', action='store', type=int, dest='size', help="Size of Map", default=10)
parser.add_argument('-d', action='store', type=int, dest='data_size', help="Data Size", default=8)
parser.add_argument('-v', action='store', type=int, dest='debug', help="Debug Level", default=0)
parser.add_argument('-w', action='store_true', dest='hang', help="Keep probe alive", default=False)

FIX= 1<< 32

def simplerand(rand: list[int]) -> (int, list[int]):
    w = rand[0]
    x = rand[1]
    y = rand[2]
    z = rand[3]
    t = x
    t ^= (t << 11) % FIX
    t ^= (t >> 8) % FIX
    x = y
    y = z
    z = w
    w ^= (w >> 19) % FIX
    w ^= t
    return (w, [w, x, y, z])

def write(data, fd=3):
    try:
        os.write(fd, bytes(data, 'ascii'))
    except OSError:
        return -1

args = parser.parse_args()

txt = f"""
#include <linux/sched.h>

struct data_ts {{
    u64 ts;
}};

struct ShiftXor {{
    u32 w;
    u32 x;
    u32 y;
    u32 z;
}};

static u32 simplerand(struct ShiftXor* rand) {{
    u32 t = rand->x;
    t ^= t << 11;
    t ^= t >> 8;
    rand->x = rand->y;
    rand->y = rand->z;
    rand->z = rand->w;
    rand->w ^= rand->w >> 19;
    rand->w ^= t;
    return rand->w;
}}

struct data_sizer {{
    u32 size[{args.data_size}/4];
}};

BPF_PERF_OUTPUT(events);
BPF_PERF_OUTPUT(useless);

BPF_ARRAY(array, struct data_sizer, {args.size});

BPF_ARRAY(on, struct data_ts, 1);
BPF_PERCPU_ARRAY(temp_buffer, struct data_sizer, 1);

int probe(struct pt_regs *ctx) {{
    u32 zero_key = 0;
    struct data_ts* counter = on.lookup(&zero_key);
    if(counter == NULL) return 0;
    u64 val = __sync_lock_test_and_set(&(counter->ts), 1);
    if( val != 0 ) return 0;

    struct data_ts data;
    struct ShiftXor rand = {{1, 4, 7, 13}};
    u32 returner = 0;
    const u32 data_size = {args.data_size};

    struct data_sizer* ptr = NULL;
    struct data_sizer* tmp = temp_buffer.lookup(&zero_key);
    if(!tmp) return -1;

    u64 start = bpf_ktime_get_ns();
    for(u64 i = 0; i < {args.number}; i++) {{
        u32 key = simplerand(&rand) % {args.size};
        ptr = array.lookup(&key);
        if(!ptr) return -1;
        memcpy(tmp, ptr, sizeof(struct data_sizer));
        for(u32 j = 0; j < (data_size/4); j++) {{
            returner ^= tmp->size[j];
        }}
    }}
    u64 stop = bpf_ktime_get_ns();
    data.ts = stop - start;

    events.perf_submit(ctx, &data, sizeof(struct data_ts));
    useless.perf_submit(ctx, &returner, sizeof(returner));
    return 0;
}}
"""

bpf_ctx = BPF(text=txt, debug=args.debug)

on = bpf_ctx["on"]
on[c_int(0)]=c_int(1)

rand = [1, 4, 7, 13]
array = bpf_ctx["array"]
array_size = args.data_size // 4
for i in range(0, args.size):
    arr = (c_uint32 * array_size)()
    for j in range(0, array_size):
        val, rand = simplerand(rand)
        arr[j] = val
    array[c_uint32(i)] = arr

bpf_ctx.attach_kprobe(event='do_nanosleep', fn_name="probe")

ready = True

def print_event(cpu, data, size):
    output = bpf_ctx["events"].event(data)
    time.sleep(0.001)
    write("get_iterations %ld\tmap_size %ld\tvalue_size %d\tTime(ns) %ld\n" % (args.number, args.size, args.data_size, output.ts))
    return -1

bpf_ctx["events"].open_perf_buffer(print_event)

on[c_int(0)]=c_int(0)
while ready:
    bpf_ctx.perf_buffer_poll()
    if not args.hang:
        ready = False
