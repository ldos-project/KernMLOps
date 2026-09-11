// SPDX-License-Identifier: GPL-2.0-only
#include <errno.h>
#include <linux/bpf.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POLICY_REPLAY 1
#include "../bpf/block_io.h"

extern char device_queue, started_4k_ios, block_io_starts, block_io_ends;
extern int block_rq_issue(struct bpf_raw_tracepoint_args *ctx);
extern int block_rq_complete(struct bpf_raw_tracepoint_args *ctx);

enum { CAPACITY = 16 };
struct entry {
    bool used;
    struct start_key key;
    union {
        struct queue_lengths queue;
        __u32 blocks;
    } value;
};
static struct entry maps[2][CAPACITY];
static bool fail_queue_init, fail_started_insert, fail_output;
static unsigned step, ktime_calls, read_calls, output_calls;
static __u64 now;
static void *expected_output;

static _Noreturn void fail(const char *message)
{
    fprintf(stderr, "step %u: %s\n", step, message);
    exit(1);
}

static int map_index(void *map)
{
    if (map == &device_queue) return 0;
    if (map == &started_4k_ios) return 1;
    fail("unsupported map");
}

static struct entry *find_entry(int id, const void *key)
{
    __u32 dev;
    memcpy(&dev, key, sizeof(dev));
    __u64 sector = id ? ((const struct start_key *)key)->sector : 0;
    for (unsigned i = 0; i < CAPACITY; ++i)
        if (maps[id][i].used && maps[id][i].key.dev == dev &&
            maps[id][i].key.sector == sector)
            return &maps[id][i];
    return NULL;
}

void *replay_lookup(void *map, const void *key)
{
    struct entry *entry = find_entry(map_index(map), key);
    return entry ? &entry->value : NULL;
}

long replay_update(void *map, const void *key, const void *value, __u64 flags)
{
    int id = map_index(map);
    struct entry *entry = find_entry(id, key);
    if (flags > BPF_EXIST) return -EINVAL;
    if (entry && flags == BPF_NOEXIST) return -EEXIST;
    if (!entry && flags == BPF_EXIST) return -ENOENT;
    if ((!entry && !id && fail_queue_init) || (id && fail_started_insert))
        return -ENOSPC;
    if (!entry) {
        for (unsigned i = 0; i < CAPACITY; ++i)
            if (!maps[id][i].used) {
                entry = &maps[id][i];
                break;
            }
        if (!entry) fail("map capacity exceeded");
        memset(entry, 0, sizeof(*entry));
        entry->used = true;
        memcpy(&entry->key.dev, key, sizeof(entry->key.dev));
        entry->key.sector = id ? ((const struct start_key *)key)->sector : 0;
    }
    memcpy(&entry->value, value, id ? sizeof(__u32) : sizeof(struct queue_lengths));
    return 0;
}

long replay_delete(void *map, const void *key)
{
    struct entry *entry = find_entry(map_index(map), key);
    if (!entry) return -ENOENT;
    entry->used = false;
    return 0;
}

__u64 replay_time(void)
{
    ++ktime_calls;
    return now;
}

long replay_read(void *destination, __u32 size, const void *source)
{
    ++read_calls;
    if (!source) {
        memset(destination, 0, size);
        return -EFAULT;
    }
    memcpy(destination, source, size);
    return 0;
}

long replay_output(void *map, void *data, __u64 size, __u64 flags)
{
    (void)flags;
    if (map != expected_output) fail("unexpected or unsupported output map");
    ++output_calls;
    if (fail_output) return -ENOSPC;
    if (!data || size > 64) fail("invalid output payload or size overflow");
    printf("E %u %s %llu ", step, map == &block_io_starts ? "starts" : "ends",
           (unsigned long long)size);
    for (__u64 i = 0; i < size; ++i)
        printf("%02x", (unsigned)((const unsigned char *)data)[i]);
    putchar('\n');
    return 0;
}

static void reset(void)
{
    memset(maps, 0, sizeof(maps));
    fail_queue_init = fail_started_insert = fail_output = false;
}

static void run(struct request *request, bool complete, int segments, int blocks,
                unsigned expected_started)
{
    /* raw tracepoint arguments are consecutive, native-width u64 slots. */
    __u64 args[] = {(__u64)(uintptr_t)request};
    struct bpf_raw_tracepoint_args *ctx = (void *)args;
    __u32 dev = ((__u32)request->q->disk->major << 20) |
                (__u32)request->q->disk->first_minor;
    unsigned tracked = 0;
    ++step;
    ktime_calls = read_calls = output_calls = 0;
    /* Removing a timestamp helper must not change another helper's result. */
    now = complete ? 9000123 : 3000123 + (__u64)step * 1000;
    expected_output = complete ? &block_io_ends : &block_io_starts;
    if ((complete ? block_rq_complete(ctx) : block_rq_issue(ctx)) != 0)
        fail("BPF program returned nonzero");

    /* S <step> Q <count> dev:segments:4ks ... T <count> dev:sector:4ks ... */
    printf("S %u", step);
    for (int id = 0; id < 2; ++id) {
        struct entry *sorted[CAPACITY];
        unsigned count = 0;
        for (unsigned i = 0; i < CAPACITY; ++i) {
            struct entry *entry = &maps[id][i];
            if (!entry->used) continue;
            unsigned j = count++;
            while (j && (sorted[j - 1]->key.dev > entry->key.dev ||
                         (sorted[j - 1]->key.dev == entry->key.dev &&
                          sorted[j - 1]->key.sector > entry->key.sector))) {
                sorted[j] = sorted[j - 1];
                --j;
            }
            sorted[j] = entry;
        }
        printf(" %s %u", id ? "T" : "Q", count);
        for (unsigned i = 0; i < count; ++i) {
            struct entry *entry = sorted[i];
            printf(" %u:", (unsigned)entry->key.dev);
            if (id)
                printf("%llu:%u", (unsigned long long)entry->key.sector,
                       (unsigned)entry->value.blocks);
            else
                printf("%d:%d", entry->value.queue.queue_length_segments,
                       entry->value.queue.queue_length_4ks);
        }
        if (id) tracked = count;
    }
    putchar('\n');
    printf("H %u %u %u\n", step, ktime_calls, read_calls);
    struct queue_lengths *queue = replay_lookup(&device_queue, &dev);
    if ((segments < 0) != (queue == NULL) ||
        (queue && (queue->queue_length_segments != segments ||
                   queue->queue_length_4ks != blocks))) {
        fprintf(stderr, "step %u: queue %u is %d/%d, expected %d/%d (-1=absent)\n",
                step, (unsigned)dev, queue ? queue->queue_length_segments : -1,
                queue ? queue->queue_length_4ks : -1, segments,
                segments < 0 ? -1 : blocks);
        exit(1);
    }
    if (tracked != expected_started) fail("started map entry count mismatch");
    if (output_calls != (unsigned)(complete || segments >= 0))
        fail("unexpected output helper call count");
}

int main(void)
{
    struct gendisk disks[] = {{.major = 8, .first_minor = 0},
                             {.major = 8, .first_minor = 16}};
    struct request_queue queues[] = {{.disk = &disks[0]}, {.disk = &disks[1]}};
    struct request a = {
        .q = &queues[0], .cmd_flags = 1, .__data_len = 4097,
        .__sector = 0x100000020ULL, .start_time_ns = 1000000,
        .io_start_time_ns = 2000000, .nr_phys_segments = 3,
    };
    struct request b = a, other = a, orphan = a, special = a;
    b.__sector -= 16;
    b.__data_len = 4096;
    b.nr_phys_segments = 2;
    other.q = &queues[1];
    other.__data_len = 8192;
    other.nr_phys_segments = 4;
    orphan.__sector += 1;
    special.rq_flags = BLOCK_REQ_SPECIAL_PAYLOAD;
    special.nr_phys_segments = 9;

    reset();
    run(&orphan, true, -1, 0, 0);
    /* Insert devices and sectors out of order; share a sector across devices. */
    run(&other, false, 4, 2, 1);
    run(&a, false, 3, 2, 2);
    run(&b, false, 5, 3, 3);
    run(&orphan, true, 5, 3, 3);
    run(&a, true, 2, 1, 2);
    run(&a, true, 2, 1, 2);
    run(&b, true, 0, 0, 1);
    run(&other, true, 0, 0, 0);

    reset();
    run(&special, false, 1, 2, 1);
    run(&special, true, 0, 0, 0);
    special.cmd_flags |= BLOCK_REQ_IDLE | BLOCK_REQ_FUA | BLOCK_REQ_META;
    run(&special, false, 0, 2, 1);
    run(&special, true, 0, 0, 0);

    reset();
    fail_output = true;
    run(&a, false, 3, 2, 1);
    run(&a, true, 0, 0, 0);
    fail_output = false;
    run(&a, true, 0, 0, 0);

    reset();
    fail_queue_init = true;
    run(&a, false, -1, 0, 0);
    run(&a, true, -1, 0, 0);
    fail_queue_init = false;
    run(&a, false, 3, 2, 1);
    run(&a, true, 0, 0, 0);

    reset();
    fail_started_insert = true;
    run(&a, false, 3, 2, 0);
    /* The original increments the queue even when tracking insertion fails. */
    run(&a, true, 3, 2, 0);
    fail_started_insert = false;
    run(&b, false, 5, 3, 1);
    run(&b, true, 3, 2, 0);
    run(&a, true, 3, 2, 0);
    return 0;
}
