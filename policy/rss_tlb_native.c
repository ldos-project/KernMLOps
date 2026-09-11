// SPDX-License-Identifier: GPL-2.0-only
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef RSS_TLB_MPMO
#include <sys/mman.h>
#include "rss_tlb_mpmo_read.h"
#endif

typedef int (*rss_tlb_callback)(void *context, int stream, const void *data, size_t size);

struct rss_tlb;
struct stream_context {
    struct rss_tlb *handle;
    int stream;
};

struct cpu_events {
    int leader_fd;
    int miss_fd;
    struct bpf_link *link;
};

struct rss_tlb {
    struct bpf_object *objects[2];
    struct bpf_link *rss_links[2];
    struct ring_buffer *rings;
    struct cpu_events *cpus;
    struct stream_context streams[2];
    rss_tlb_callback callback;
    void *context;
    size_t drain_limit;
    int cpu_count;
    int error_fds[2];
    int callback_result;
    int stop_result;
    bool stopped;
#ifdef RSS_TLB_MPMO
    struct rss_tlb_mpmo_reader mpmo[2];
    uint64_t polls;
    uint64_t reader_errors;
#endif
};

#ifndef RSS_TLB_MPMO
static int consume_record(void *context, void *data, size_t size)
{
    struct stream_context *stream = context;
    struct rss_tlb *h = stream->handle;
    int result;

    if (!h->callback) /* Failed opens discard records during cleanup. */
        return 0;
    result = size != (stream->stream == 0 ? 24U : 64U) ? -EMSGSIZE :
        h->callback(h->context, stream->stream, data, size);
    if (result)
        h->callback_result = result;
    /* Force libbpf to stop, rather than interpret a positive result as a count. */
    return result > 0 ? -ECANCELED : result;
}
#endif

int rss_tlb_stop(void *handle)
{
    struct rss_tlb *h = handle;
    int result = 0, err;

    if (!h)
        return -EINVAL;
    if (h->stopped)
        return h->stop_result;
    h->stopped = true;
    for (int i = 0; i < h->cpu_count; ++i) {
        struct cpu_events *p = &h->cpus[i];

        if (p->leader_fd >= 0 &&
            ioctl(p->leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0 &&
            !result)
            result = -errno;
        err = bpf_link__destroy(p->link);
        p->link = NULL;
        if (err && !result)
            result = err;
    }
    /* Remove the emitter before its ownership-capture bridge. */
    for (int i = 1; i >= 0; --i) {
        err = bpf_link__destroy(h->rss_links[i]);
        h->rss_links[i] = NULL;
        if (err && !result)
            result = err;
    }
#ifndef RSS_TLB_MPMO
    for (size_t attempt = 0; h->rings; ++attempt) {
        if (attempt == h->drain_limit) {
            if (result >= 0)
                result = -ELOOP;
            break;
        }
        h->callback_result = 0;
        err = ring_buffer__consume(h->rings);
        if (h->callback_result) {
            err = h->callback_result;
            if (!result || (result > 0 && err < 0))
                result = err;
            /* The record was consumed; callback batch limits must not lose the rest. */
        } else if (err <= 0) {
            if (err < 0 && result >= 0)
                result = err;
            break;
        }
    }
#endif
    h->stop_result = result;
    return result;
}

void rss_tlb_close(void *handle)
{
    struct rss_tlb *h = handle;

    if (!h)
        return;
    (void)rss_tlb_stop(h);
#ifdef RSS_TLB_MPMO
    for (int i = 0; i < 2; ++i)
        if (h->mpmo[i].data)
            munmap(h->mpmo[i].data, h->mpmo[i].mapping_size);
#else
    ring_buffer__free(h->rings);
#endif
    for (int i = 0; i < h->cpu_count; ++i) {
        if (h->cpus[i].miss_fd >= 0)
            close(h->cpus[i].miss_fd);
        if (h->cpus[i].leader_fd >= 0)
            close(h->cpus[i].leader_fd);
    }
    bpf_object__close(h->objects[1]);
    bpf_object__close(h->objects[0]);
    free(h->cpus);
    free(h);
}

void *rss_tlb_open(const char *rss_object, const char *dtlb_object,
                   rss_tlb_callback cb, void *context, char *error, size_t error_size)
{
    const char *paths[2] = { rss_object, dtlb_object };
    const char *event_names[2] = { "rss_events", "dtlb_events" };
#ifdef RSS_TLB_MPMO
    const char *head_names[2] = { "mpmo_head_rss", "mpmo_head_dtlb" };
    const char *buffer_names[2] = { "mpmo_buff_rss", "mpmo_buff_dtlb" };
    struct bpf_map *buffers[2];
#endif
    const char *program_names[3] = { "rss_owner_capture", "rss_emit", "dtlb_sample" };
    const char *step = "invalid rss_tlb_open arguments";
    struct rss_tlb *h = NULL;
    struct bpf_map *events[2], *errors[2], *misses = NULL, *owners;
    struct bpf_program *programs[3];
    struct perf_event_attr access = {
        .type = PERF_TYPE_HW_CACHE,
        .size = sizeof(struct perf_event_attr),
        .config = PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                  (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16),
        .sample_freq = 1000,
        .read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
        .disabled = 1,
        .freq = 1,
    };
    struct perf_event_attr miss = {
        .type = PERF_TYPE_HW_CACHE,
        .size = sizeof(struct perf_event_attr),
        .config = PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                  (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
        .read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
        .disabled = 0,
        .freq = 0,
    };
    int err = -EINVAL, cpu = -1, ncpus, online = 0;

    if (error && error_size)
        error[0] = '\0';
    if (!rss_object || !*rss_object || !dtlb_object || !*dtlb_object
#ifndef RSS_TLB_MPMO
        || !cb
#endif
    )
        goto fail;
    step = "allocate native RSS/DTLB handle";
    h = calloc(1, sizeof(*h));
    if (!h)
        goto system_fail;
    h->callback = cb;
    h->context = context;
    step = "determine possible CPUs";
    ncpus = libbpf_num_possible_cpus();
    if (ncpus <= 0) {
        err = ncpus < 0 ? ncpus : -ENODEV;
        goto fail;
    }
    step = "allocate per-CPU perf events";
    h->cpus = calloc((size_t)ncpus, sizeof(*h->cpus));
    if (!h->cpus)
        goto system_fail;
    h->cpu_count = ncpus;
    for (int i = 0; i < ncpus; ++i)
        h->cpus[i] = (struct cpu_events) { .leader_fd = -1, .miss_fd = -1 };

    for (int i = 0; i < 2; ++i) {
        step = i ? "open DTLB object" : "open RSS object";
        h->objects[i] = bpf_object__open_file(paths[i], NULL);
        if (!h->objects[i])
            goto system_fail;
        step = i ? "validate DTLB ring/error maps" : "validate RSS ring/error maps";
        events[i] = bpf_object__find_map_by_name(h->objects[i], event_names[i]);
        errors[i] = bpf_object__find_map_by_name(h->objects[i], "event_errors");
        if (!events[i] || !errors[i]) {
            err = -ENOENT;
            goto fail;
        }
        if (bpf_map__type(events[i]) != BPF_MAP_TYPE_RINGBUF ||
            bpf_map__type(errors[i]) != BPF_MAP_TYPE_ARRAY ||
            bpf_map__key_size(errors[i]) != sizeof(uint32_t) ||
            bpf_map__value_size(errors[i]) != sizeof(uint64_t) ||
            bpf_map__max_entries(errors[i]) != (i ? 2U : 3U)) {
            err = -EINVAL;
            goto fail;
        }
#ifdef RSS_TLB_MPMO
        step = i ? "validate DTLB MPMO maps" : "validate RSS MPMO maps";
        struct bpf_map *head = bpf_object__find_map_by_name(h->objects[i], head_names[i]);
        buffers[i] = bpf_object__find_map_by_name(h->objects[i], buffer_names[i]);
        if (!head || !buffers[i]) {
            err = -ENOENT;
            goto fail;
        }
        if (bpf_map__type(head) != BPF_MAP_TYPE_ARRAY ||
            bpf_map__key_size(head) != sizeof(uint32_t) ||
            bpf_map__value_size(head) != sizeof(uint64_t) ||
            bpf_map__max_entries(head) != 1 ||
            bpf_map__map_flags(head) != BPF_F_MMAPABLE ||
            bpf_map__type(buffers[i]) != BPF_MAP_TYPE_ARRAY ||
            bpf_map__key_size(buffers[i]) != sizeof(uint32_t) ||
            bpf_map__value_size(buffers[i]) != sizeof(uint64_t) +
                (i ? RSS_TLB_MPMO_DTLB_SIZE : RSS_TLB_MPMO_RSS_SIZE) ||
            bpf_map__max_entries(buffers[i]) != RSS_TLB_MPMO_QUEUE_SIZE ||
            bpf_map__map_flags(buffers[i]) != BPF_F_MMAPABLE) {
            err = -EINVAL;
            goto fail;
        }
        step = i ? "disable unused DTLB ring map" : "disable unused RSS ring map";
        err = bpf_map__set_autocreate(events[i], false);
        if (err)
            goto fail;
#else
        /* Every record occupies at least an eight-byte ring header. */
        h->drain_limit += bpf_map__max_entries(events[i]) / 8U;
#endif
    }
#ifndef RSS_TLB_MPMO
    ++h->drain_limit; /* Allow a final consume call to observe empty rings. */
#endif
    step = "validate rss_owners hash map";
    owners = bpf_object__find_map_by_name(h->objects[0], "rss_owners");
    if (!owners || bpf_map__type(owners) != BPF_MAP_TYPE_HASH) {
        err = owners ? -EINVAL : -ENOENT;
        goto fail;
    }
    step = "validate miss_counters perf-event array";
    misses = bpf_object__find_map_by_name(h->objects[1], "miss_counters");
    if (!misses || bpf_map__type(misses) != BPF_MAP_TYPE_PERF_EVENT_ARRAY ||
        bpf_map__key_size(misses) != sizeof(uint32_t) ||
        bpf_map__value_size(misses) != sizeof(uint32_t)) {
        err = misses ? -EINVAL : -ENOENT;
        goto fail;
    }
    step = "resize miss_counters to possible CPUs";
    err = bpf_map__set_max_entries(misses, (uint32_t)ncpus);
    if (err)
        goto fail;
    for (int i = 0; i < 3; ++i) {
        step = program_names[i];
        programs[i] = bpf_object__find_program_by_name(h->objects[i == 2], step);
        if (!programs[i]) {
            err = -ENOENT;
            goto fail;
        }
    }
    for (int i = 0; i < 2; ++i) {
        step = i ? "load DTLB object" : "load RSS object";
        err = bpf_object__load(h->objects[i]);
        if (err)
            goto fail;
        h->error_fds[i] = bpf_map__fd(errors[i]);
#ifndef RSS_TLB_MPMO
        h->streams[i] = (struct stream_context) { .handle = h, .stream = i };
#endif
    }
#ifdef RSS_TLB_MPMO
    step = "determine MPMO mmap page size";
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        err = -EINVAL;
        goto fail;
    }
    for (int i = 0; i < 2; ++i) {
        size_t bytes = (size_t)bpf_map__value_size(buffers[i]) * RSS_TLB_MPMO_QUEUE_SIZE;
        struct rss_tlb_mpmo_reader *reader = &h->mpmo[i];

        step = i ? "mmap DTLB MPMO data" : "mmap RSS MPMO data";
        reader->mapping_size = (bytes + (size_t)page_size - 1) / (size_t)page_size *
                               (size_t)page_size;
        /* BPF array mmap offset zero exposes value zero, not the kernel metadata page. */
        reader->data = mmap(NULL, reader->mapping_size, PROT_READ | PROT_WRITE,
                            MAP_SHARED, bpf_map__fd(buffers[i]), 0);
        if (reader->data == MAP_FAILED) {
            reader->data = NULL;
            goto system_fail;
        }
    }
#else
    step = "create RSS ring consumer";
    h->rings = ring_buffer__new(bpf_map__fd(events[0]), consume_record, &h->streams[0], NULL);
    if (!h->rings)
        goto system_fail;
    step = "add DTLB ring consumer";
    err = ring_buffer__add(h->rings, bpf_map__fd(events[1]), consume_record, &h->streams[1]);
    if (err)
        goto fail;
#endif

    /* Both consumers must exist before attaching any producer. */
    step = "attach RSS raw_tp/rss_stat ownership capture";
    h->rss_links[0] = bpf_program__attach_raw_tracepoint(programs[0], "rss_stat");
    if (!h->rss_links[0])
        goto system_fail;
    step = "attach RSS tp/kmem/rss_stat emitter";
    h->rss_links[1] = bpf_program__attach_tracepoint(programs[1], "kmem", "rss_stat");
    if (!h->rss_links[1])
        goto system_fail;
    for (cpu = 0; cpu < ncpus; ++cpu) {
        struct cpu_events *p = &h->cpus[cpu];
        int attach_fd;

        step = "DTLB READ ACCESS perf_event_open";
        p->leader_fd = (int)syscall(SYS_perf_event_open, &access, -1, cpu, -1,
                                    PERF_FLAG_FD_CLOEXEC);
        if (p->leader_fd < 0) {
            err = -errno;
            if (err == -ENODEV)
                continue;
            goto fail;
        }
        step = "DTLB READ MISS grouped perf_event_open";
        p->miss_fd = (int)syscall(SYS_perf_event_open, &miss, -1, cpu, p->leader_fd,
                                  PERF_FLAG_FD_CLOEXEC);
        if (p->miss_fd < 0) {
            err = -errno;
            if (err == -ENODEV) {
                close(p->leader_fd);
                p->leader_fd = -1;
                continue;
            }
            goto fail;
        }
        step = "DTLB READ MISS miss_counters update";
        if (bpf_map_update_elem(bpf_map__fd(misses), &cpu, &p->miss_fd, 0) < 0)
            goto system_fail;
        /* libbpf attachment can enable the leader, so reset the entire group first. */
        step = "DTLB READ ACCESS/MISS group reset";
        if (ioctl(p->leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0)
            goto system_fail;
        step = "DTLB READ ACCESS duplicate for BPF attachment";
        attach_fd = fcntl(p->leader_fd, F_DUPFD_CLOEXEC, 0);
        if (attach_fd < 0)
            goto system_fail;
        step = "DTLB READ ACCESS attach dtlb_sample";
        p->link = bpf_program__attach_perf_event(programs[2], attach_fd);
        if (!p->link) {
            err = -errno;
            close(attach_fd); /* libbpf takes ownership only on success. */
            goto fail;
        }
        step = "DTLB READ ACCESS/MISS group enable";
        if (ioctl(p->leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0)
            goto system_fail;
        ++online;
    }
    cpu = -1;
    step = "no online CPU for DTLB READ ACCESS/MISS";
    if (!online) {
        err = -ENODEV;
        goto fail;
    }
    return h;

system_fail:
    err = -errno;
fail:
    if (h)
        h->callback = NULL;
    rss_tlb_close(h);
    if (error && error_size) {
        if (cpu >= 0)
            snprintf(error, error_size, "CPU %d %s: %s (errno=%d)",
                     cpu, step, strerror(-err), -err);
        else
            snprintf(error, error_size, "%s: %s (errno=%d)", step, strerror(-err), -err);
    }
    errno = -err;
    return NULL;
}

int rss_tlb_poll(void *handle, int timeout_ms)
{
    struct rss_tlb *h = handle;
#ifndef RSS_TLB_MPMO
    int result;
#endif

    if (!h || timeout_ms < -1)
        return -EINVAL;
    if (h->stopped)
        return -ESHUTDOWN;
#ifdef RSS_TLB_MPMO
    int result = 0;

    ++h->polls;
    for (int i = 0; i < 2; ++i) {
        int err = rss_tlb_mpmo_read_slot0(&h->mpmo[i],
                                         i ? RSS_TLB_MPMO_DTLB_SIZE : RSS_TLB_MPMO_RSS_SIZE);

        if (err < 0) {
            ++h->reader_errors;
            result = err;
        } else if (result >= 0) {
            result += err;
        }
    }
    return result;
#else
    h->callback_result = 0;
    result = ring_buffer__poll(h->rings, timeout_ms);
    return h->callback_result ? h->callback_result : result;
#endif
}

#ifdef RSS_TLB_MPMO
int rss_tlb_mpmo_stats(void *handle, uint64_t *counts)
{
    struct rss_tlb *h = handle;

    if (!h || !counts)
        return -EINVAL;
    uint64_t values[10] = {
        h->polls, h->mpmo[0].reads, h->mpmo[1].reads,
        h->mpmo[0].empty, h->mpmo[1].empty, h->mpmo[0].busy, h->mpmo[1].busy,
        0, 0, h->reader_errors,
    };

    /* FNV-1a of the full latest copy, including padding, only at stats-query time. */
    for (int i = 0; i < 2; ++i) {
        if (!h->mpmo[i].reads)
            continue;
        size_t size = i ? RSS_TLB_MPMO_DTLB_SIZE : RSS_TLB_MPMO_RSS_SIZE;
        uint64_t checksum = UINT64_C(14695981039346656037);

        for (size_t byte = 0; byte < size; ++byte) {
            checksum ^= h->mpmo[i].latest[byte];
            checksum *= UINT64_C(1099511628211);
        }
        values[7 + i] = checksum;
    }
    memcpy(counts, values, sizeof(values));
    return 0;
}
#endif

int rss_tlb_errors(void *handle, uint64_t *counts)
{
    struct rss_tlb *h = handle;
    uint64_t values[5];

    if (!h || !counts)
        return -EINVAL;
    for (int stream = 0; stream < 2; ++stream)
        for (uint32_t key = 0; key < 2; ++key)
            if (bpf_map_lookup_elem(h->error_fds[stream], &key, &values[stream * 2 + key]) < 0)
                return -errno;
    uint32_t unavailable = 2;
    if (bpf_map_lookup_elem(h->error_fds[0], &unavailable, &values[4]) < 0)
        return -errno;
    /* RSS drops/bridge errors, DTLB drops/read errors, unavailable RSS owners. */
    memcpy(counts, values, sizeof(values));
    return 0;
}
