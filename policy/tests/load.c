// SPDX-License-Identifier: GPL-2.0-only
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <bpf/libbpf.h>

struct policy_event_selected {
    uint32_t syscall_nr;
    uint32_t doubled_syscall_nr;
};

_Static_assert(sizeof(struct policy_event_selected) == 8,
               "selected event must be 8 bytes");

struct observation {
    size_t events;
    bool saw_getpid;
};

static int print_libbpf(enum libbpf_print_level level, const char *format,
                        va_list args)
{
    (void)level;
    return vfprintf(stderr, format, args);
}

static int handle_event(void *ctx, void *data, size_t size)
{
    struct observation *observed = ctx;
    struct policy_event_selected event;
    uint32_t expected;

    if (size != sizeof(event)) {
        fprintf(stderr, "Unexpected event size: got %zu bytes, expected %zu\n",
                size, sizeof(event));
        return -EMSGSIZE;
    }

    memcpy(&event, data, sizeof(event));
    expected = event.syscall_nr * UINT32_C(2);
    if (event.doubled_syscall_nr != expected) {
        fprintf(stderr,
                "Invalid event: syscall_nr=%" PRIu32
                ", doubled_syscall_nr=%" PRIu32 " (expected %" PRIu32 ")\n",
                event.syscall_nr, event.doubled_syscall_nr, expected);
        return -EINVAL;
    }

    observed->events++;
    if (event.syscall_nr == (uint32_t)SYS_getpid)
        observed->saw_getpid = true;

    /* Yield after one sample so busy traffic cannot bypass the deadline. */
    return 1;
}

int main(int argc, char **argv)
{
    static char verifier_log[1024 * 1024];
    const struct bpf_object_open_opts opts = {
        .sz = sizeof(opts),
        .kernel_log_buf = verifier_log,
        .kernel_log_size = sizeof(verifier_log),
        .kernel_log_level = 1,
    };
    struct bpf_object *object = NULL;
    struct bpf_link *link = NULL;
    struct ring_buffer *ring = NULL;
    struct bpf_program *program;
    struct bpf_map *map;
    struct observation observed = {0};
    struct timespec deadline, now;
    int err, map_fd;
    int result = EXIT_FAILURE;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <rewritten-bpf-object>\n", argv[0]);
        return EXIT_FAILURE;
    }

    libbpf_set_print(print_libbpf);
    object = bpf_object__open_file(argv[1], &opts);
    if (!object) {
        fprintf(stderr, "Cannot open BPF object '%s': %s\n",
                argv[1], strerror(errno));
        goto cleanup;
    }

    /* Payload layout is checked on received bytes; stack-only types may not
     * appear in BTF even though loader and CO-RE metadata are present. */
    if (!bpf_object__btf(object)) {
        fprintf(stderr, "BPF object '%s' has no BTF\n", argv[1]);
        goto cleanup;
    }

    err = bpf_object__load(object);
    if (err) {
        fprintf(stderr, "Cannot load BPF object '%s': %s (%d)\n",
                argv[1], strerror(-err), err);
        if (verifier_log[0])
            fprintf(stderr, "Verifier log:\n%s\n", verifier_log);
        if (err == -EPERM || err == -EACCES)
            fprintf(stderr,
                    "BPF loading was denied; run with suitable BPF/tracing "
                    "privileges (typically root or CAP_BPF/CAP_PERFMON).\n");
        goto cleanup;
    }

    map = bpf_object__find_map_by_name(object, "events");
    if (!map) {
        fprintf(stderr, "BPF object has no map named events\n");
        goto cleanup;
    }
    program = bpf_object__find_program_by_name(object, "capture_syscall");
    if (!program) {
        fprintf(stderr, "BPF object has no program named capture_syscall\n");
        goto cleanup;
    }
    map_fd = bpf_map__fd(map);
    if (map_fd < 0) {
        fprintf(stderr, "Cannot obtain events map FD: %s\n", strerror(-map_fd));
        goto cleanup;
    }
    ring = ring_buffer__new(map_fd, handle_event, &observed, NULL);
    if (!ring) {
        fprintf(stderr, "Cannot create events ring buffer: %s\n", strerror(errno));
        goto cleanup;
    }
    link = bpf_program__attach_raw_tracepoint(program, "sys_enter");
    if (!link) {
        fprintf(stderr, "Cannot attach capture_syscall to raw sys_enter: %s\n",
                strerror(errno));
        goto cleanup;
    }

    if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) {
        fprintf(stderr, "Cannot read monotonic clock: %s\n", strerror(errno));
        goto cleanup;
    }
    deadline.tv_sec += 5;
    while (!observed.saw_getpid) {
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
            fprintf(stderr, "Cannot read monotonic clock: %s\n", strerror(errno));
            goto cleanup;
        }
        if (now.tv_sec > deadline.tv_sec ||
            (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec)) {
            fprintf(stderr,
                    "Timed out after 5 seconds waiting for getpid (syscall %ld); "
                    "observed %zu valid events\n",
                    (long)SYS_getpid, observed.events);
            goto cleanup;
        }
        if (syscall(SYS_getpid) == -1) {
            fprintf(stderr, "getpid syscall failed: %s\n", strerror(errno));
            goto cleanup;
        }
        err = ring_buffer__poll(ring, 50);
        if (err < 0 && err != -EINTR) {
            fprintf(stderr, "Ring buffer polling or event validation failed: %s (%d)\n",
                    strerror(-err), err);
            goto cleanup;
        }
    }

    printf("Verified %zu 8-byte events, including getpid (syscall %ld)\n",
           observed.events, (long)SYS_getpid);
    result = EXIT_SUCCESS;

cleanup:
    ring_buffer__free(ring);
    bpf_link__destroy(link);
    bpf_object__close(object);
    return result;
}
