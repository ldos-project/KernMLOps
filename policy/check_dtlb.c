// SPDX-License-Identifier: GPL-2.0-only
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static int open_counter(int cpu, int group, int misses)
{
    struct perf_event_attr attr = {
        .type = PERF_TYPE_HW_CACHE,
        .size = sizeof(attr),
        .config = PERF_COUNT_HW_CACHE_DTLB |
                  ((uint64_t)PERF_COUNT_HW_CACHE_OP_READ << 8) |
                  ((uint64_t)(misses ? PERF_COUNT_HW_CACHE_RESULT_MISS :
                              PERF_COUNT_HW_CACHE_RESULT_ACCESS) << 16),
        .disabled = group == -1,
        .read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                       PERF_FORMAT_TOTAL_TIME_RUNNING,
    };
    return syscall(__NR_perf_event_open, &attr, -1, cpu, group, PERF_FLAG_FD_CLOEXEC);
}

int main(void)
{
    const size_t pages = 8192, length = pages * 4096;
    cpu_set_t original, pinned;
    int cpu = sched_getcpu(), accesses = -1, misses = -1, result = 1;
    volatile unsigned char *memory = MAP_FAILED;
    uint64_t values[5], checksum = 0;

    if (cpu < 0 || cpu >= CPU_SETSIZE ||
        sched_getaffinity(0, sizeof(original), &original)) {
        perror("read CPU affinity");
        return 1;
    }
    CPU_ZERO(&pinned);
    CPU_SET(cpu, &pinned);
    if (sched_setaffinity(0, sizeof(pinned), &pinned)) {
        perror("pin private PMU check");
        return 1;
    }
    accesses = open_counter(cpu, -1, 0);
    if (accesses < 0) {
        fprintf(stderr, "DTLB READ ACCESS is unavailable on CPU %d: %s\n", cpu, strerror(errno));
        goto cleanup;
    }
    misses = open_counter(cpu, accesses, 1);
    if (misses < 0) {
        fprintf(stderr, "DTLB READ MISS cannot join the access group on CPU %d: %s\n", cpu, strerror(errno));
        goto cleanup;
    }
    memory = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        perror("allocate private test memory");
        goto cleanup;
    }
    /* This affects only this temporary mapping, not the host's THP policy. */
    if (madvise((void *)memory, length, MADV_NOHUGEPAGE)) {
        perror("set private mapping page size");
        goto cleanup;
    }
    for (size_t page = 0; page < pages; page++)
        memory[page * 4096] = 1;
    if (ioctl(accesses, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) ||
        ioctl(accesses, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)) {
        perror("enable paired DTLB counters");
        goto cleanup;
    }
    for (unsigned repeat = 0; repeat < 32; repeat++)
        for (size_t page = 0; page < pages; page++)
            checksum += memory[((page * 2053) % pages) * 4096];
    if (ioctl(accesses, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)) {
        perror("disable paired DTLB counters");
        goto cleanup;
    }
    if (read(accesses, values, sizeof(values)) != sizeof(values) || values[0] != 2 ||
        !values[1] || !values[2] || !values[3]) {
        fprintf(stderr, "DTLB pair did not produce scheduled access counts\n");
        goto cleanup;
    }
    printf("CPU %d: accesses=%" PRIu64 ", misses=%" PRIu64
           ", enabled_ns=%" PRIu64 ", running_ns=%" PRIu64
           ", miss_ratio=%.8f, checksum=%" PRIu64 "\n",
           cpu, values[3], values[4], values[1], values[2],
           (double)values[4] / values[3], checksum);
    result = 0;
cleanup:
    if (misses >= 0) close(misses);
    if (accesses >= 0) close(accesses);
    if (memory != MAP_FAILED) munmap((void *)memory, length);
    if (sched_setaffinity(0, sizeof(original), &original)) {
        perror("restore CPU affinity");
        result = 1;
    }
    return result;
}
