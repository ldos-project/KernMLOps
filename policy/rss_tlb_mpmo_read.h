// SPDX-License-Identifier: GPL-2.0-only
#ifndef RSS_TLB_MPMO_READ_H
#define RSS_TLB_MPMO_READ_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    RSS_TLB_MPMO_QUEUE_SIZE = 32,
    RSS_TLB_MPMO_RSS_SIZE = 24,
    RSS_TLB_MPMO_DTLB_SIZE = 64,
};

enum rss_tlb_mpmo_state {
    RSS_TLB_MPMO_INVALID = 0,
    RSS_TLB_MPMO_READABLE = 1,
    RSS_TLB_MPMO_READING = 2,
    RSS_TLB_MPMO_WRITING = 3,
};

_Static_assert(__atomic_always_lock_free(sizeof(uint64_t), 0),
               "MPMO requires lock-free 64-bit atomics");

struct rss_tlb_mpmo_reader {
    void *data;
    size_t mapping_size;
    uint64_t reads;
    uint64_t empty;
    uint64_t busy;
    unsigned char latest[RSS_TLB_MPMO_DTLB_SIZE];
};

/* The caller supplies a validated, aligned data mapping and payload size. */
static inline int rss_tlb_mpmo_read_slot0(struct rss_tlb_mpmo_reader *reader,
                                        size_t payload_size)
{
    uint64_t *state = reader->data;
    uint64_t expected = RSS_TLB_MPMO_READABLE;

    /* A single strong CAS both observes slot zero and, if readable, claims it. */
    if (!__atomic_compare_exchange_n(state, &expected, RSS_TLB_MPMO_READING, 0,
                                     __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        switch (expected) {
        case RSS_TLB_MPMO_INVALID:
            ++reader->empty;
            return 0;
        case RSS_TLB_MPMO_READING:
        case RSS_TLB_MPMO_WRITING:
            ++reader->busy;
            return 0;
        default:
            return -EPROTO;
        }
    }
    memcpy(reader->latest, (const unsigned char *)reader->data + sizeof(*state),
           payload_size);
    __atomic_store_n(state, RSS_TLB_MPMO_INVALID, __ATOMIC_RELEASE);
    ++reader->reads;
    return 1;
}

#endif
