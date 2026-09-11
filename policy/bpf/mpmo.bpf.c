// SPDX-License-Identifier: GPL-2.0-only
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

#if !defined(MPMO_BYTES) || (MPMO_BYTES != 24 && MPMO_BYTES != 64)
#error "compile with MPMO_BYTES=24 (RSS) or 64 (DTLB)"
#endif
#if !defined(MPMO_HEAD) || !defined(MPMO_BUFF)
#error "define MPMO_HEAD and MPMO_BUFF to the stream's short map names"
#endif
#ifdef KERMIT_CAPTURE
#error "the MPMO shim requires the KERMIT slot protocol, not capture mode"
#endif

/* Only this opt-in transport uses 32 slots; do not change the shared default. */
#undef KERMIT_QUEUE_SIZE
#define KERMIT_QUEUE_SIZE 32
#include "../../python/kernmlops/data_collection/bpf_instrumentation/bpf/kermit.h"

struct policy_mpmo_slot {
    __u64 valid;
    unsigned char data[MPMO_BYTES];
};

_Static_assert(__builtin_offsetof(struct policy_mpmo_slot, data) == 8,
               "KERMIT state must precede the payload");
_Static_assert(sizeof(struct policy_mpmo_slot) == 8 + MPMO_BYTES,
               "unexpected slot padding");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __uint(map_flags, BPF_F_MMAPABLE);
    __type(key, __u32);
    __type(value, __u64);
} MPMO_HEAD SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, KERMIT_QUEUE_SIZE);
    __uint(map_flags, BPF_F_MMAPABLE);
    __type(key, __u32);
    __type(value, struct policy_mpmo_slot);
} MPMO_BUFF SEC(".maps");

/* External for LLVM linking; the lowerer inlines and removes this definition. */
__attribute__((always_inline)) long policy_mpmo_emit(const void *payload)
{
    __u32 zero = 0;
    __u64 *head = bpf_map_lookup_elem(&MPMO_HEAD, &zero);

    if (!head)
        return -1;

    __u32 index = __atomic_fetch_add(head, 1, __ATOMIC_RELAXED) % KERMIT_QUEUE_SIZE;
    struct policy_mpmo_slot *slot = bpf_map_lookup_elem(&MPMO_BUFF, &index);

    if (!slot)
        return -1;

    __u64 valid = *(volatile __u64 *)&slot->valid;

    /* One attempt: reuse unread slots, but never a reader's or writer's slot. */
    if ((valid != KERMIT_INVALID && valid != KERMIT_READABLE) ||
        !__atomic_compare_exchange_n(&slot->valid, &valid, KERMIT_WRITING, 0,
                                     __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return -1;

    /* Constant-size memcpy also copies selected-layout padding, without a loop. */
    __builtin_memcpy(slot->data, payload, MPMO_BYTES);
    __atomic_exchange_n(&slot->valid, KERMIT_READABLE, __ATOMIC_RELEASE);
    return 0;
}
