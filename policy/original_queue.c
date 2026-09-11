/* Slot-zero consumer for the original Kermit queue; no producer/head changes. */
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Static_assert(__atomic_always_lock_free(sizeof(uint64_t), 0),
               "original queues require lock-free 64-bit atomics");

int original_queue_read(void* mapping, size_t data_offset, size_t payload_size,
                        void* snapshot, uint64_t* cpu) {
  if (!mapping || !snapshot || !cpu || (uintptr_t)mapping % _Alignof(uint64_t) ||
      !((data_offset == 8 && payload_size == 24) ||
        (data_offset == 16 && payload_size == 32)))
    return -EINVAL;

  uint64_t* state = mapping;
  uint64_t expected = 1;
  if (!__atomic_compare_exchange_n(state, &expected, 2, 0,
                                   __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
    if (expected == 0)
      return 0;
    if (expected == 2 || expected == 3)
      return 2;
    return -EPROTO;
  }

  memcpy(snapshot, (char*)mapping + data_offset, payload_size);
  if (data_offset == 16)
    memcpy(cpu, (char*)mapping + 8, sizeof(*cpu));
  __atomic_store_n(state, 0, __ATOMIC_RELEASE);
  return 1;
}
