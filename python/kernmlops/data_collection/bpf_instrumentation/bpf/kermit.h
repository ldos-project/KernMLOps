#ifndef KERMIT_H
#define KERMIT_H

#ifdef KERMIT_CAPTURE

#define KERMIT_OUTPUT(name, type)            BPF_PERF_OUTPUT(name)
#define KERMIT_RESERVE(name, type, attempts) (&(type){0})
#define KERMIT_SUBMIT(name, ctx, ptr)        name.perf_submit((ctx), (ptr), sizeof(*(ptr)))

#else

#ifndef KERMIT_QUEUE_SIZE
#define KERMIT_QUEUE_SIZE 4096
#endif

enum kermit_slot_state {
  KERMIT_INVALID = 0,
  KERMIT_READABLE = 1,
  KERMIT_READING = 2,
  KERMIT_WRITING = 3,
};

/* Capture supplies CPU identity outside the record; opt-in MPMO perf keeps it here. */
#ifdef KERMIT_CPU_METADATA
#define KERMIT_SLOT_METADATA u64 cpu;
#define KERMIT_STORE_METADATA(slot) ((slot)->cpu = bpf_get_smp_processor_id())
#else
#define KERMIT_SLOT_METADATA
#define KERMIT_STORE_METADATA(slot) ((void)0)
#endif

#define KERMIT_OUTPUT(name, type)                                                           \
  struct kermit_slot_##name {                                                               \
    u64 valid;                                                                              \
    KERMIT_SLOT_METADATA                                                                    \
    type data;                                                                              \
  };                                                                                        \
  BPF_F_TABLE("array", u32, u64, mpmo_head_##name, 1, BPF_F_MMAPABLE);                      \
  BPF_F_TABLE("array", u32, struct kermit_slot_##name, mpmo_buff_##name, KERMIT_QUEUE_SIZE, \
              BPF_F_MMAPABLE)

// Reuse unread slots, but never overwrite a slot owned by a reader or writer.
#define KERMIT_RESERVE(name, type, attempts)                                                       \
  ({                                                                                               \
    u32 _kermit_zero = 0;                                                                          \
    u64* _kermit_head = mpmo_head_##name.lookup(&_kermit_zero);                                    \
    type* _kermit_data = NULL;                                                                     \
    if (_kermit_head) {                                                                            \
      u32 _kermit_index =                                                                          \
          __atomic_fetch_add(_kermit_head, 1, __ATOMIC_RELAXED) % KERMIT_QUEUE_SIZE;               \
      struct kermit_slot_##name* _kermit_slot = mpmo_buff_##name.lookup(&_kermit_index);           \
      if (_kermit_slot) {                                                                          \
        for (int _kermit_attempt = 0; _kermit_attempt < (attempts); _kermit_attempt++) {           \
          u64 _kermit_valid = *(volatile u64*)&_kermit_slot->valid;                                \
          if ((_kermit_valid == KERMIT_INVALID || _kermit_valid == KERMIT_READABLE) &&             \
              __atomic_compare_exchange_n(&_kermit_slot->valid, &_kermit_valid, KERMIT_WRITING, 0, \
                                          __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {                   \
            _kermit_data = &_kermit_slot->data;                                                    \
            break;                                                                                 \
          }                                                                                        \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
    _kermit_data;                                                                                  \
  })

#define KERMIT_SUBMIT(name, ctx, ptr)                                                \
  do {                                                                               \
    struct kermit_slot_##name* _kermit_slot =                                        \
        (void*)((char*)(ptr) - __builtin_offsetof(struct kermit_slot_##name, data)); \
    KERMIT_STORE_METADATA(_kermit_slot);                                             \
    __atomic_exchange_n(&_kermit_slot->valid, KERMIT_READABLE, __ATOMIC_RELEASE);    \
  } while (0)

#endif

#endif
