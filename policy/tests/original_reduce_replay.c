/* Offline helper model for the exported ORIGINAL BCC IR, never a BPF probe. */
#include <stdint.h>
#include <string.h>

struct replay_map {
  int kind, size, entries, present;
  _Alignas(8) unsigned char bytes[32 * 128];
};
struct replay_map replay_maps[16];
int replay_fail, replay_reads, replay_emits, replay_payload_size;
int replay_log[512], replay_log_size;
unsigned char replay_payload[128];
uint64_t replay_clock = UINT64_C(9876543210123);
uint64_t replay_counter = UINT64_C(0x12345678fedcba98);
uint64_t replay_enabled = UINT64_C(123456789);
uint64_t replay_running = UINT64_C(98765432);

static void log_helper(int id) {
  if (replay_log_size < 512)
    replay_log[replay_log_size++] = id;
}

void replay_config(int fd, int kind, int size, int entries) {
  struct replay_map *map = &replay_maps[-fd];
  map->kind = kind;
  map->size = size;
  map->entries = entries;
}

void replay_reset(void) {
  for (int i = 0; i < 16; ++i) {
    memset(replay_maps[i].bytes, 0, sizeof(replay_maps[i].bytes));
    replay_maps[i].present = 0;
  }
  replay_fail = replay_reads = replay_emits = replay_payload_size = replay_log_size = 0;
  memset(replay_payload, 0, sizeof(replay_payload));
}

void *replay_bytes(int fd) { return replay_maps[-fd].bytes; }
int replay_present(int fd) { return replay_maps[-fd].present; }

uint64_t replay_pseudo(uint64_t kind, int64_t fd) {
  (void)kind;
  return (uint64_t)fd;
}

void *replay_lookup(void *handle, const void *key) {
  struct replay_map *map = &replay_maps[-(intptr_t)handle];
  uint32_t index;
  memcpy(&index, key, 4);
  log_helper(100 + map->kind);
  if (map->kind == 1)
    return map->present && !(replay_fail & 2) ? map->bytes : 0;
  if ((map->kind == 2 && (replay_fail & 4)) ||
      (map->kind == 3 && (replay_fail & 8)) || index >= (uint32_t)map->entries)
    return 0;
  return map->bytes + index * map->size;
}

int replay_update(void *handle, const void *key, const void *value, uint64_t flags) {
  struct replay_map *map = &replay_maps[-(intptr_t)handle];
  (void)key;
  log_helper(2);
  /* Model BPF_NOEXIST, including preservation of the full 32-byte bridge. */
  if ((replay_fail & 16) || (flags == 1 && map->present))
    return -17;
  memcpy(map->bytes, value, map->size);
  map->present = 1;
  return 0;
}

int replay_delete(void *handle, const void *key) {
  struct replay_map *map = &replay_maps[-(intptr_t)handle];
  (void)key;
  log_helper(3);
  map->present = 0;
  return 0;
}

int replay_read(void *to, uint64_t size, const void *from) {
  (void)from;
  log_helper(4);
  ++replay_reads;
  if (replay_fail & 128)
    return -14;
  /* The original raw RSS producer reads mm->owner twice, then pid and tgid. */
  uint64_t value = size == 8 ? UINT64_C(0x100000) : (replay_reads == 2 ? 101 : 202);
  memcpy(to, &value, size);
  return 0;
}

uint64_t replay_time(void) { log_helper(5); return replay_clock; }
uint64_t replay_cpu(void) { log_helper(8); return 7; }
uint64_t replay_pid(void) { log_helper(14); return (UINT64_C(202) << 32) | 101; }
uint64_t replay_uid(void) { log_helper(15); return 42; }

int replay_perf(void *ctx, void *to, uint32_t size) {
  (void)ctx;
  log_helper(56);
  if (replay_fail & 1)
    return -22;
  uint64_t values[] = {replay_counter, replay_enabled, replay_running};
  if (size != sizeof(values))
    return -22;
  memcpy(to, values, size);
  return 0;
}

int replay_output(void *ctx, void *map, uint64_t flags, void *data, uint32_t size) {
  (void)ctx; (void)map; (void)flags;
  log_helper(25);
  if (replay_fail & 32)
    return -28;
  if (size > sizeof(replay_payload))
    return -7;
  memcpy(replay_payload, data, size);
  replay_payload_size = size;
  ++replay_emits;
  return 0;
}

/* Called only by the native replay instrumentation, immediately before CAS. */
void replay_before_cas(uint64_t *valid) {
  if (replay_fail & 64)
    *valid = 2;
}
