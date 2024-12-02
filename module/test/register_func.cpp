#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fnctl.h>
#include <linux/bpf.h>
#include <optional>
#include <sys/ioctl.h>

enum fstore_cmd {
  REGISTER_MAP = 0x0,
  UNREGISTER_MAP = 0x1,
};

std::optional<uint64_t> convert8byteStringHash(char* string) {
  uint64_t hash = 0;
  uint8_t i = 0;
  for (; string[i] != '\0' && i < 8; i++) {
    hash |= ((uint64_t)string[i]) << (i * 8);
  }
  if (string[i] != '\0') {
    return std::nullopt;
  }
  return hash;
}

consteval uint64_t unsafeHashConvert(const char* string) {
  uint64_t hash = 0;
  uint8_t i = 0;
  for (; string[i] != '\0' && i < 8; i++) {
    hash |= ((uint64_t)string[i]) << (i * 8);
  }
  return hash;
}

typedef struct register_input {
  uint64_t map_name;
  uint32_t fd;
} register_t;

int main() {
  union bpf_attr attr = {
      .map_type = BPF_MAP_TYPE_PERF_EVENT_ARRAY,
      .key_size = 32,
      .value_size = 32,
      .max_entries = 100,
  };
  int ebpf_fd = bpf(BPF_MAP_CREATE, &attr, sizeof(attr));
  if (ebpf_fd < 0)
    return ebpf_fd;

  register_t reg = {
      .map_name = unsafeHashConvert("hello"),
      .fd = 0,
  };
  int fd = open("/dev/fstore", O_RDWR);
  if (fd < 0)
    return -EBADF;

  int err = ioctl(fd, REGISTER_MAP, reg);
  assert(err != 0);
  assert(errno == EBADF);

  reg.fd = ebpf_fd;
  err = ioctl(fd, REGISTER_MAP, (unsigned long)&reg);
  assert(err == 0);

  err = ioctl(fd, UNREGISTER_MAP, (unsigned long)reg.map_name);
  assert(err == 1);
}
