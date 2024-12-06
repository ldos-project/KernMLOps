#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/bpf.h>
#include <optional>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

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

struct register_input {
  uint64_t map_name;
  uint32_t fd;
};

int main() {
  union bpf_attr attr = {
      .map_type = BPF_MAP_TYPE_ARRAY,
      .key_size = 4,
      .value_size = 8,
      .max_entries = 100,
  };

  int ebpf_fd = syscall(SYS_bpf, BPF_MAP_CREATE, &attr, sizeof(attr));
  if (ebpf_fd < 0) {
    auto err = errno;
    std::cerr << "Failed to create map: " << err << ", " << std::strerror(err) << std::endl;
    return ebpf_fd;
  }

  register_input reg = {
      .map_name = unsafeHashConvert("hello"),
      .fd = 0,
  };
  int fd = open("/dev/fstore_device", O_RDWR);
  if (fd < 0) {
    auto err = errno;
    std::cerr << "Failed to open module: " << err << ", " << std::strerror(err) << std::endl;
    return -EBADF;
  }

  int err = ioctl(fd, REGISTER_MAP, (unsigned long)&reg);
  assert(errno == EBADF);
  assert(err != 0);

  reg.fd = ebpf_fd;
  err = ioctl(fd, REGISTER_MAP, (unsigned long)&reg);
  assert(err == 0);

  err = ioctl(fd, UNREGISTER_MAP, (unsigned long)reg.map_name);
  assert(err == 1);
}
