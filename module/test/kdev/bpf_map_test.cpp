#include "../../fstore/fstore.h"
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/bpf.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

constexpr __u64 SAMPLE_VALUE = 0xDEADBEEF;

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

  bzero(&attr, sizeof(attr));

  int err = 0;
  __u32 zero_key = 0;
  __u64 sample_value = SAMPLE_VALUE;

  attr.map_fd = ebpf_fd;
  attr.key = (__u64)&zero_key;
  attr.value = (__u64)&sample_value;
  attr.flags = BPF_EXIST;

  err = syscall(SYS_bpf, BPF_MAP_UPDATE_ELEM, &attr, sizeof(attr));
  assert(err == 0);

  // Clear flags or EINVAL
  attr.flags = 0;

  sample_value = 0;
  assert(*((__u64*)attr.value) == 0);

  err = syscall(SYS_bpf, BPF_MAP_LOOKUP_ELEM, &attr, sizeof(attr));
  if (err == -1) {
    auto err = errno;
    std::cerr << "Failed to get value from map: " << err << ", " << std::strerror(err) << std::endl;
    return -1;
  }
  assert(*((__u64*)attr.value) == SAMPLE_VALUE);
}
