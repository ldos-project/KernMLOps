#include "../../../fstore/fstore.h"
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

  bzero(attr, sizeof(attr));

  __u32 zero_key = 0;
  __u64 sample_value = SAMPLE_VALUE;

  attr.map_fd = ebpf_fd;
  attr.key = ptr_to_u64(&zero_key);
  attr.value = ptr_to_u64(&sample_value);
  attr.flags = BPF_EXIST;

  err = syscall(SYS_bpf, BPF_MAP_UPDATE_ELEM, &attr, sizeof(attr));
  assert(err == 0);
}
