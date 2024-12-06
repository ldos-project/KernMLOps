#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/bpf.h>
#include <sys/ioctl.h>
#include <unistd.h>

#ifndef __NR_bpf
#if defined(__i386__)
#define __NR_bpf 357
#elif defined(__x86_64__)
#define __NR_bpf 321
#elif defined(__aarch64__)
#define __NR_bpf 280
#elif defined(__sparc__)
#define __NR_bpf 349
#elif defined(__s390__)
#define __NR_bpf 351
#elif defined(__arc__)
#define __NR_bpf 280
#elif defined(__mips__) && defined(_ABIO32)
#define __NR_bpf 4355
#elif defined(__mips__) && defined(_ABIN32)
#define __NR_bpf 6319
#elif defined(__mips__) && defined(_ABI64)
#define __NR_bpf 5315
#else
#error __NR_bpf not defined. libbpf does not support your arch.
#endif
#endif

int main() {
  union bpf_attr attr = {
      .map_type = BPF_MAP_TYPE_ARRAY,
      .key_size = 4,
      .value_size = 8,
      .max_entries = 64,
  };

  // bzero(&attr, sizeof(union bpf_attr));

  int ebpf_fd = syscall(__NR_bpf, BPF_MAP_CREATE, &attr, sizeof(attr));
  if (ebpf_fd < 0) {
    auto err = errno;
    std::cerr << "Failed to create map: " << err << ", " << std::strerror(err) << std::endl;
    return ebpf_fd;
  }
}
