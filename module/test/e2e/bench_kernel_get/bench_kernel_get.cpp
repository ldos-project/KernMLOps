#include "bench_kernel_get.h"
#include "../../../fstore/fstore.h"
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <getopt.h>
#include <iostream>
#include <linux/bpf.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ASSERT_ERRNO(x)                                                            \
  if (!(x)) {                                                                      \
    int err_errno = errno;                                                         \
    fprintf(stderr, "%s:%d: errno:%s\n", __FILE__, __LINE__, strerror(err_errno)); \
    std::exit(-err_errno);                                                         \
  }

constexpr __u64 SAMPLE_VALUE = 0xDEADBEEF;
constexpr __u64 DEFAULT_NUMBER = 10;
constexpr __u32 DEFAULT_SIZE = 10;
constexpr __u32 DEFAULT_DATA_SIZE = 8;
constexpr int STAT_FD = 3;

int bpf_create_map(union bpf_attr& attr) {
  int ebpf_fd = syscall(SYS_bpf, BPF_MAP_CREATE, &attr, sizeof(attr));
  if (ebpf_fd < 0) {
    auto err = errno;
    std::cerr << "Failed to create map: " << err << ", " << std::strerror(err) << std::endl;
    std::exit(ebpf_fd);
  }
  return ebpf_fd;
}

__u64 benchmark_fstore(int benchmark_fd, __u32 data_size, __u32 size, __u64 number) {
  union bpf_attr attr = {
      .map_type = BPF_MAP_TYPE_ARRAY,
      .key_size = 4,
      .value_size = data_size,
      .max_entries = size,
  };

  int ebpf_fd = bpf_create_map(attr);

  register_input reg = {
      .map_name = unsafeHashConvert("benchget"),
      .fd = 0,
  };
  int fd = open("/dev/fstore_device", O_RDWR);
  ASSERT_ERRNO(fd >= 0);

  reg.fd = ebpf_fd;
  int err = ioctl(fd, REGISTER_MAP, (unsigned long)&reg);
  ASSERT_ERRNO(err == 0);

  bzero(&attr, sizeof(attr));

  __u32 key = 0;
  __u64 sample_value = SAMPLE_VALUE;
  __u64* sample_buffer = (__u64*)malloc(sizeof(char) * data_size);

  attr.map_fd = ebpf_fd;
  attr.key = (__u64)&key;
  attr.value = (__u64)sample_buffer;
  attr.flags = BPF_EXIST;

  ShiftXor rand{1, 4, 7, 13};

  for (__u64 i = 0; i < size; i++) {
    key = i;
    for (size_t i = 0; i < data_size / 8; i++) {
      sample_buffer[i] = simplerand(&rand);
    }
    err = syscall(SYS_bpf, BPF_MAP_UPDATE_ELEM, &attr, sizeof(attr));
    ASSERT_ERRNO(err == 0);
  }

  bench_get_args gsa = {
      .map_name = unsafeHashConvert("benchget"),
      .number = number,
  };
  err = ioctl(benchmark_fd, BENCH_GET_MANY, (unsigned long)&gsa);
  ASSERT_ERRNO(err == 0);

  err = ioctl(fd, UNREGISTER_MAP, unsafeHashConvert("benchget"));
  ASSERT_ERRNO(err == 1);
  return gsa.number;
}

__u64 benchmark_array(int benchmark_fd, __u32 data_size, __u32 size, __u64 number) {
  bench_get_args gsa = {
      .map_name = size,
      .number = number,
  };
  int err = ioctl(benchmark_fd, BENCH_GET_ARRAY, (unsigned long)&gsa);
  ASSERT_ERRNO(err == 0);
  return gsa.number;
}

enum Command : __u32 {
  NONE = 0x0,
  FSTORE = 0x1,
  ARRAY = (0x1 << 1),
};

int main(int argc, char** argv) {
  __u64 number = DEFAULT_NUMBER;
  __u32 size = DEFAULT_SIZE;
  __u32 data_size = 0;
  enum Command cmd = NONE;
  int array = false;
  int test = false;
  int err = 0;

  int stats_print = fcntl(STAT_FD, F_GETFD);

  int c;
  while ((c = getopt(argc, argv, "n:s:d:af")) != -1) {
    switch (c) {
      case 'n':
        number = strtoull(optarg, NULL, 10);
        break;
      case 's':
        size = strtoul(optarg, NULL, 10);
        break;
      case 'd':
        data_size = strtoul(optarg, NULL, 10);
        break;
      case 'a':
        cmd = (Command)(cmd | ARRAY);
        break;
      case 'f':
        cmd = (Command)(cmd | FSTORE);
        break;
      default:
        fprintf(stderr, "%s [-n <number>] [-s <map-size>] [-d <data-size> ] [-a | -f]\n", argv[0]);
        exit(-1);
        break;
    }
  }

  // Resize
  data_size = data_size < DEFAULT_DATA_SIZE ? DEFAULT_DATA_SIZE : data_size;
  data_size = data_size / DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE;

  __u64 time_ns = 0;

  int gsfd = open("/dev/" NAME "_device", O_RDWR);
  ASSERT_ERRNO(gsfd >= 0);

  if (cmd & ARRAY) {
    time_ns = benchmark_array(gsfd, data_size, size, number);
  }
  if (cmd & FSTORE) {
    time_ns = benchmark_fstore(gsfd, data_size, size, number);
  }

  // Important this comes after the free
  if (stats_print >= 0) {
    std::cerr << "Output to stats" << std::endl;
    err = dprintf(STAT_FD, "get_iterations %lld\tmap_size %d\tvalue_size %d\tTime(ns) %lld\n",
                  number, size, data_size, time_ns);
    assert(err > 0);
  }

  return 0;
}
