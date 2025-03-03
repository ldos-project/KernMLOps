#include "../e2e/bench_kernel_get/bench_kernel_get.h"
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/bpf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ASSERT_ERRNO(x)                                                            \
  if (!(x)) {                                                                      \
    int err_errno = errno;                                                         \
    fprintf(stderr, "%s:%d: errno:%s\n", __FILE__, __LINE__, strerror(err_errno)); \
    exit(-err_errno);                                                              \
  }

// constexpr __u64 SAMPLE_VALUE = 0xDEADBEEF;
constexpr __u64 DEFAULT_NUMBER = 10;
constexpr __u32 DEFAULT_SIZE = 10;
constexpr __u32 DEFAULT_DATA_SIZE = 8;
constexpr int STAT_FD = 3;
constexpr int RET_FD = 4;
constexpr __u32 MAX = 16384;

struct data_t {
  size_t size[MAX / sizeof(__u32)];
};

data_t temp_buffer;

int main(int argc, char** argv) {
  __u64 number = DEFAULT_NUMBER;
  __u32 size = DEFAULT_SIZE;
  __u32 data_size = 0;

  int c;
  while ((c = getopt(argc, argv, "n:s:d:")) != -1) {
    switch (c) {
      case 'n':
        number = strtoll(optarg, NULL, 10);
        break;
      case 's':
        size = strtol(optarg, NULL, 10);
        break;
      case 'd':
        data_size = strtol(optarg, NULL, 10);
        break;
      default:
        fprintf(stderr, "%s [-n <number>] [-s <map-size>] [-d <data-size> ]\n", argv[0]);
        exit(-1);
        break;
    }
  }

  // Resize
  data_size = data_size < DEFAULT_DATA_SIZE ? DEFAULT_DATA_SIZE : data_size;
  data_size = data_size / DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE;

  // Open the device
  const char* dev_name = "/dev/ksched";
  int fd = open(dev_name, O_RDWR);
  if (fd == -1) {
    perror("Failed to open device");
    return -1;
  }

  // Map the memory region
  int shm_size = size * data_size;
  void* ksched_shm =
      mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE | MAP_LOCKED, fd, 0);
  if (ksched_shm == MAP_FAILED) {
    perror("mmap failed");
    close(fd);
    return -1;
  }

  ShiftXor rand{1, 4, 7, 13};

  // Write random data into memory region
  __u32 key = 0;
  __u64* sample_buffer = (__u64*)malloc(sizeof(char) * data_size);
  for (__u64 i = 0; i < size; i++) {
    key = i;
    for (size_t j = 0; j < data_size / 8; j++) {
      sample_buffer[j] = simplerand(&rand);
    }
    memcpy((char*)ksched_shm + i * data_size, sample_buffer, data_size);
  }

  rand = {1, 4, 7, 13};

  // Read random data from memory region and time it
  __u32 returner = 0;
  const auto start = std::chrono::steady_clock::now();
  for (__u64 i = 0; i < number; i++) {
    key = simplerand(&rand) % size;
    memcpy(&temp_buffer, ((char*)ksched_shm) + (key * data_size), data_size);
    for (__u32 j = 0; j < data_size / 4; j++) {
      returner ^= temp_buffer.size[j];
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

  // Write the file
  int err = fcntl(STAT_FD, F_GETFD);
  if (err != -1) {
    std::cerr << "Output to stats" << std::endl;
    err = dprintf(STAT_FD, "get_iterations %lld\tmap_size %d\tvalue_size %d\tTime(ns) %ld\n",
                  number, size, data_size, time);
    assert(err > 0);
  }

  err = fcntl(RET_FD, F_GETFD);
  if (err != -1) {
    std::cerr << "Output to returner" << std::endl;
    err = dprintf(RET_FD, "returner %d", returner);
    assert(err > 0);
  }
}
