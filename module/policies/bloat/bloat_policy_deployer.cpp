#include "../../fstore/fstore.h"
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <iostream>
#include <linux/bpf.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <tuple>
#include <unistd.h>
#include <vector>

const char* probes_file = "bloat_probe.o";

#define RETURN_ERRNO(x)                                                            \
  if (!(x)) {                                                                      \
    int err_errno = errno;                                                         \
    fprintf(stderr, "%s:%d: errno:%s\n", __FILE__, __LINE__, strerror(err_errno)); \
    return -err_errno;                                                             \
  }

int perf_event_open(struct perf_event_attr* hw_event, pid_t pid, int cpu, int group_fd,
                    unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// take pid as input

int main(int argc, char** argv) {
  int pid = std::atoi(argv[0]);
  struct bpf_object* obj;
  obj = bpf_object__open_file(probes_file, NULL);
  RETURN_ERRNO(obj);

  RETURN_ERRNO(bpf_object__load(obj) == 0);

  struct bpf_program* prog = bpf_object__find_program_by_name(obj, "raw_trace_rss_stat");
  int prog_fd = bpf_program__fd(prog);

  int singular_cache_fd = bpf_object__find_map_fd_by_name(obj, "singular_cache_array");
  RETURN_ERRNO(singular_cache_fd >= 0);
  int rss_headers_fd = bpf_object__find_map_fd_by_name(obj, "rss_head_idx");
  RETURN_ERRNO(rss_headers_fd >= 0);
  int rss_data_fd = bpf_object__find_map_fd_by_name(obj, "rss_buffer");
  RETURN_ERRNO(rss_data_fd >= 0);
  int dtlb_header_fd = bpf_object__find_map_fd_by_name(obj, "dtlb_header");
  RETURN_ERRNO(dtlb_header_fd >= 0);
  int dtlb_data_fd = bpf_object__find_map_fd_by_name(obj, "dtlb_buffer");
  RETURN_ERRNO(dtlb_data_fd >= 0);

  // Attach to tracepoint
  struct bpf_link* link = bpf_program__attach_raw_tracepoint(prog, "rss_stat");
  RETURN_ERRNO(link);

  prog = bpf_object__find_program_by_name(obj, "trace_rss_stat");
  prog_fd = bpf_program__fd(prog);

  // Attach to tracepoint
  link = bpf_program__attach_tracepoint(prog, "kmem", "rss_stat");
  RETURN_ERRNO(link);

  prog = bpf_object__find_program_by_name(obj, "dtlb_miss_rate");
  prog_fd = bpf_program__fd(prog);

  // Create perf setup
  struct perf_event_attr attr = {};
  attr.type = PERF_TYPE_HW_CACHE;
  attr.config = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
  attr.size = sizeof(attr);
  attr.disabled = 1;
  attr.sample_freq = 1000;

  int perf_fd = perf_event_open(&attr, pid, -1, -1, 0);

  RETURN_ERRNO(ioctl(perf_fd, PERF_EVENT_IOC_SET_BPF, prog_fd) >= 0);
  RETURN_ERRNO(ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0) >= 0);

  return 0;
}
