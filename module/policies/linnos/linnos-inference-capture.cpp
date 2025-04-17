#include "linnos-inference-capture.h"
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
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <tuple>
#include <unistd.h>
#include <vector>

int map_fd = -1;
int fstore_fd = -1;
bool installed = false;
int kmod_fd = -1;

void cleanup() {
  if (kmod_fd >= 0)
    close(kmod_fd);

  if (installed)
    ioctl(map_fd, UNREGISTER_MAP, unsafeHashConvert(NAME));

  if (fstore_fd >= 0)
    close(fstore_fd);

  if (map_fd >= 0)
    close(map_fd);
}

#define ASSERT_ERRNO(x)                                                            \
  if (!(x)) {                                                                      \
    int err_errno = errno;                                                         \
    fprintf(stderr, "%s:%d: errno:%s\n", __FILE__, __LINE__, strerror(err_errno)); \
    cleanup();                                                                     \
    std::exit(-err_errno);                                                         \
  }

std::vector<std::tuple<const char*, const char*>> probes = std::vector({
    std::tuple("trace_block_rq_issue", "block_rq_issue"),
    std::tuple("trace_block_rq_complete", "block_rq_complete"),
});

const char* probes = "linnos-inference-capture.o";
const char* kmodule = "linnos-inference-capture.ko";
const char* kname = "linnos-inference-capture";

const char* map_name = NAME;
u64 map_id = unsafeHashConvert(map_name);

int main() {
  struct bpf_object* obj;

  // See if we can open the object file
  obj = bpf_object__open_file("linnos-inference-capture.o", NULL);
  ASSERT_ERRNO(obj)

  if (bpf_object__load(obj)) {
    std::cerr << "Failed to load BPF object\n";
    return 1;
  }

  // Insert the probes into the kernel
  int prog_fd;
  for (auto& [probe_name, tracepoint] : probes) {
    struct bpf_program* prog = bpf_object__find_program_by_name(obj, probe_name);
    prog_fd = bpf_program__fd(prog);

    // Attach to tracepoint
    struct bpf_link* link = bpf_program__attach_raw_tracepoint(prog, tracepoint);
    ASSERT_ERRNO(link);
  }

  // Access the BPF array map put this into the fstore
  map_fd = bpf_object__find_map_fd_by_name(obj, "segment_array");
  ASSERT_ERRNO(map_fd >= 0);

  std::cout << "Probe deployed" << std::endl;

  // Now we need to register with the feature store
  // open the feature store.
  int fstore_fd = open("/dev/fstore_device", O_RDWR);
  ASSERT_ERRNO(fstore_fd >= 0);

  register_input reg = {
      .map_name = unsafeHashConvert(NAME),
      .fd = 0,
  };
  int err = ioctl(map_fd, REGISTER_MAP, (unsigned long)&reg);
  ASSERT_ERRNO(err == 0);
  installed = true;

  // Now we install the module
  kmod_fd = open(kmodule, O_RDONLY);
  ASSERT_ERRNO(kmod_fd >= 0);

  int ret = syscall(SYS_finit_module, kmod_fd, "", 0);
  ASSERT_ERRNO(ret >= 0);

  int ret = syscall(SYS_delete_module, kname, 0);
  ASSERT_ERRNO(ret == 0);

  cleanup();
}
