#include "../../fstore/fstore.h"
#include "linnos_policy.h"
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

int fstore_fd = -1;
int kmod_fd = -1;
auto links = std::vector<struct bpf_link*>();

struct map {
  __u64 map_id;
  const char* name;
};

constexpr map installed_maps[] = {
    {unsafeHashConvert(QUEUE), "segment_array"},  {unsafeHashConvert(MPSO0), "mpmo_buff_dev0"},
    {unsafeHashConvert(MPSO1), "mpmo_buff_dev1"}, {unsafeHashConvert(MPSO2), "mpmo_buff_dev2"},
    {unsafeHashConvert(MPH0), "mpmo_head_dev0"},  {unsafeHashConvert(MPH1), "mpmo_head_dev1"},
    {unsafeHashConvert(MPH2), "mpmo_head_dev2"},
};

#define ASSERT_ERRNO_DIRTY(x)                                                      \
  if (!(x)) {                                                                      \
    int err_errno = errno;                                                         \
    fprintf(stderr, "%s:%d: errno:%s\n", __FILE__, __LINE__, strerror(err_errno)); \
  }

void cleanup() {
  if (kmod_fd >= 0)
    close(kmod_fd);

  for (const auto& map_pair : installed_maps) {
    int ret = ioctl(fstore_fd, UNREGISTER_MAP, map_pair.map_id);
    ASSERT_ERRNO_DIRTY(ret == 1);
  }

  for (const auto& link : links) {
    bpf_link__destroy(link);
  }

  if (fstore_fd >= 0)
    close(fstore_fd);
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

#ifndef PROBES_OUT
#define PROBES_OUT linnos_icollect.o
#endif

#ifndef KMODULE
#define KMODULE linnos_policy.ko
#endif

#ifndef KMODNAME
#define KMODNAME linnos_policy
#endif

#define STRINGIFY2(X) #X
#define STR(s)        STRINGIFY2(s)

const char* probes_file = STR(PROBES_OUT);
const char* kmodule = STR(KMODULE);
const char* kname = STR(KMODNAME);

int main() {
  struct bpf_object* obj;

  std::cout << "Attempting to Insert: " << probes_file << std::endl;
  std::cout << "Attempting to deploy: " << kmodule << std::endl;
  std::cout << "Attempting as: " << kname << std::endl;

  // See if we can open the object file
  obj = bpf_object__open_file(probes_file, NULL);
  ASSERT_ERRNO(obj)

  ASSERT_ERRNO(bpf_object__load(obj) == 0);

  // Insert the probes into the kernel
  int prog_fd;
  for (auto& [probe_name, tracepoint] : probes) {
    struct bpf_program* prog = bpf_object__find_program_by_name(obj, probe_name);

    prog_fd = bpf_program__fd(prog);

    // Attach to tracepoint
    struct bpf_link* link = bpf_program__attach_raw_tracepoint(prog, tracepoint);
    ASSERT_ERRNO(link);
    links.push_back(link);
  }

  std::cout << "Probe deployed" << std::endl;

  // Now we need to register with the feature store
  // open the feature store.
  fstore_fd = open("/dev/fstore_device", O_RDWR);
  ASSERT_ERRNO(fstore_fd >= 0);

  // Access the BPF array map put this into the fstore
  for (const auto& map : installed_maps) {
    int map_fd = bpf_object__find_map_fd_by_name(obj, map.name);
    ASSERT_ERRNO(map_fd >= 0);
    register_input reg = {
        .map_name = map.map_id,
        .fd = (__u32)map_fd,
    };
    int err = ioctl(fstore_fd, REGISTER_MAP, (unsigned long)&reg);
    ASSERT_ERRNO(err == 0);
  }

  // Now we install the module
  kmod_fd = open(kmodule, O_RDONLY);
  ASSERT_ERRNO(kmod_fd >= 0);

  int ret = syscall(SYS_finit_module, kmod_fd, "", 0);
  ASSERT_ERRNO(ret >= 0);

  std::string read_stdin = "";
  while (read_stdin != "END")
    getline(std::cin, read_stdin);

  ret = syscall(SYS_delete_module, kname, 0);
  ASSERT_ERRNO(ret == 0);

  cleanup();
}
