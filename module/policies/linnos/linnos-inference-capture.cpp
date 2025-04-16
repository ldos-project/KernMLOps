#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <iostream>
#include <tuple>
#include <unistd.h>
#include <vector>

std::vector<std::tuple<const char*, const char*>> probes = std::vector({
    std::tuple("trace_block_rq_issue", "block_rq_issue"),
    std::tuple("trace_block_rq_complete", "block_rq_complete"),
});

int main() {
  struct bpf_object* obj;
  int prog_fd;
  int map_fd;

  // See if we can open the object file
  obj = bpf_object__open_file("linnos-inference-capture.o", NULL);
  if (!obj) {
    std::cerr << "Failed to open BPF object file" << std::endl;
    return errno;
  }

  if (bpf_object__load(obj)) {
    std::cerr << "Failed to load BPF object\n";
    return 1;
  }

  // Insert the probes into the kernel
  for (auto& [probe_name, tracepoint] : probes) {
    struct bpf_program* prog = bpf_object__find_program_by_name(obj, probe_name);
    prog_fd = bpf_program__fd(prog);

    // Attach to tracepoint
    struct bpf_link* link = bpf_program__attach_raw_tracepoint(prog, tracepoint);
    if (!link) {
      std::cerr << "Failed to attach probe: " << probe_name << " to tracepoint: " << tracepoint
                << std::endl;
      return 1;
    }
  }

  // Access the BPF array map put this into the fstore
  map_fd = bpf_object__find_map_fd_by_name(obj, "segment_array");
  if (map_fd < 0) {
    std::cerr << "Failed to find map: segment_array" << std::endl;
    return 1;
  }

  std::cout << "Worked" << std::endl;
  while (true)
    ;
}
