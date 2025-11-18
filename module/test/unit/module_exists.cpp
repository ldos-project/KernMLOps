
#include <cerrno>
#include <fcntl.h>
#include <iostream>

int main() {
  int fd = open("/dev/fstore_device", O_RDWR);
  if (fd < 0) {
    std::cerr << "The module does not seem to be inserted" << std::endl;
    return -EBADF;
  }
  return 0;
}
