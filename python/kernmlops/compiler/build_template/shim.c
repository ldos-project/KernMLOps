
#include <linux/kernel.h>
#include <linux/slab.h>

#define EINVAL 22
#define ENOMEM 12

int posix_memalign(void** memptr, size_t alignment, size_t size) {
  void *raw, *aligned;

  // alignment must be power of two and multiple of sizeof(void *)
  if ((alignment % sizeof(void*)) != 0 || (alignment & (alignment - 1)) != 0)
    return EINVAL;

  // over-allocate to make room for alignment adjustment + original ptr
  raw = kmalloc(size + alignment - 1 + sizeof(void*), GFP_ATOMIC);
  if (!raw)
    return ENOMEM;

  // align the pointer
  aligned = (void*)ALIGN((unsigned long)raw + sizeof(void*), alignment);

  // stash the original pointer just before aligned block
  ((void**)aligned)[-1] = raw;

  *memptr = aligned;
  return 0;
}

void free(void* ptr) {
  if (ptr)
    kfree(((void**)ptr)[-1]);
}
