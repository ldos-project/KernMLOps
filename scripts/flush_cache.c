#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

void flush_cache() {
    // Flush memory cache
    system("sync && echo 3 > /proc/sys/vm/drop_caches");

    // Attempt to flush CPU cache by accessing large memory
    const size_t large_mem = 256 * 1024 * 1024; // 256 MB
    char *buffer = (char *)malloc(large_mem);
    if (buffer) {
        for (size_t i = 0; i < large_mem; i++) {
            buffer[i] = i & 0xFF; // Write to memory
            asm volatile("" : : "r" (buffer[i])); // Prevent optimization
        }
        free(buffer);
    }
}

int main() {
    printf("Flushing cache...\n");
    flush_cache();
    printf("Cache flushed.\n");
    return 0;
}
