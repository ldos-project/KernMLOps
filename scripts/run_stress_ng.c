#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define INTERVAL  10000000 // 10ms in nanoseconds
#define BASE_PATH "/KernMLOps/scripts/"

void execute_command(const char* command) {
  char full_command[512];
  snprintf(full_command, sizeof(full_command), "%s &", command);
  system(full_command);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <file_index>\n", argv[0]);
    return 1;
  }

  char file_path[512];
  snprintf(file_path, sizeof(file_path), "%sstress-ng-args/%s.txt", BASE_PATH, argv[1]);

  FILE* file = fopen(file_path, "r");
  if (!file) {
    perror("File not found!");
    return 1;
  }

  char line[512];
  struct timespec loop_start, loop_finish;
  while (fgets(line, sizeof(line), file)) {
    clock_gettime(CLOCK_MONOTONIC, &loop_start);

    // Remove newline character from the line
    line[strcspn(line, "\n")] = 0;

    if (strlen(line) > 0) {
      execute_command(line);
    }

    clock_gettime(CLOCK_MONOTONIC, &loop_finish);
    long elapsed = (loop_finish.tv_sec - loop_start.tv_sec) * 1000000000L +
                   (loop_finish.tv_nsec - loop_start.tv_nsec);
    long sleep_time = (INTERVAL - elapsed) / 1000L;
    if (sleep_time > 0) {
      usleep(sleep_time);
    }
  }

  fclose(file);
  return 0;
}
