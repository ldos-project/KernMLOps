// SPDX-License-Identifier: GPL-2.0-only
#define _GNU_SOURCE
#include <bpf/libbpf.h>
#include <errno.h>
#include <inttypes.h>
#include <linux/magic.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/vfs.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define WORK_BYTES (8U * 1024 * 1024)
#define TIMEOUT_SECONDS 8

static volatile sig_atomic_t interrupted;

static void stop(int signal_number)
{
    (void)signal_number;
    interrupted = 1;
}

struct stream {
    const char *name;
    size_t size;
    uint64_t count;
    uint64_t latency[2];
};

static int sample(void *ctx, void *data, size_t size)
{
    struct stream *stream = ctx;

    if (size != stream->size) {
        fprintf(stderr, "%s record: expected %zu bytes, got %zu\n",
                stream->name, stream->size, size);
        return -EMSGSIZE;
    }
    if (size == 16)
        memcpy(stream->latency, data, sizeof(stream->latency));
    stream->count++;
    /* Stop after one record per ready ring, even under busy global I/O. */
    return 1;
}

int main(int argc, char **argv)
{
    static char verifier_log[1024 * 1024];
    LIBBPF_OPTS(bpf_object_open_opts, opts,
                .kernel_log_buf = verifier_log,
                .kernel_log_size = sizeof(verifier_log),
                .kernel_log_level = 1);
    const char *names[] = { "block_rq_issue", "block_rq_complete" };
    struct stream starts = { .name = "issue", .size = 48 };
    struct stream ends = { .name = "completion", .size = 16 };
    struct bpf_object *object = NULL;
    struct bpf_link *links[2] = { NULL, NULL };
    struct ring_buffer *ring = NULL;
    struct statfs fs;
    struct timespec deadline, now;
    char *path = NULL;
    pid_t writer = -1;
    bool path_linked = false;
    struct sigaction action = { .sa_handler = stop };
    int fd = -1, result = 1, err, start_fd, end_fd;

    if (argc != 3 || argv[2][0] == '\0') {
        fprintf(stderr, "Usage: block-io-load object.bpf.o work-directory\n");
        return 1;
    }
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) < 0 || sigaction(SIGTERM, &action, NULL) < 0) {
        perror("install cancellation handlers");
        return 1;
    }
    fprintf(stderr,
            "Live validation loads BPF and attaches both raw block tracepoints;\n"
            "permission for those privileged operations must be approved before execution.\n"
            "Creates one unique regular file, writes at most %u bytes, then fsyncs.\n"
            "Checks 48-byte issue and 16-byte completion records for up to %d seconds.\n"
            "Global streams include background I/O. Completions have no identity: neither\n"
            "file attribution nor exact value equivalence between live events is asserted.\n",
            WORK_BYTES, TIMEOUT_SECONDS);
    if (asprintf(&path, "%s/block-io-load-XXXXXX", argv[2]) < 0) {
        perror("allocate temporary-file path");
        goto cleanup;
    }
    fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp in work-directory");
        goto cleanup;
    }
    path_linked = true;
    /* The open descriptor still supports writes/fsync, without a pathname that
     * could survive cancellation or a killed writer. */
    if (unlink(path) < 0) {
        perror("unlink newly created temporary file");
        goto cleanup;
    }
    path_linked = false;
    if (fstatfs(fd, &fs) < 0) {
        perror("fstatfs temporary file");
        goto cleanup;
    }
    fprintf(stderr, "Temporary-file filesystem type: 0x%lx\n", (unsigned long)fs.f_type);
    if (fs.f_type == (long)TMPFS_MAGIC || fs.f_type == (long)RAMFS_MAGIC) {
        fprintf(stderr, "TMPFS/RAMFS cannot supply disk I/O; use a disk-backed work-directory.\n");
        goto cleanup;
    }
    fprintf(stderr, "Network or stacked filesystems may not produce local block events.\n");

    object = bpf_object__open_file(argv[1], &opts);
    if (!object) {
        fprintf(stderr, "Open BPF object '%s': %s\n", argv[1], strerror(errno));
        goto cleanup;
    }
    err = bpf_object__load(object);
    verifier_log[sizeof(verifier_log) - 1] = '\0';
    if (err) {
        fprintf(stderr, "Load BPF object '%s': %s; check BPF privileges and kernel BTF/support.\n"
                "Verifier log:\n%s\n", argv[1], strerror(-err),
                verifier_log[0] ? verifier_log : "(unavailable; loading may have failed before verification)");
        goto cleanup;
    }
    start_fd = bpf_object__find_map_fd_by_name(object, "block_io_starts");
    end_fd = bpf_object__find_map_fd_by_name(object, "block_io_ends");
    if (start_fd < 0 || end_fd < 0) {
        fprintf(stderr, "Find ring map '%s': %s\n",
                start_fd < 0 ? "block_io_starts" : "block_io_ends",
                strerror(-(start_fd < 0 ? start_fd : end_fd)));
        goto cleanup;
    }
    ring = ring_buffer__new(start_fd, sample, &starts, NULL);
    if (!ring) {
        perror("ring_buffer__new(block_io_starts)");
        goto cleanup;
    }
    err = ring_buffer__add(ring, end_fd, sample, &ends);
    if (err) {
        fprintf(stderr, "ring_buffer__add(block_io_ends): %s\n", strerror(-err));
        goto cleanup;
    }
    for (int i = 0; i < 2; i++) {
        struct bpf_program *program = bpf_object__find_program_by_name(object, names[i]);

        if (!program) {
            fprintf(stderr, "Missing BPF program '%s'\n", names[i]);
            goto cleanup;
        }
        links[i] = bpf_program__attach_raw_tracepoint(program, names[i]);
        if (!links[i]) {
            fprintf(stderr, "Attach raw tracepoint '%s': %s; check permissions and tracepoint availability.\n",
                    names[i], strerror(errno));
            goto cleanup;
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &deadline) < 0) {
        perror("clock_gettime");
        goto cleanup;
    }
    deadline.tv_sec += TIMEOUT_SECONDS;
    if (interrupted)
        goto cleanup;
    pid_t parent = getpid();
    writer = fork();
    if (writer < 0) {
        perror("fork I/O writer");
        goto cleanup;
    }
    if (writer == 0) {
        unsigned char buffer[128 * 1024];
        size_t written = 0;
        int writer_result = 1;

        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) < 0 || getppid() != parent)
            _exit(1);
        /* Raw-tracepoint links are reference-counted FDs. Close the child's
         * copies before doing any potentially blocking disk I/O. */
        for (int i = 0; i < 2; i++)
            close(bpf_link__fd(links[i]));
        ring_buffer__free(ring);
        bpf_object__close(object);
        memset(buffer, 0xa5, sizeof(buffer));
        while (written < WORK_BYTES) {
            size_t chunk = WORK_BYTES - written;
            ssize_t n;

            if (chunk > sizeof(buffer))
                chunk = sizeof(buffer);
            n = write(fd, buffer, chunk);
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0) {
                if (n == 0)
                    errno = EIO;
                perror("write temporary file");
                goto writer_out;
            }
            written += (size_t)n;
        }
        do {
            err = fsync(fd);
        } while (err < 0 && errno == EINTR);
        if (err < 0) {
            perror("fsync temporary file");
            goto writer_out;
        }
        writer_result = 0;
writer_out:
        fprintf(stderr, "Workload: %zu/%u bytes written; fsync %s.\n",
                written, WORK_BYTES, writer_result ? "not completed" : "completed");
        if (close(fd) < 0) {
            perror("close writer temporary file");
            writer_result = 1;
        }
        _exit(writer_result);
    }

    for (;;) {
        if (interrupted) {
            fprintf(stderr, "Live validation cancelled.\n");
            goto cleanup;
        }
        if (clock_gettime(CLOCK_MONOTONIC, &now) < 0) {
            perror("clock_gettime");
            goto cleanup;
        }
        if (now.tv_sec > deadline.tv_sec ||
            (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec)) {
            fprintf(stderr, "Timed out after %d seconds: file workload or issue/completion observations incomplete.\n",
                    TIMEOUT_SECONDS);
            goto cleanup;
        }
        err = ring_buffer__poll(ring, 50);
        if (err < 0 && err != -EINTR) {
            fprintf(stderr, "Poll block I/O streams: %s\n", strerror(-err));
            goto cleanup;
        }
        if (writer > 0) {
            int status;
            pid_t waited = waitpid(writer, &status, WNOHANG);

            if (waited < 0) {
                if (errno == EINTR)
                    continue;
                perror("waitpid I/O writer");
                goto cleanup;
            }
            if (waited == writer) {
                writer = -1;
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                    fprintf(stderr, "Temporary-file I/O writer failed.\n");
                    goto cleanup;
                }
            }
        }
        if (writer < 0 && starts.count && ends.count) {
            result = 0;
            break;
        }
    }

cleanup:
    /* Detach before waiting for a writer that might be stuck in kernel I/O. */
    for (int i = 1; i >= 0; i--)
        bpf_link__destroy(links[i]);
    ring_buffer__free(ring);
    bpf_object__close(object);
    if (writer > 0) {
        if (kill(writer, SIGKILL) < 0 && errno != ESRCH)
            perror("stop I/O writer");
        for (int attempt = 0; attempt < 20; attempt++) {
            pid_t waited = waitpid(writer, NULL, WNOHANG);
            if (waited == writer || (waited < 0 && errno == ECHILD)) {
                writer = -1;
                break;
            }
            const struct timespec pause = { .tv_nsec = 10000000 };
            nanosleep(&pause, NULL);
        }
        if (writer > 0)
            fprintf(stderr, "Writer has a pending SIGKILL; not waiting for blocked I/O.\n");
    }
    if (fd >= 0) {
        if (path_linked && unlink(path) < 0) {
            perror("unlink temporary file");
            result = 1;
        }
        if (close(fd) < 0) {
            perror("close temporary file");
            result = 1;
        }
    }
    free(path);
    printf("Observed global records after attachment: issue=%" PRIu64 ", completion=%" PRIu64 "\n",
           starts.count, ends.count);
    if (ends.count)
        printf("Last completion: block_latency_us=%" PRIu64 ", block_io_latency_us=%" PRIu64 "\n",
               ends.latency[0], ends.latency[1]);
    if (!result)
        puts("Live block I/O validation passed; no workload attribution or value-equivalence claim.");
    return result;
}
