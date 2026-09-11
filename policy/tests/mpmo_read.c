// SPDX-License-Identifier: GPL-2.0-only
/* Include the collector to test its poll/stats/stop paths without loading BPF. */
#include "../rss_tlb_mpmo_native.c"
#include <limits.h>
#include <pthread.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

struct rss_slot {
    uint64_t state;
    unsigned char payload[RSS_TLB_MPMO_RSS_SIZE];
};

struct dtlb_slot {
    uint64_t state;
    unsigned char payload[RSS_TLB_MPMO_DTLB_SIZE];
};

struct queues {
    struct rss_slot rss[RSS_TLB_MPMO_QUEUE_SIZE];
    struct dtlb_slot dtlb[RSS_TLB_MPMO_QUEUE_SIZE];
};

_Static_assert(RSS_TLB_MPMO_QUEUE_SIZE == 32, "queue must have 32 slots");
_Static_assert(sizeof(struct rss_slot) == 32, "RSS stride");
_Static_assert(sizeof(struct dtlb_slot) == 72, "DTLB stride");
_Static_assert(offsetof(struct rss_slot, payload) == 8, "RSS payload offset");
_Static_assert(offsetof(struct dtlb_slot, payload) == 8, "DTLB payload offset");

static unsigned callback_calls;

static int forbidden_callback(void *context, int stream, const void *data, size_t size)
{
    (void)context;
    (void)stream;
    (void)data;
    (void)size;
    ++callback_calls;
    return -EIO;
}

static void init_queues(struct queues *q)
{
    for (size_t i = 0; i < RSS_TLB_MPMO_QUEUE_SIZE; ++i) {
        /* Slot one is readable; later slots have nonzero, invalid sentinels. */
        q->rss[i].state = q->dtlb[i].state =
            i > 1 ? UINT64_C(0xabc0000000000000) + i : i;
        for (size_t byte = 0; byte < sizeof(q->rss[i].payload); ++byte)
            q->rss[i].payload[byte] = (unsigned char)(i * 7 + byte * 3);
        for (size_t byte = 0; byte < sizeof(q->dtlb[i].payload); ++byte)
            q->dtlb[i].payload[byte] = (unsigned char)(i * 11 + byte * 5);
    }
}

static uint64_t checksum(const unsigned char *data, size_t size)
{
    uint64_t result = UINT64_C(14695981039346656037);

    for (size_t i = 0; i < size; ++i)
        result = (result ^ data[i]) * UINT64_C(1099511628211);
    return result;
}

static void check_stats(struct rss_tlb *h, const uint64_t expected[10])
{
    uint64_t actual[10];

    CHECK(rss_tlb_mpmo_stats(h, actual) == 0);
    CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
    if (!actual[9]) {
        CHECK(actual[1] + actual[3] + actual[5] == actual[0]);
        CHECK(actual[2] + actual[4] + actual[6] == actual[0]);
    }
}

static void test_poll(void)
{
    struct queues q, expected_queue;
    struct rss_tlb h = { .callback = forbidden_callback };
    uint64_t counts[10] = {0};
    const int timeouts[] = { -1, 0, 1, INT_MAX };

    init_queues(&q);
    h.mpmo[0].data = q.rss;
    h.mpmo[1].data = q.dtlb;
    memset(h.mpmo[0].latest, 0xa5, sizeof(h.mpmo[0].latest));
    check_stats(&h, counts);
    CHECK(rss_tlb_poll(NULL, 0) == -EINVAL);
    CHECK(rss_tlb_poll(&h, -2) == -EINVAL);
    CHECK(rss_tlb_poll(&h, INT_MIN) == -EINVAL);
    CHECK(rss_tlb_mpmo_stats(NULL, counts) == -EINVAL);
    CHECK(rss_tlb_mpmo_stats(&h, NULL) == -EINVAL);
    check_stats(&h, counts);

    expected_queue = q;
    for (size_t i = 0; i < sizeof(timeouts) / sizeof(timeouts[0]); ++i) {
        CHECK(rss_tlb_poll(&h, timeouts[i]) == 0);
        ++counts[0];
        ++counts[3];
        ++counts[4];
        check_stats(&h, counts);
        CHECK(memcmp(&q, &expected_queue, sizeof(q)) == 0);
    }
    for (unsigned i = 0; i < 2; ++i) {
        q.rss[0].state = i ? RSS_TLB_MPMO_WRITING : RSS_TLB_MPMO_READING;
        q.dtlb[0].state = i ? RSS_TLB_MPMO_READING : RSS_TLB_MPMO_WRITING;
        expected_queue = q;
        CHECK(rss_tlb_poll(&h, 0) == 0);
        ++counts[0];
        ++counts[5];
        ++counts[6];
        check_stats(&h, counts);
        CHECK(memcmp(&q, &expected_queue, sizeof(q)) == 0);
    }

    q.rss[0].state = q.dtlb[0].state = RSS_TLB_MPMO_READABLE;
    expected_queue = q;
    expected_queue.rss[0].state = expected_queue.dtlb[0].state = RSS_TLB_MPMO_INVALID;
    CHECK(rss_tlb_poll(&h, INT_MAX) == 2);
    CHECK(memcmp(&q, &expected_queue, sizeof(q)) == 0);
    CHECK(memcmp(h.mpmo[0].latest, q.rss[0].payload, sizeof(q.rss[0].payload)) == 0);
    CHECK(memcmp(h.mpmo[1].latest, q.dtlb[0].payload, sizeof(q.dtlb[0].payload)) == 0);
    for (size_t i = RSS_TLB_MPMO_RSS_SIZE; i < sizeof(h.mpmo[0].latest); ++i)
        CHECK(h.mpmo[0].latest[i] == 0xa5);
    ++counts[0];
    ++counts[1];
    ++counts[2];
    counts[7] = checksum(q.rss[0].payload, sizeof(q.rss[0].payload));
    counts[8] = checksum(q.dtlb[0].payload, sizeof(q.dtlb[0].payload));
    check_stats(&h, counts);

    /* Stats must hash the saved copy, not the now-reusable mapped payload. */
    memset(q.rss[0].payload, 0x5a, sizeof(q.rss[0].payload));
    memset(q.dtlb[0].payload, 0x6b, sizeof(q.dtlb[0].payload));
    expected_queue = q;
    check_stats(&h, counts);
    for (int stream = 0; stream < 2; ++stream) {
        size_t size = stream ? RSS_TLB_MPMO_DTLB_SIZE : RSS_TLB_MPMO_RSS_SIZE;

        for (size_t byte = 0; byte < size; ++byte) {
            uint64_t changed[10];

            /* Every byte, including DTLB padding, is hashed only on query. */
            h.mpmo[stream].latest[byte] ^= 0x80;
            CHECK(rss_tlb_mpmo_stats(&h, changed) == 0);
            CHECK(changed[7 + stream] != counts[7 + stream]);
            changed[7 + stream] = counts[7 + stream];
            CHECK(memcmp(changed, counts, sizeof(changed)) == 0);
            h.mpmo[stream].latest[byte] ^= 0x80;
        }
    }
    CHECK(memcmp(&q, &expected_queue, sizeof(q)) == 0);
    check_stats(&h, counts);

    q.rss[0].state = RSS_TLB_MPMO_READABLE;
    CHECK(rss_tlb_poll(&h, 0) == 1);
    ++counts[0];
    ++counts[1];
    ++counts[4];
    counts[7] = checksum(q.rss[0].payload, sizeof(q.rss[0].payload));
    CHECK(memcmp(h.mpmo[0].latest, q.rss[0].payload, sizeof(q.rss[0].payload)) == 0);
    CHECK(memcmp(&q, &expected_queue, sizeof(q)) == 0);
    check_stats(&h, counts);

    q.rss[0].state = q.dtlb[0].state = RSS_TLB_MPMO_READABLE;
    expected_queue = q;
    CHECK(rss_tlb_stop(&h) == 0);
    CHECK(rss_tlb_stop(&h) == 0);
    CHECK(rss_tlb_poll(&h, 0) == -ESHUTDOWN);
    CHECK(rss_tlb_poll(&h, -2) == -EINVAL);
    CHECK(memcmp(&q, &expected_queue, sizeof(q)) == 0);
    check_stats(&h, counts);
    CHECK(callback_calls == 0);
}

static void test_invalid_states(void)
{
    const uint64_t states[] = { 4, UINT64_C(1) << 32, UINT64_MAX };

    for (size_t i = 0; i < sizeof(states) / sizeof(states[0]); ++i) {
        for (unsigned bad = 1; bad <= 3; ++bad) {
            struct queues q, expected_queue;
            struct rss_tlb h = {0};
            uint64_t counts[10] = { 1, !(bad & 1), !(bad & 2), 0, 0, 0, 0, 0, 0,
                                    !!(bad & 1) + !!(bad & 2) };

            init_queues(&q);
            h.mpmo[0].data = q.rss;
            h.mpmo[1].data = q.dtlb;
            q.rss[0].state = bad & 1 ? states[i] : RSS_TLB_MPMO_READABLE;
            q.dtlb[0].state = bad & 2 ? states[i] : RSS_TLB_MPMO_READABLE;
            expected_queue = q;
            if (!(bad & 1)) {
                expected_queue.rss[0].state = RSS_TLB_MPMO_INVALID;
                counts[7] = checksum(q.rss[0].payload, sizeof(q.rss[0].payload));
            }
            if (!(bad & 2)) {
                expected_queue.dtlb[0].state = RSS_TLB_MPMO_INVALID;
                counts[8] = checksum(q.dtlb[0].payload, sizeof(q.dtlb[0].payload));
            }
            CHECK(rss_tlb_poll(&h, 0) == -EPROTO);
            CHECK(memcmp(&q, &expected_queue, sizeof(q)) == 0);
            check_stats(&h, counts);
            CHECK(rss_tlb_stop(&h) == 0);
            check_stats(&h, counts);
        }
    }
}

static void test_close(void)
{
    struct rss_tlb *h = calloc(1, sizeof(*h));
    void *aliases[2], *mappings[2];
    long page_size = sysconf(_SC_PAGESIZE);

    CHECK(h != NULL);
    CHECK(page_size > 0);
    for (int i = 0; i < 2; ++i) {
        size_t bytes = (i ? sizeof(struct dtlb_slot) : sizeof(struct rss_slot)) *
                       RSS_TLB_MPMO_QUEUE_SIZE;
        size_t length = (bytes + (size_t)page_size - 1) / (size_t)page_size *
                        (size_t)page_size;
        int fd = memfd_create("rss-tlb-mpmo-test", MFD_CLOEXEC);

        CHECK(fd >= 0);
        CHECK(ftruncate(fd, (off_t)length) == 0);
        mappings[i] = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        aliases[i] = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        CHECK(mappings[i] != MAP_FAILED && aliases[i] != MAP_FAILED);
        CHECK(close(fd) == 0);
        h->mpmo[i].data = mappings[i];
        h->mpmo[i].mapping_size = length;
        *(uint64_t *)aliases[i] = RSS_TLB_MPMO_READABLE;
    }
    rss_tlb_close(h);
    for (int i = 0; i < 2; ++i) {
        unsigned char resident;
        size_t bytes = (i ? sizeof(struct dtlb_slot) : sizeof(struct rss_slot)) *
                       RSS_TLB_MPMO_QUEUE_SIZE;

        CHECK(*(uint64_t *)aliases[i] == RSS_TLB_MPMO_READABLE);
        errno = 0;
        CHECK(mincore(mappings[i], (size_t)page_size, &resident) == -1);
        CHECK(errno == ENOMEM);
        CHECK(munmap(aliases[i], bytes) == 0);
    }
    rss_tlb_close(NULL);
}

enum { READER_COUNT = 4, ROUNDS = 128 };

struct contender {
    pthread_barrier_t *barrier;
    struct rss_tlb_mpmo_reader reader;
    size_t payload_size;
    int result;
};

static void wait_barrier(pthread_barrier_t *barrier)
{
    int result = pthread_barrier_wait(barrier);

    CHECK(result == 0 || result == PTHREAD_BARRIER_SERIAL_THREAD);
}

static void *compete(void *context)
{
    struct contender *c = context;

    for (unsigned round = 0; round < ROUNDS; ++round) {
        wait_barrier(c->barrier);
        c->result = rss_tlb_mpmo_read_slot0(&c->reader, c->payload_size);
        wait_barrier(c->barrier);
    }
    return NULL;
}

static void test_concurrent_ownership(void)
{
    struct queues q, expected_queue;
    pthread_barrier_t barrier;
    pthread_t threads[READER_COUNT];
    struct contender contenders[READER_COUNT];

    CHECK(pthread_barrier_init(&barrier, NULL, READER_COUNT + 1) == 0);
    for (int stream = 0; stream < 2; ++stream) {
        init_queues(&q);
        void *data = stream ? (void *)q.dtlb : (void *)q.rss;
        size_t size = stream ? RSS_TLB_MPMO_DTLB_SIZE : RSS_TLB_MPMO_RSS_SIZE;
        unsigned char *payload = (unsigned char *)data + sizeof(uint64_t);

        for (unsigned i = 0; i < READER_COUNT; ++i) {
            contenders[i] = (struct contender) {
                .barrier = &barrier, .reader = { .data = data }, .payload_size = size,
            };
            CHECK(pthread_create(&threads[i], NULL, compete, &contenders[i]) == 0);
        }
        for (unsigned round = 0; round < ROUNDS; ++round) {
            unsigned winners = 0;

            for (size_t byte = 0; byte < size; ++byte)
                payload[byte] = (unsigned char)(round * 13 + byte * 7);
            expected_queue = q;
            /* Publishing happens before competing readers leave the barrier. */
            __atomic_store_n((uint64_t *)data, RSS_TLB_MPMO_READABLE, __ATOMIC_RELEASE);
            wait_barrier(&barrier);
            wait_barrier(&barrier);
            for (unsigned i = 0; i < READER_COUNT; ++i) {
                struct contender *c = &contenders[i];

                CHECK(c->result == 0 || c->result == 1);
                if (c->result == 1) {
                    ++winners;
                    CHECK(memcmp(c->reader.latest, payload, size) == 0);
                }
                CHECK(c->reader.reads + c->reader.empty + c->reader.busy == round + 1);
            }
            CHECK(winners == 1);
            CHECK(__atomic_load_n((uint64_t *)data, __ATOMIC_ACQUIRE) == RSS_TLB_MPMO_INVALID);
            CHECK(memcmp(&q, &expected_queue, sizeof(q)) == 0);
        }
        for (unsigned i = 0; i < READER_COUNT; ++i)
            CHECK(pthread_join(threads[i], NULL) == 0);
    }
    CHECK(pthread_barrier_destroy(&barrier) == 0);
}

int main(void)
{
    char error[256];

    libbpf_set_print(NULL);
    CHECK(rss_tlb_stop(NULL) == -EINVAL);
    CHECK(rss_tlb_errors(NULL, NULL) == -EINVAL);
    CHECK(rss_tlb_open(NULL, NULL, NULL, NULL, error, sizeof(error)) == NULL);
    CHECK(errno == EINVAL);
    /* A null callback reaches object parsing; /dev/null can never load BPF. */
    CHECK(rss_tlb_open("/dev/null", "/dev/null", NULL, NULL, error, sizeof(error)) == NULL);
    CHECK(strstr(error, "open RSS object:") != NULL);
    test_poll();
    test_invalid_states();
    test_close();
    test_concurrent_ownership();
    CHECK(callback_calls == 0);
    puts("MPMO slot-zero reader tests passed");
    return 0;
}
