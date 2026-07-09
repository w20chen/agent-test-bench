// Pointer-chase memory latency microbenchmark for LLC interference probes.
//
// Build on the target Linux host:
//   gcc -O3 -std=c11 -Wall -Wextra -o chase scripts/experiments/chase.c
//
// Usage:
//   ./chase <memory_mb> <seconds>
//
// Output contract consumed by probe_kunpeng_llc_slices.py:
//   ns_per_access <float> accesses <integer>

#define _POSIX_C_SOURCE 200112L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <time.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

enum {
    CACHE_LINE_BYTES = 64,
    DEFAULT_BATCH_ACCESSES = 4096,
};

typedef struct {
    size_t next;
    unsigned char padding[CACHE_LINE_BYTES - sizeof(size_t)];
} ChaseNode;

_Static_assert(sizeof(ChaseNode) == CACHE_LINE_BYTES, "ChaseNode must be one cache line");

static volatile size_t sink_index;

static void usage(const char *program) {
    fprintf(stderr, "usage: %s <memory_mb> <seconds>\n", program);
}

static int parse_u64(const char *text, const char *name, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        return -1;
    }
    *out = (uint64_t)value;
    return 0;
}

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static int monotonic_ns(uint64_t *out) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        return -1;
    }
    *out = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    return 0;
}

static void shuffle_order(size_t *order, size_t count) {
    uint64_t state = 0x9e3779b97f4a7c15ull ^ (uint64_t)count;
    for (size_t i = count - 1; i > 0; --i) {
        size_t j = (size_t)(xorshift64(&state) % (uint64_t)(i + 1));
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}

static int build_ring(ChaseNode *nodes, size_t node_count) {
    size_t *order = malloc(node_count * sizeof(*order));
    if (order == NULL) {
        perror("malloc order");
        return -1;
    }

    for (size_t i = 0; i < node_count; ++i) {
        order[i] = i;
    }
    shuffle_order(order, node_count);
    for (size_t i = 0; i < node_count; ++i) {
        nodes[order[i]].next = order[(i + 1) % node_count];
    }

    free(order);
    return 0;
}

static int allocate_nodes(ChaseNode **nodes, size_t bytes) {
#if defined(_WIN32)
    *nodes = _aligned_malloc(bytes, CACHE_LINE_BYTES);
    return *nodes == NULL ? ENOMEM : 0;
#else
    return posix_memalign((void **)nodes, CACHE_LINE_BYTES, bytes);
#endif
}

static void free_nodes(ChaseNode *nodes) {
#if defined(_WIN32)
    _aligned_free(nodes);
#else
    free(nodes);
#endif
}

static size_t keep_live(size_t value) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : "+r"(value) : : "memory");
#elif defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
    return value;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        usage(argv[0]);
        return 2;
    }

    uint64_t memory_mb = 0;
    uint64_t seconds = 0;
    if (parse_u64(argv[1], "memory_mb", &memory_mb) != 0 ||
        parse_u64(argv[2], "seconds", &seconds) != 0) {
        usage(argv[0]);
        return 2;
    }

    if (memory_mb > UINT64_MAX / (1024ull * 1024ull)) {
        fprintf(stderr, "memory_mb is too large: %" PRIu64 "\n", memory_mb);
        return 2;
    }
    if (seconds > UINT64_MAX / 1000000000ull) {
        fprintf(stderr, "seconds is too large: %" PRIu64 "\n", seconds);
        return 2;
    }
    uint64_t requested_bytes = memory_mb * 1024ull * 1024ull;
    if (requested_bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "requested memory is too large for this platform\n");
        return 2;
    }
    size_t node_count = (size_t)(requested_bytes / sizeof(ChaseNode));
    if (node_count < 2) {
        fprintf(stderr, "memory region must contain at least two cache lines\n");
        return 2;
    }
    if ((uint64_t)node_count != requested_bytes / sizeof(ChaseNode)) {
        fprintf(stderr, "requested memory is too large for this platform\n");
        return 2;
    }

    ChaseNode *nodes = NULL;
    int rc = allocate_nodes(&nodes, node_count * sizeof(*nodes));
    if (rc != 0) {
        fprintf(stderr, "aligned allocation: %s\n", strerror(rc));
        return 1;
    }
    memset(nodes, 0, node_count * sizeof(*nodes));

    if (build_ring(nodes, node_count) != 0) {
        free_nodes(nodes);
        return 1;
    }

    uint64_t start_ns = 0;
    uint64_t now_ns = 0;
    if (monotonic_ns(&start_ns) != 0) {
        free_nodes(nodes);
        return 1;
    }
    uint64_t duration_ns = seconds * 1000000000ull;
    uint64_t accesses = 0;
    size_t idx = 0;

    do {
        for (int i = 0; i < DEFAULT_BATCH_ACCESSES; ++i) {
            idx = nodes[idx].next;
        }
        idx = keep_live(idx);
        if (accesses > UINT64_MAX - DEFAULT_BATCH_ACCESSES) {
            fprintf(stderr, "access counter overflow\n");
            free_nodes(nodes);
            return 1;
        }
        accesses += DEFAULT_BATCH_ACCESSES;
        if (monotonic_ns(&now_ns) != 0) {
            free_nodes(nodes);
            return 1;
        }
    } while (now_ns - start_ns < duration_ns);

    sink_index = idx;
    double ns_per_access = (double)(now_ns - start_ns) / (double)accesses;
    printf("ns_per_access %.3f accesses %" PRIu64 "\n", ns_per_access, accesses);

    free_nodes(nodes);
    return 0;
}
