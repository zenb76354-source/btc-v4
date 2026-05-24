/* ================================================================
 *  MAIN.CU — BTC Recovery H36 Pure GPU
 *  
 *  Cuda kernel: k_h36
 *  GPU: SHA256(ms) → secp256k1 → SHA256(pub) → RIPEMD160 → compare
 *  يشتغل 100% على GPU — ما يحتاج CPU غير للوج
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpu/kernels.cuh"

/* ================================================================
 *  TIMESTAMP RANGE: 2009-01-01 → 2012-01-01
 *  ms since epoch
 * ================================================================ */

#define START_MS 1230768000000ULL  /* 2009-01-01 00:00:00.000 UTC */
#define END_MS   1325376000000ULL  /* 2012-01-01 00:00:00.000 UTC */
#define TOTAL_KEYS (END_MS - START_MS)  /* 94,680,000,000 */

/* ================================================================
 *  TARGETS (hash160)
 * ================================================================ */

#define NUM_TARGETS 8

/* Injected via constant memory */
__constant__ uint8_t d_targets[NUM_TARGETS * 20];

/* Target struct for host */
typedef struct {
    const char *addr;
    uint8_t hash160[20];
} Target;

/* ================================================================
 *  GPU KERNEL — Generate & Check
 *  كل thread ياخذ ms واحد، يولد private key، يحسب pubkey، hash160، يقارن
 * ================================================================ */

__global__ void k_h36_pure(
    uint64_t start_ms,
    uint64_t total,
    volatile uint64_t *found_key  /* 4 × uint64 = private key if found */
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    uint64_t ms = start_ms + tid;

    /* SHA256 of ms (8-byte big-endian) */
    uint8_t scalar[32];
    {
        uint8_t msg[8];
        for (int i = 0; i < 8; i++) msg[7-i] = (uint8_t)(ms >> (i*8));
        d_sha256(msg, 8, scalar);
    }

    /* Convert scalar bytes to 4× uint64 LE */
    uint64_t k[4];
    for (int i = 0; i < 4; i++) {
        k[i] = ((uint64_t)scalar[i*8]<<56) | ((uint64_t)scalar[i*8+1]<<48) |
               ((uint64_t)scalar[i*8+2]<<40) | ((uint64_t)scalar[i*8+3]<<32) |
               ((uint64_t)scalar[i*8+4]<<24) | ((uint64_t)scalar[i*8+5]<<16) |
               ((uint64_t)scalar[i*8+6]<<8)  | (uint64_t)scalar[i*8+7];
    }

    /* Compute hash160 via ECC */
    uint8_t h160[20];
    if (!d_pk2h160(k, h160)) return;

    /* Compare against all 8 targets */
    for (int t = 0; t < NUM_TARGETS; t++) {
        int match = 1;
        for (int i = 0; i < 20; i++) {
            if (h160[i] != d_targets[t*20 + i]) { match = 0; break; }
        }
        if (match) {
            /* Write found key */
            for (int i = 0; i < 4; i++) found_key[i] = k[i];
        }
    }
}

/* ================================================================
 *  HELPERS
 * ================================================================ */

static void print_h160(const uint8_t *h) {
    for (int i = 0; i < 20; i++) printf("%02x", h[i]);
    printf("\n");
}

static void print_key(const uint64_t *k) {
    for (int i = 0; i < 4; i++) printf("%016llx", (unsigned long long)k[i]);
    printf("\n");
}

static uint64_t time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

/* ================================================================
 *  TARGET SETUP
 * ================================================================ */

static void init_targets(Target *targets) {
    static const uint8_t h160s[8][20] = {
        {0xc8,0xe5,0x09,0xee,0xe7,0xf7,0xbc,0xbc,0x11,0x1f,
         0x31,0x56,0xc0,0x4f,0x0b,0xc1,0xd7,0xb1,0xdb,0xf5},
        {0x9d,0x9a,0x9b,0x77,0x5b,0x1b,0xbe,0x33,0xe1,0xf1,
         0xba,0x7b,0xd0,0x50,0xc5,0x75,0xf6,0x2d,0xb0,0x91},
        {0xdb,0x4b,0x1a,0x77,0x39,0x45,0x6d,0x7d,0x43,0x98,
         0xc1,0xa7,0x1d,0x04,0x94,0x50,0x42,0x66,0x5c,0x3a},
        {0x39,0x9a,0x4f,0x8f,0x8f,0x73,0xd3,0x2b,0x8d,0x52,
         0x0e,0x6a,0x54,0x74,0x05,0xea,0x06,0x09,0x2e,0x2a},
        {0x3c,0x09,0x4b,0xb7,0x04,0x84,0xc3,0x15,0x7e,0x40,
         0xfd,0xa5,0x36,0xe6,0xfb,0x64,0x16,0x78,0x0e,0xe2},
        {0x35,0x7a,0xd8,0x6e,0x87,0xf3,0x15,0xa8,0x25,0x2e,
         0xde,0x8b,0x6a,0xb4,0xe3,0xe0,0xa9,0x75,0x44,0xaa},
        {0x28,0x4c,0x34,0x0f,0x0e,0xbf,0x7a,0x10,0x0b,0xc7,
         0x0c,0x44,0x2f,0x83,0x19,0x77,0xaa,0xd7,0xb3,0xb7},
        {0x7a,0x05,0xa1,0x5e,0xaf,0xbe,0x19,0xec,0xff,0x63,
         0xbc,0x7a,0x3d,0x3b,0x9d,0x3a,0xfd,0x75,0x00,0xa7}
    };
    static const char *addrs[8] = {
        "12rMpw5TCK5KPCiKKzBZ9xJqNkRgLWMyY",
        "13xDPd1MjeHrPTDCEzPFjSxqnJFn7u23Mr",
        "1JA4MpuFYDRPQDsbBQAK3BqGvkAZMrPwu5",
        "13GvAdkctq8Dn4e5VQsDsaRCtxdp3GJZnm",
        "1DTy9z4JvtqYsg44oagVpHqyQpF7ZLLs45",
        "1MVLP2k28LqgPqSDjWbF5Xg37xSDWCPHB",
        "15QezNwEH2QCJ8X7kPzfRfYSsm9BErYyby",
        "198aMn6HVAfF8dpK3P58jofVGwPfN8nffD"
    };
    for (int i = 0; i < 8; i++) {
        targets[i].addr = addrs[i];
        memcpy(targets[i].hash160, h160s[i], 20);
    }
}

/* ================================================================
 *  MAIN
 * ================================================================ */

int main() {
    printf("\n============================================\n");
    printf(" BTC RECOVERY — H36 Pure GPU (RTX 5090)\n");
    printf("============================================\n");
    printf(" Targets: A1..A7, E1\n");
    printf(" Range:   2009-01-01 → 2012-01-01\n");
    printf(" Keys:    %llu ms timestamps\n", (unsigned long long)TOTAL_KEYS);
    printf("============================================\n\n");

    /* Init targets */
    Target targets[NUM_TARGETS];
    init_targets(targets);

    /* Copy targets to GPU constant memory */
    uint8_t h_targets[NUM_TARGETS * 20];
    for (int t = 0; t < NUM_TARGETS; t++)
        memcpy(h_targets + t * 20, targets[t].hash160, 20);

    cudaMemcpyToSymbol(d_targets, h_targets, NUM_TARGETS * 20);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (targets): %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[OK] Targets loaded to GPU constant memory\n\n");

    /* Allocate device memory for found key */
    uint64_t *d_found;
    cudaMalloc(&d_found, 4 * sizeof(uint64_t));

    /* Get device info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs: %d, Max threads/SM: %d\n", prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
    printf("Global mem: %.1f GB\n\n", prop.totalGlobalMem / 1e9);

    /* Thread configuration */
    int threads = 256;
    uint64_t keys_per_launch = 10000000;  /* 10M per kernel launch (don't exceed GPU mem) */
    uint64_t total = TOTAL_KEYS;
    uint64_t start_ms = START_MS;

    uint64_t t0 = time_ms();
    uint64_t reported = 0;
    int report_interval = 100;  /* report every N launches */

    printf("Starting H36 pure GPU sweep...\n\n");

    for (uint64_t processed = 0; processed < total; ) {
        uint64_t batch = total - processed;
        if (batch > keys_per_launch) batch = keys_per_launch;

        /* Reset found */
        uint64_t h_zero[4] = {0,0,0,0};
        cudaMemcpy(d_found, h_zero, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

        /* Launch kernel */
        uint64_t blocks = (batch + threads - 1) / threads;
        k_h36_pure<<<(int)blocks, threads>>>(start_ms + processed, batch, d_found);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        /* Check if found */
        uint64_t h_found[4];
        cudaMemcpy(h_found, d_found, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (h_found[0] || h_found[1] || h_found[2] || h_found[3]) {
            printf("\n============================================\n");
            printf(" *** KEY FOUND ***\n");
            printf(" ms = %llu\n", (unsigned long long)(start_ms + processed));
            printf(" private key: ");
            print_key(h_found);
            printf("============================================\n\n");

            /* Write to file */
            FILE *f = fopen("found_key.txt", "a");
            if (f) {
                fprintf(f, "ms: %llu\n", (unsigned long long)(start_ms + processed));
                fprintf(f, "key: ");
                for (int i = 0; i < 4; i++)
                    fprintf(f, "%016llx", (unsigned long long)h_found[i]);
                fprintf(f, "\n\n");
                fclose(f);
            }
        }

        processed += batch;
        int launch_num = (int)(processed / keys_per_launch);

        if (launch_num % report_interval == 0 && launch_num > reported / keys_per_launch) {
            uint64_t elapsed = time_ms() - t0;
            double rate = (double)processed / (elapsed / 1000.0);
            reported = processed;

            printf("[H36] %llu / %llu (%.1f%%) — %.2f Mkeys/s\n",
                   (unsigned long long)processed,
                   (unsigned long long)total,
                   100.0 * processed / total,
                   rate / 1e6);
        }

        /* Update variable to allow progress-based scheduling */
        (void)0;  /* ensure loop continues */
    }

    uint64_t elapsed = time_ms() - t0;
    printf("\n============================================\n");
    printf(" SWEEP COMPLETE\n");
    printf(" Total time: %llu seconds\n", (unsigned long long)(elapsed / 1000));
    printf(" Keys checked: %llu / %llu\n", (unsigned long long)total, (unsigned long long)TOTAL_KEYS);
    printf(" Average rate: %.2f Mkeys/s\n",
           (double)total / (elapsed / 1000.0) / 1e6);
    printf("============================================\n\n");

    /* Cleanup */
    cudaFree(d_found);
    return 0;
}
