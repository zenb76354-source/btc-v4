/* ================================================================
 *  H36 — GPU Timestamp Millisecond Sweep (2009-2011)
 *  
 *  Generates private keys from unix millisecond timestamps:
 *    key = SHA256(ms_since_epoch_big_endian)
 *  
 *  Range: 2009-01-01 00:00:00.000 → 2012-01-01 00:00:00.000
 *  = 94,675,968,000 ms ≈ 94.7 billion keys
 *  On RTX 5090 (~100B keys/s) → ~1 second
 *
 *  Checks ALL 8 targets simultaneously in one GPU sweep
 *
 *  Compile part of: nvcc -O2 -arch=sm_100 -std=c++11 \
 *    gpu/timestamp_sweep.cu main.cu -o btc-recovery \
 *    -lsecp256k1 -lssl -lcrypto -Xcompiler -fopenmp
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../common/targets.h"
#include "kernels.cuh"

/* ---------------------------------------------------------------
 *  GPU KERNEL: kernel_timestamp_ms_multi
 *  
 *  Each thread handles ONE millisecond timestamp.
 *  threadIdx.x + blockIdx.x * blockDim.x = ms_offset from start
 *  
 *  key = SHA256(8 bytes of ms_timestamp BE) → pubkey → hash160
 *  Compare hash160 against ALL target hash160s at once
 * --------------------------------------------------------------- */

__global__ void kernel_timestamp_ms_multi(
    uint64_t start_ms,       /* starting timestamp in ms */
    uint64_t num_ms,         /* number of ms to check */
    const uint8_t (*targets)[20],  /* array of target hash160s */
    int num_targets,
    volatile int *found,
    uint8_t *found_priv      /* [0..31] = private key, [32] = target index */
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ms) return;
    if (*found) return;

    uint64_t time_ms = start_ms + idx;

    /* Build 8-byte big-endian timestamp */
    uint8_t msg[8];
    for (int i = 0; i < 8; i++)
        msg[7 - i] = (uint8_t)(time_ms >> (i * 8));

    /* SHA256 → private key */
    uint8_t sha[32];
    d_sha256(msg, 8, sha);

    /* Convert to 4-limb LE scalar (d_pk2h160 expects uint64_t[4] LE) */
    uint64_t sc[4];
    for (int i = 0; i < 4; i++) {
        sc[i] = 0;
        for (int j = 0; j < 8; j++)
            sc[i] |= ((uint64_t)sha[i * 8 + j]) << (j * 8);
    }

    /* Derive hash160 */
    uint8_t h160[20];
    if (!d_pk2h160(sc, h160)) return;

    /* Compare against all targets */
    for (int t = 0; t < num_targets; t++) {
        int match = 1;
        for (int di = 0; di < 20; di++) {
            if (h160[di] != targets[t][di]) { match = 0; break; }
        }
        if (match) {
            *found = 1;
            for (int i = 0; i < 32; i++) found_priv[i] = sha[i];
            found_priv[32] = (uint8_t)t;
            return;
        }
    }
}

/* ---------------------------------------------------------------
 *  Host function: H36 — GPU Timestamp ms Sweep
 *  
 *  Returns target index (1..8) if found, 0 otherwise
 * --------------------------------------------------------------- */

int h36_timestamp_ms_sweep(void) {
    /* Range: 2009-01-01 00:00:00.000 → 2012-01-01 00:00:00.000 */
    const uint64_t START_MS = 1230768000000ULL;   /* 2009-01-01 */
    const uint64_t END_MS   = 1325376000000ULL;   /* 2012-01-01 */
    const uint64_t NUM_MS   = END_MS - START_MS;  /* 94,675,968,000 */

    /* Upload all targets */
    uint8_t d_targets[NUM_TARGETS][20];
    for (int i = 0; i < NUM_TARGETS; i++)
        memcpy(d_targets[i], TARGET_H160[i], 20);

    uint8_t *gpu_targets;
    int *gpu_found;
    uint8_t *gpu_found_priv;

    cudaMalloc(&gpu_targets, NUM_TARGETS * 20);
    cudaMemcpy(gpu_targets, d_targets, NUM_TARGETS * 20, cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_found, sizeof(int));
    cudaMalloc(&gpu_found_priv, 33);

    int h_found = 0;
    cudaMemcpy(gpu_found, &h_found, sizeof(int), cudaMemcpyHostToDevice);

    /* Launch kernel — large grid to cover ~95B keys */
    const int THREADS = 256;
    uint64_t blocks = (NUM_MS + THREADS - 1) / THREADS;

    /* RTX 5090: cap blocks to avoid launch failure, iterate */
    const uint64_t MAX_BLOCKS_PER_LAUNCH = 65535 * 4;  /* ~67M threads per launch */
    uint64_t remaining = NUM_MS;
    uint64_t current_start = START_MS;
    int range_idx = 0;

    log_msg("[H36] GPU ms sweep: %llu ms from 2009-01-01 to 2012-01-01",
            (unsigned long long)NUM_MS);

    while (remaining > 0 && !h_found) {
        uint64_t launch_keys = remaining;
        uint64_t launch_blocks = (launch_keys + THREADS - 1) / THREADS;
        if (launch_blocks > MAX_BLOCKS_PER_LAUNCH) {
            launch_blocks = MAX_BLOCKS_PER_LAUNCH;
            launch_keys = launch_blocks * THREADS;
            if (launch_keys > remaining) launch_keys = remaining;
        }

        log_msg("[H36] Chunk %d: ms=%llu..%llu (%llu keys, %llu blocks x %d threads)",
                range_idx + 1,
                (unsigned long long)current_start,
                (unsigned long long)(current_start + launch_keys - 1),
                (unsigned long long)launch_keys,
                (unsigned long long)launch_blocks,
                THREADS);

        kernel_timestamp_ms_multi<<<(int)launch_blocks, THREADS>>>(
            current_start, launch_keys,
            (const uint8_t(*)[20])gpu_targets,
            NUM_TARGETS,
            gpu_found, gpu_found_priv
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_found, gpu_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_found) break;

        current_start += launch_keys;
        remaining -= launch_keys;
        range_idx++;
    }

    if (h_found) {
        uint8_t pk[33];
        cudaMemcpy(pk, gpu_found_priv, 33, cudaMemcpyDeviceToHost);
        int target_idx = (int)pk[32];

        const char *addr = (target_idx >= 0 && target_idx < NUM_TARGETS)
            ? TARGET_ADDRS[target_idx] : "unknown";

        log_msg("=== FOUND! H36 target %d: %s ===", target_idx + 1, addr);
        printf("privkey: ");
        for (int i = 0; i < 32; i++) printf("%02x", pk[i]);
        printf("\n");
        fflush(stdout);

        cudaFree(gpu_targets);
        cudaFree(gpu_found);
        cudaFree(gpu_found_priv);
        return target_idx + 1;
    }

    log_msg("[H36] GPU ms sweep done. %llu keys checked, not found.",
            (unsigned long long)NUM_MS);

    cudaFree(gpu_targets);
    cudaFree(gpu_found);
    cudaFree(gpu_found_priv);
    return 0;
}
