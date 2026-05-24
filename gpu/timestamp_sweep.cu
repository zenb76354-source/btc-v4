/* ================================================================
 *  H36 — GPU Timestamp Millisecond Sweep (2009-2011)
 *  
 *  GPU generates candidate private keys from millisecond timestamps:
 *    key = SHA256(ms_since_epoch_big_endian)
 *  
 *  CPU verifies each key using real secp256k1
 *  
 *  Range: 2009-01-01 00:00:00.000 → 2012-01-01 00:00:00.000
 *  = 94,675,968,000 ms ≈ 94.7 billion keys
 *  On RTX 5090 (~100B keys/s) → ~1 second total
 *
 *  Compile part of: nvcc -O2 -arch=sm_100 -std=c++11 \
 *    main.cu cpu/hypotheses.cu gpu/timestamp_sweep.cu \
 *    -lsecp256k1 -lssl -lcrypto -Xcompiler -fopenmp
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common/targets.h"
#include "../common/check.h"
#include "kernels_api.h"

/* ---------------------------------------------------------------
 *  GPU KERNEL: kernel_timestamp_gen
 *  
 *  Each thread handles ONE millisecond timestamp.
 *  SHA256(timestamp_be) → candidate private key
 *  Writes key to output buffer for CPU verification
 * --------------------------------------------------------------- */

__global__ void kernel_timestamp_gen(
    uint64_t start_ms,
    uint64_t num_ms,
    uint8_t *out_keys      /* [num_ms][32] */
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ms) return;

    uint64_t time_ms = start_ms + idx;

    /* Build 8-byte big-endian timestamp */
    uint8_t msg[8];
    for (int i = 0; i < 8; i++)
        msg[7 - i] = (uint8_t)(time_ms >> (i * 8));

    /* SHA256 → private key, store in output buffer */
    d_sha256(msg, 8, out_keys + idx * 32);
}

/* ---------------------------------------------------------------
 *  Host function: H36 — GPU Timestamp ms Sweep with CPU verify
 *  
 *  Uses GPU as high-speed key generator, CPU verifies each.
 *  Returns 1 if found, 0 otherwise.
 * --------------------------------------------------------------- */

int h36_timestamp_ms_sweep(void) {
    const uint64_t START_MS = 1230768000000ULL;   /* 2009-01-01 00:00:00.000 */
    const uint64_t END_MS   = 1325376000000ULL;   /* 2012-01-01 00:00:00.000 */
    const uint64_t NUM_MS   = END_MS - START_MS;  /* 94,675,968,000 */

    secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    const int THREADS = 256;
    const uint64_t KEYS_PER_BATCH = 500000000ULL;  /* 500M keys per batch → ~16GB max */
    uint8_t *gpu_keys;
    cudaMalloc(&gpu_keys, KEYS_PER_BATCH * 32);

    uint8_t *h_keys = (uint8_t*)malloc(KEYS_PER_BATCH * 32);
    if (!h_keys) {
        log_msg("[H36] malloc failed");
        secp256k1_context_destroy(ctx);
        return 0;
    }

    log_msg("[H36] GPU ms sweep: %llu ms from 2009-01-01 to 2012-01-01",
            (unsigned long long)NUM_MS);

    uint64_t processed = 0;
    int found = 0;

    #ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
    #endif

    while (processed < NUM_MS && !found) {
        uint64_t batch_size = KEYS_PER_BATCH;
        if (processed + batch_size > NUM_MS)
            batch_size = NUM_MS - processed;

        uint64_t blocks = (batch_size + THREADS - 1) / THREADS;

        kernel_timestamp_gen<<<(int)blocks, THREADS>>>(
            START_MS + processed, batch_size, gpu_keys
        );
        cudaDeviceSynchronize();

        cudaMemcpy(h_keys, gpu_keys, batch_size * 32, cudaMemcpyDeviceToHost);

        /* CPU verification with OpenMP — each thread has own secp256k1 ctx */
        volatile int omp_found = 0;
        uint64_t omp_found_idx = 0;
        uint8_t omp_found_key[32];

        int max_t = 1;
        #ifdef _OPENMP
        max_t = omp_get_max_threads();
        #endif
        secp256k1_context **tctxs = (secp256k1_context**)malloc(max_t * sizeof(secp256k1_context*));
        for (int ti = 0; ti < max_t; ti++)
            tctxs[ti] = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

        #pragma omp parallel for
        for (uint64_t ki = 0; ki < batch_size; ki++) {
            if (omp_found) continue;
            #ifdef _OPENMP
            secp256k1_context *tctx = tctxs[omp_get_thread_num()];
            #else
            secp256k1_context *tctx = tctxs[0];
            #endif
            if (check_privkey_multi(tctx, h_keys + ki * 32)) {
                #pragma omp critical
                {
                    if (!omp_found) {
                        omp_found = 1;
                        omp_found_idx = processed + ki;
                        memcpy(omp_found_key, h_keys + ki * 32, 32);
                        found = 1;
                    }
                }
            }
        }

        for (int ti = 0; ti < max_t; ti++)
            secp256k1_context_destroy(tctxs[ti]);
        free(tctxs);

        processed += batch_size;

        if (omp_found) {
            log_msg("[H36] FOUND at offset %llu!", (unsigned long long)omp_found_idx);
        }

        log_msg("[H36] %llu / %llu (%.1f%%)",
                (unsigned long long)processed,
                (unsigned long long)NUM_MS,
                100.0 * processed / NUM_MS);
    }

    if (found) {
        log_msg("[H36] FOUND in batch at offset %llu!", (unsigned long long)processed);
    } else {
        log_msg("[H36] GPU ms sweep done. %llu keys checked, not found.",
                (unsigned long long)NUM_MS);
    }

    cudaFree(gpu_keys);
    free(h_keys);
    secp256k1_context_destroy(ctx);
    return found;
}
