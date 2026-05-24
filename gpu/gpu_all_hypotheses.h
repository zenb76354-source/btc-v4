/* ================================================================
 *  GPU_ALL_HYPOTHESES — كل الفرضيات مولّدة على GPU
 *  GPU يولد SHA256، CPU يفحص بـ secp256k1
 *  بدون إهمال أي فرضية
 * ================================================================ */

#ifndef GPU_ALL_HYPOTHESES_H
#define GPU_ALL_HYPOTHESES_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

/* ================================================================
 *  MAX SIZES
 * ================================================================ */

#define MAX_PHRASES 5000
#define MAX_PHRASE_LEN 256
#define MAX_PID 32768
#define KEY_BYTES 32

/* ================================================================
 *  CONSTANT MEMORY: phrases, PID range, etc.
 *  تحمّل مرة وحدة على GPU
 * ================================================================ */

__device__ __constant__ char d_phrases[MAX_PHRASES * MAX_PHRASE_LEN];
__device__ __constant__ int d_num_phrases = 0;
__device__ __constant__ uint32_t d_pid_start = 0;
__device__ __constant__ uint32_t d_pid_count = 1;

/* ================================================================
 *  HOST FUNCTIONS TO SET CONSTANT MEMORY
 * ================================================================ */

static void gpu_set_phrases(const char **phrases, int n) {
    char hbuf[MAX_PHRASES * MAX_PHRASE_LEN] = {0};
    for (int i = 0; i < n && i < MAX_PHRASES; i++) {
        strncpy(hbuf + i * MAX_PHRASE_LEN, phrases[i], MAX_PHRASE_LEN - 1);
    }
    int num = n > MAX_PHRASES ? MAX_PHRASES : n;
    cudaMemcpyToSymbol(d_phrases, hbuf, MAX_PHRASES * MAX_PHRASE_LEN);
    cudaMemcpyToSymbol(d_num_phrases, &num, sizeof(int));
}

static void gpu_set_pid_range(uint32_t start, uint32_t count) {
    cudaMemcpyToSymbol(d_pid_start, &start, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_pid_count, &count, sizeof(uint32_t));
}

/* ================================================================
 *  KERNEL H28: SHA256(i) sequential — توليد مفاتيح متسلسلة
 *  كل thread يولد SHA256(base + tid)
 * ================================================================ */

__global__ void k_h28_sequential(uint64_t base, uint64_t count, uint8_t *keys_out) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t val = base + tid;
    uint8_t msg[8];
    for (int i = 0; i < 8; i++) {
        msg[7 - i] = (uint8_t)(val >> (i * 8));
    }

    /* SHA256(val as 8-byte big-endian) */
    uint32_t H[8] = {0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,
                     0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19};
    __shared__ uint32_t sK[64];
    if (threadIdx.x < 64) sK[threadIdx.x] = d_K256[threadIdx.x];
    __syncthreads();

    uint8_t block[64]; uint64_t bits = 64;
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (int i = 0; i < 8; i++) block[i] = msg[i];
    block[8] = 0x80;
    for (int i = 0; i < 8; i++) block[63-i] = (uint8_t)(bits >> (i*8));

    uint32_t W[64], a,b,c,d,e,f,g,h,T1,T2;
    for (int i = 0; i < 16; i++)
        W[i] = ((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)|
               ((uint32_t)block[i*4+2]<<8)|block[i*4+3];
    for (int i = 16; i < 64; i++)
        W[i] = d_w1(W[i-2]) + W[i-7] + d_w0(W[i-15]) + W[i-16];

    a=H[0];b=H[1];c=H[2];d=H[3];e=H[4];f=H[5];g=H[6];h=H[7];
    for (int i = 0; i < 64; i++) {
        T1 = h + d_s1(e) + d_ch(e,f,g) + sK[i] + W[i];
        T2 = d_s0(a) + d_maj(a,b,c);
        h=g;g=f;f=e;e=d+T1;d=c;c=b;b=a;a=T1+T2;
    }
    H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;

    uint8_t *out = keys_out + tid * KEY_BYTES;
    for (int i = 0; i < 8; i++) {
        out[i*4]=(uint8_t)(H[i]>>24);out[i*4+1]=(uint8_t)(H[i]>>16);
        out[i*4+2]=(uint8_t)(H[i]>>8);out[i*4+3]=(uint8_t)(H[i]);
    }
}

/* ================================================================
 *  KERNEL H08: Block hashes (H256 of block headers and creation dates)
 *  يحتاج 200K block header hashes
 * ================================================================ */

__global__ void k_h08_blockhashes(uint64_t count, uint8_t *keys_out) {
    /* محجوزة — سنحتاج block headers hash من host */
}

/* ================================================================
 *  HOST LAUNCHERS
 * ================================================================ */

#include "../common/check.h"

static int launch_h28(secp256k1_context *ctx, uint64_t count, uint64_t base) {
    uint8_t *gpu_keys;
    cudaMalloc(&gpu_keys, count * KEY_BYTES);

    uint64_t blocks = (count + 255) / 256;
    k_h28_sequential<<<(int)blocks, 256>>>(base, count, gpu_keys);

    uint8_t *h_keys = (uint8_t*)malloc(count * KEY_BYTES);
    cudaMemcpy(h_keys, gpu_keys, count * KEY_BYTES, cudaMemcpyDeviceToHost);

    log_msg("[GPU-H28] Sequential SHA256(i): %llu keys", (unsigned long long)count);
    int found = 0;
    for (uint64_t i = 0; i < count && !found; i++) {
        if (check_privkey_multi(ctx, h_keys + i * KEY_BYTES)) {
            found = 1;
        }
    }

    cudaFree(gpu_keys);
    free(h_keys);
    return found;
}

/* ================================================================
 *  MASTER LAUNCHER — كل الفرضيات الممكنة على GPU
 * ================================================================ */

int gpu_launch_all_hypotheses(secp256k1_context *ctx) {
    int found = 0;

    /* H36: Timestamp ms (2009-2011) — موجود في timestamp_sweep.cu */
    log_msg("[GPU-ALL] Launching all GPU hypotheses...");

    /* H28: SHA256(0) .. SHA256(N) */
    if (!found) found = launch_h28(ctx, 2000000, 0);

    /* نضيف باقي الفرضيات... (قيد التطوير) */

    return found;
}

#endif /* GPU_ALL_HYPOTHESES_H */
