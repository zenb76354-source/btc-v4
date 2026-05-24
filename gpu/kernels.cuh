#ifndef KERNELS_CUH
#define KERNELS_CUH

/* ================================================================
 *  GPU DEVICE FUNCTIONS — SHA256, RIPEMD160, secp256k1 for CUDA
 *  All pure device code, no host calls
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>

/* ---------------------------------------------------------------
 *  SHA-256 device functions (FIPS 180-4)
 * --------------------------------------------------------------- */

__device__ static uint32_t d_rot(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ static uint32_t d_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ static uint32_t d_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ static uint32_t d_s0(uint32_t x) { return d_rot(x,2) ^ d_rot(x,13) ^ d_rot(x,22); }
__device__ static uint32_t d_s1(uint32_t x) { return d_rot(x,6) ^ d_rot(x,11) ^ d_rot(x,25); }
__device__ static uint32_t d_w0(uint32_t x) { return d_rot(x,7) ^ d_rot(x,18) ^ (x >> 3); }
__device__ static uint32_t d_w1(uint32_t x) { return d_rot(x,17) ^ d_rot(x,19) ^ (x >> 10); }

__device__ static const uint32_t d_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ static void d_sha256(const uint8_t *msg, uint32_t len, uint8_t out[32]) {
    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };

    uint8_t buf[128];
    uint32_t i, bidx = 0;
    for (i = 0; i < len; i++) {
        buf[bidx++] = msg[i];
        if (bidx == 64) {
            uint32_t w[64];
            for (int j = 0; j < 16; j++)
                w[j] = ((uint32_t)buf[j*4] << 24) | ((uint32_t)buf[j*4+1] << 16) |
                        ((uint32_t)buf[j*4+2] << 8) | buf[j*4+3];
            for (int j = 16; j < 64; j++)
                w[j] = d_w1(w[j-2]) + w[j-7] + d_w0(w[j-15]) + w[j-16];

            uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
            uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
            for (int j = 0; j < 64; j++) {
                uint32_t t1 = hh + d_s1(e) + d_ch(e,f,g) + d_K[j] + w[j];
                uint32_t t2 = d_s0(a) + d_maj(a,b,c);
                hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            }
            h[0] += a; h[1] += b; h[2] += c; h[3] += d;
            h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
            bidx = 0;
        }
    }

    /* Padding */
    uint64_t bits = len * 8;
    buf[bidx++] = 0x80;
    while (bidx != 56) {
        if (bidx == 64) {
            uint32_t w[64];
            for (int j = 0; j < 16; j++)
                w[j] = ((uint32_t)buf[j*4] << 24) | ((uint32_t)buf[j*4+1] << 16) |
                        ((uint32_t)buf[j*4+2] << 8) | buf[j*4+3];
            for (int j = 16; j < 64; j++)
                w[j] = d_w1(w[j-2]) + w[j-7] + d_w0(w[j-15]) + w[j-16];
            uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
            uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
            for (int j = 0; j < 64; j++) {
                uint32_t t1 = hh + d_s1(e) + d_ch(e,f,g) + d_K[j] + w[j];
                uint32_t t2 = d_s0(a) + d_maj(a,b,c);
                hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            }
            h[0] += a; h[1] += b; h[2] += c; h[3] += d;
            h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
            bidx = 0;
        }
        buf[bidx++] = 0;
    }
    for (int j = 0; j < 8; j++)
        buf[56 + j] = (uint8_t)(bits >> (56 - j*8));

    uint32_t w[64];
    for (int j = 0; j < 16; j++)
        w[j] = ((uint32_t)buf[j*4] << 24) | ((uint32_t)buf[j*4+1] << 16) |
                ((uint32_t)buf[j*4+2] << 8) | buf[j*4+3];
    for (int j = 16; j < 64; j++)
        w[j] = d_w1(w[j-2]) + w[j-7] + d_w0(w[j-15]) + w[j-16];

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int j = 0; j < 64; j++) {
        uint32_t t1 = hh + d_s1(e) + d_ch(e,f,g) + d_K[j] + w[j];
        uint32_t t2 = d_s0(a) + d_maj(a,b,c);
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;

    for (int j = 0; j < 8; j++) {
        out[j*4]   = (uint8_t)(h[j] >> 24);
        out[j*4+1] = (uint8_t)(h[j] >> 16);
        out[j*4+2] = (uint8_t)(h[j] >> 8);
        out[j*4+3] = (uint8_t)(h[j]);
    }
}

/* ---------------------------------------------------------------
 *  RIPEMD-160 device functions
 * --------------------------------------------------------------- */

__device__ static uint32_t d_rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ static uint32_t d_rf1(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
__device__ static uint32_t d_rf2(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
__device__ static uint32_t d_rf3(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
__device__ static uint32_t d_rf4(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
__device__ static uint32_t d_rf5(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

__device__ static void d_ripemd160(const uint8_t *msg, uint32_t len, uint8_t out[20]) {
    uint32_t h[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};

    uint8_t buf[64];
    uint32_t idx = 0, total = len;

    /* Hash in 64-byte chunks */
    for (uint32_t i = 0; i < len; i++) {
        buf[idx++] = msg[i];
        if (idx == 64) {
            /* Process block */
            uint32_t w[16];
            for (int j = 0; j < 16; j++)
                w[j] = ((uint32_t)buf[j*4]) | ((uint32_t)buf[j*4+1] << 8) |
                       ((uint32_t)buf[j*4+2] << 16) | ((uint32_t)buf[j*4+3] << 24);

            uint32_t A = h[0], B = h[1], C = h[2], D = h[3], E = h[4];
            uint32_t A2 = h[0], B2 = h[1], C2 = h[2], D2 = h[3], E2 = h[4];

            /* Round constants */
            const uint32_t k[5] = {0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
            const uint32_t kp[5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000};

            /* Round indices */
            const int r[5][16] = {
                {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
                {7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8},
                {3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12},
                {1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2},
                {4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13}
            };
            const int rp[5][16] = {
                {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12},
                {6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2},
                {15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13},
                {8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14},
                {12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11}
            };
            const int s[5][16] = {
                {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8},
                {7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12},
                {11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5},
                {11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12},
                {9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6}
            };
            const int sp[5][16] = {
                {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6},
                {9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11},
                {9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5},
                {15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8},
                {8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11}
            };

            /* Left line */
            for (int rd = 0; rd < 5; rd++) {
                for (int j = 0; j < 16; j++) {
                    uint32_t fn;
                    switch (rd) {
                        case 0: fn = d_rf1(B,C,D); break;
                        case 1: fn = d_rf2(B,C,D); break;
                        case 2: fn = d_rf3(B,C,D); break;
                        case 3: fn = d_rf4(B,C,D); break;
                        default: fn = d_rf5(B,C,D); break;
                    }
                    uint32_t t = A + fn + w[r[rd][j]] + k[rd];
                    t = d_rol(t, s[rd][j]) + E;
                    A = E; E = D; D = d_rol(C, 10); C = B; B = t;
                }
            }

            /* Right line */
            for (int rd = 0; rd < 5; rd++) {
                for (int j = 0; j < 16; j++) {
                    uint32_t fn;
                    switch (rd) {
                        case 0: fn = d_rf5(B2,C2,D2); break;
                        case 1: fn = d_rf4(B2,C2,D2); break;
                        case 2: fn = d_rf3(B2,C2,D2); break;
                        case 3: fn = d_rf2(B2,C2,D2); break;
                        default: fn = d_rf1(B2,C2,D2); break;
                    }
                    uint32_t t = A2 + fn + w[rp[rd][j]] + kp[rd];
                    t = d_rol(t, sp[rd][j]) + E2;
                    A2 = E2; E2 = D2; D2 = d_rol(C2, 10); C2 = B2; B2 = t;
                }
            }

            uint32_t tmp = h[0] + A + B2;
            h[0] = h[1] + B + C2;
            h[1] = h[2] + C + D2;
            h[2] = h[3] + D + E2;
            h[3] = h[4] + E + A2;
            h[4] = tmp;
            idx = 0;
        }
    }

    /* Padding */
    uint64_t bits = total * 8;
    buf[idx++] = 0x80;
    while (idx != 56) {
        if (idx == 64) {
            uint32_t w[16];
            for (int j = 0; j < 16; j++)
                w[j] = ((uint32_t)buf[j*4]) | ((uint32_t)buf[j*4+1] << 8) |
                       ((uint32_t)buf[j*4+2] << 16) | ((uint32_t)buf[j*4+3] << 24);
            uint32_t A = h[0], B = h[1], C = h[2], D = h[3], E = h[4];
            uint32_t A2 = h[0], B2 = h[1], C2 = h[2], D2 = h[3], E2 = h[4];

            const uint32_t k[5] = {0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
            const uint32_t kp[5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000};
            const int r[5][16] = {
                {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
                {7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8},
                {3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12},
                {1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2},
                {4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13}
            };
            const int rp[5][16] = {
                {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12},
                {6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2},
                {15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13},
                {8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14},
                {12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11}
            };
            const int s[5][16] = {
                {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8},
                {7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12},
                {11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5},
                {11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12},
                {9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6}
            };
            const int sp[5][16] = {
                {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6},
                {9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11},
                {9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5},
                {15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8},
                {8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11}
            };

            for (int rd = 0; rd < 5; rd++) {
                for (int j = 0; j < 16; j++) {
                    uint32_t fn;
                    switch (rd) { case 0: fn=d_rf1(B,C,D); break; case 1: fn=d_rf2(B,C,D); break;
                        case 2: fn=d_rf3(B,C,D); break; case 3: fn=d_rf4(B,C,D); break; default: fn=d_rf5(B,C,D); }
                    uint32_t t = A + fn + w[r[rd][j]] + k[rd];
                    t = d_rol(t, s[rd][j]) + E; A = E; E = D; D = d_rol(C,10); C = B; B = t;
                }
            }
            for (int rd = 0; rd < 5; rd++) {
                for (int j = 0; j < 16; j++) {
                    uint32_t fn;
                    switch (rd) { case 0: fn=d_rf5(B2,C2,D2); break; case 1: fn=d_rf4(B2,C2,D2); break;
                        case 2: fn=d_rf3(B2,C2,D2); break; case 3: fn=d_rf2(B2,C2,D2); break; default: fn=d_rf1(B2,C2,D2); }
                    uint32_t t = A2 + fn + w[rp[rd][j]] + kp[rd];
                    t = d_rol(t, sp[rd][j]) + E2; A2 = E2; E2 = D2; D2 = d_rol(C2,10); C2 = B2; B2 = t;
                }
            }
            uint32_t tmp = h[0] + A + B2;
            h[0] = h[1] + B + C2; h[1] = h[2] + C + D2; h[2] = h[3] + D + E2;
            h[3] = h[4] + E + A2; h[4] = tmp;
            idx = 0;
        }
        buf[idx++] = 0;
    }
    for (int j = 0; j < 8; j++)
        buf[56 + j] = (uint8_t)(bits >> (j * 8));

    uint32_t w[16];
    for (int j = 0; j < 16; j++)
        w[j] = ((uint32_t)buf[j*4]) | ((uint32_t)buf[j*4+1] << 8) |
               ((uint32_t)buf[j*4+2] << 16) | ((uint32_t)buf[j*4+3] << 24);

    uint32_t A = h[0], B = h[1], C = h[2], D = h[3], E = h[4];
    uint32_t A2 = h[0], B2 = h[1], C2 = h[2], D2 = h[3], E2 = h[4];

    const uint32_t k[5] = {0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
    const uint32_t kp[5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000};
    const int r[5][16] = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8},
        {3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12},
        {1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2},
        {4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13}};
    const int rp[5][16] = {{5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12},
        {6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2},
        {15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13},
        {8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14},
        {12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11}};
    const int s[5][16] = {{11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8},
        {7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12},
        {11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5},
        {11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12},
        {9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6}};
    const int sp[5][16] = {{8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6},
        {9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11},
        {9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5},
        {15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8},
        {8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11}};

    for (int rd = 0; rd < 5; rd++)
        for (int j = 0; j < 16; j++) {
            uint32_t fn;
            switch (rd) { case 0: fn=d_rf1(B,C,D); break; case 1: fn=d_rf2(B,C,D); break;
                case 2: fn=d_rf3(B,C,D); break; case 3: fn=d_rf4(B,C,D); break; default: fn=d_rf5(B,C,D); }
            uint32_t t = A + fn + w[r[rd][j]] + k[rd];
            t = d_rol(t, s[rd][j]) + E; A = E; E = D; D = d_rol(C,10); C = B; B = t;
        }
    for (int rd = 0; rd < 5; rd++)
        for (int j = 0; j < 16; j++) {
            uint32_t fn;
            switch (rd) { case 0: fn=d_rf5(B2,C2,D2); break; case 1: fn=d_rf4(B2,C2,D2); break;
                case 2: fn=d_rf3(B2,C2,D2); break; case 3: fn=d_rf2(B2,C2,D2); break; default: fn=d_rf1(B2,C2,D2); }
            uint32_t t = A2 + fn + w[rp[rd][j]] + kp[rd];
            t = d_rol(t, sp[rd][j]) + E2; A2 = E2; E2 = D2; D2 = d_rol(C2,10); C2 = B2; B2 = t;
        }
    uint32_t tmp = h[0] + A + B2;
    h[0] = h[1] + B + C2; h[1] = h[2] + C + D2; h[2] = h[3] + D + E2;
    h[3] = h[4] + E + A2; h[4] = tmp;

    for (int i = 0; i < 5; i++) {
        out[i*4]   = (uint8_t)(h[i]);
        out[i*4+1] = (uint8_t)(h[i] >> 8);
        out[i*4+2] = (uint8_t)(h[i] >> 16);
        out[i*4+3] = (uint8_t)(h[i] >> 24);
    }
}

/* ---------------------------------------------------------------
 *  secp256k1 field arithmetic on GPU (F_p)
 *  p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
 * --------------------------------------------------------------- */

#define P0 0xFFFFFFFEFFFFFC2FULL
#define P1 0xFFFFFFFFFFFFFFFFULL
#define P2 0xFFFFFFFFFFFFFFFFULL
#define P3 0xFFFFFFFFFFFFFFFFULL

typedef struct { uint64_t d[4]; } d_fe;

__device__ static int d_fe_is_zero(const d_fe *a) {
    return a->d[0]==0 && a->d[1]==0 && a->d[2]==0 && a->d[3]==0;
}

__device__ static void d_fe_set(d_fe *r, uint64_t v) {
    r->d[0] = v; r->d[1] = r->d[2] = r->d[3] = 0;
}

__device__ static void d_fe_add(d_fe *r, const d_fe *a, const d_fe *b) {
    uint64_t c = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t s = a->d[i] + b->d[i] + c;
        r->d[i] = s;
        c = (s < a->d[i]) ? 1 : 0;
    }
}

__device__ static void d_fe_sub(d_fe *r, const d_fe *a, const d_fe *b) {
    uint64_t b2 = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t d = a->d[i] - b->d[i] - b2;
        r->d[i] = d;
        b2 = (a->d[i] < b->d[i] + b2) ? 1 : 0;
    }
}

__device__ static void d_fe_reduce(d_fe *r) {
    int ge = 0;
    if (r->d[3] > P3) ge = 1;
    else if (r->d[3] == P3) {
        if (r->d[2] > P2) ge = 1;
        else if (r->d[2] == P2) {
            if (r->d[1] > P1) ge = 1;
            else if (r->d[1] == P1) { if (r->d[0] >= P0) ge = 1; }
        }
    }
    if (ge) {
        uint64_t b = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t sub = (i == 0) ? P0 : 0xFFFFFFFFFFFFFFFFULL;
            uint64_t v = r->d[i] - sub - b;
            r->d[i] = v;
            b = (r->d[i] < sub + b) ? 1 : 0;
        }
    }
}

#endif  /* KERNELS_CUH */