/* ================================================================
 *  MAIN.CU — Minimal test: just SHA256 + H28 sequential
 *  لا constant data كبير — كل جداول RIPEMD160 على الـ stack
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* SHA256 */
__device__ static uint32_t d_rot(uint32_t x,int n){return(x<<n)|(x>>(32-n));}
__device__ static uint32_t d_ch(uint32_t x,uint32_t y,uint32_t z){return(x&y)^(~x&z);}
__device__ static uint32_t d_maj(uint32_t x,uint32_t y,uint32_t z){return(x&y)^(x&z)^(y&z);}
__device__ static uint32_t d_s0(uint32_t x){return d_rot(x,2)^d_rot(x,13)^d_rot(x,22);}
__device__ static uint32_t d_s1(uint32_t x){return d_rot(x,6)^d_rot(x,11)^d_rot(x,25);}
__device__ static uint32_t d_w0(uint32_t x){return d_rot(x,7)^d_rot(x,18)^(x>>3);}
__device__ static uint32_t d_w1(uint32_t x){return d_rot(x,17)^d_rot(x,19)^(x>>10);}

__device__ static void d_sha256(const uint8_t *m,uint32_t l,uint8_t o[32]){
    uint32_t H[8]={0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19};
    uint32_t W[64],a,b,c,d,e,f,g,h,T1,T2; uint8_t blk[64]; uint64_t bits=l*8;
    for(int i=0;i<64;i++)blk[i]=0;
    for(uint32_t i=0;i<l;i++)blk[i]=m[i]; blk[l]=0x80;
    for(int i=0;i<8;i++)blk[63-i]=(uint8_t)(bits>>(i*8));
    for(int i=0;i<16;i++)W[i]=((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)|((uint32_t)blk[i*4+2]<<8)|blk[i*4+3];
    const uint32_t K[64]={0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
        0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
        0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
        0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
        0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
        0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
        0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
        0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2};
    for(int i=16;i<64;i++)W[i]=d_w1(W[i-2])+W[i-7]+d_w0(W[i-15])+W[i-16];
    a=H[0];b=H[1];c=H[2];d=H[3];e=H[4];f=H[5];g=H[6];h=H[7];
    for(int i=0;i<64;i++){T1=h+d_s1(e)+d_ch(e,f,g)+K[i]+W[i];T2=d_s0(a)+d_maj(a,b,c);
        h=g;g=f;f=e;e=d+T1;d=c;c=b;b=a;a=T1+T2;}
    H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
    for(int i=0;i<8;i++){o[i*4]=(uint8_t)(H[i]>>24);o[i*4+1]=(uint8_t)(H[i]>>16);o[i*4+2]=(uint8_t)(H[i]>>8);o[i*4+3]=(uint8_t)(H[i]);}
}

/* Simple H28 test — just SHA256(i) */
__global__ void k_test(uint64_t cn, volatile int *f, volatile uint64_t *fk) {
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=cn||*f) return;
    uint32_t v=(uint32_t)tid;
    uint8_t m[4]={(uint8_t)(v>>24),(uint8_t)(v>>16),(uint8_t)(v>>8),(uint8_t)v};
    uint8_t pk[32]; d_sha256(m,4,pk);
    /* Just store result — no ECC for this test */
    if(tid==0){ fk[0]=(uint64_t)pk[0]; }
}

int main() {
    printf("Test build starting...\n");
    int *d_flag; uint64_t *d_fk;
    cudaMalloc(&d_flag,4); cudaMalloc(&d_fk,32);
    int hf=0; cudaMemcpy(d_flag,&hf,4,cudaMemcpyHostToDevice);
    k_test<<<100,256>>>(25600,d_flag,d_fk);
    cudaDeviceSynchronize();
    printf("Done\n");
    return 0;
}
