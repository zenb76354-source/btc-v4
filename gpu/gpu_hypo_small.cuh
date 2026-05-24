/* ================================================================
 *  GPU_HYPO_SMALL.CUH — منفصل لكل hypothesis
 *  كل hypothesis = kernel منفصل عشان ما نضغط constant memory
 * ================================================================ */

#ifndef GPU_HYPO_SMALL_CUH
#define GPU_HYPO_SMALL_CUH

#include "kernels.cuh"

/* Constant memory */
__constant__ uint8_t d_targets[8*20];
__constant__ char d_phrases[4096*256];
__constant__ int d_num_phrases;
__constant__ uint8_t d_block_hashes[200000*32];

/* Device strlen */
__device__ static int n_strlen(const char *s) {
    int i=0; while(s[i]) i++; return i;
}

/* CHECK MACRO */
#define CHECK_RET(pk, f, fk) do {                                           \
    if(*(f)) break;                                                         \
    uint64_t __k[4];                                                        \
    for(int __i=0;__i<4;__i++) __k[__i] =                                   \
        ((uint64_t)(pk)[__i*8]<<56)|((uint64_t)(pk)[__i*8+1]<<48)|          \
        ((uint64_t)(pk)[__i*8+2]<<40)|((uint64_t)(pk)[__i*8+3]<<32)|        \
        ((uint64_t)(pk)[__i*8+4]<<24)|((uint64_t)(pk)[__i*8+5]<<16)|        \
        ((uint64_t)(pk)[__i*8+6]<<8)|(uint64_t)(pk)[__i*8+7];              \
    uint8_t __h[20];                                                        \
    if(!d_pk2h160(__k, __h)) break;                                         \
    for(int __t=0;__t<8;__t++){                                             \
        int __m=1;                                                          \
        for(int __i=0;__i<20;__i++) if(__h[__i]!=d_targets[__t*20+__i]){__m=0;break;}\
        if(__m){atomicExch((int*)(f),1);for(int __i=0;__i<4;__i++)fk[__i]=__k[__i];return;}\
    }                                                                       \
} while(0)

#endif /* GPU_HYPO_SMALL_CUH */
