/* ================================================================
 *  GPU_HYPO_SMALL.CUH — All small hypotheses
 *  صار kernel واحد multilevel: يأخذ string buffer من __constant__
 *  
 *  كل الـ string data موجودة في main.cu → cudaMemcpyToSymbol → 
 *  kernel واحد يقرأ الـ buffers ويولد كل التخمينات
 * ================================================================ */

#ifndef GPU_HYPO_SMALL_CUH
#define GPU_HYPO_SMALL_CUH

#include "kernels.cuh"

/* Device-side strlen */
__device__ static int n_strlen(const char *s) {
    int i=0; while(s[i]) i++; return i;
}

__constant__ uint8_t d_targets[8*20];
__constant__ char d_phrases[4096*256];
__constant__ int d_num_phrases;
__constant__ uint8_t d_block_hashes[200000*32];

/* String buffers: packed arrays (each null-terminated) */
#define DICT_MAX 8192

__constant__ char d_dict[DICT_MAX];

/* CHECK MACRO (compact) */
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

/* Master kernel: arg = hypothesis type, dict[] contains strings each null-terminated */
__global__ void k_hypo_master(
    uint64_t type,
    uint64_t count,
    volatile int *found,
    volatile uint64_t *found_key
) {
    if(type == 0){
        /* H21: empty string */
        if(threadIdx.x||blockIdx.x) return;
        uint8_t pk[32]; d_sha256((const uint8_t*)"",0,pk); CHECK_RET(pk,found,found_key);
    }
    else if(type == 11){
        /* H11: weak keys — dictionary items, each SHA256d */
        if(threadIdx.x||blockIdx.x) return;
        uint8_t pk[32]; int pos=0;
        while(pos < DICT_MAX && d_dict[pos]){
            d_sha256((const uint8_t*)(d_dict+pos), n_strlen(d_dict+pos), pk);
            CHECK_RET(pk,found,found_key);
            pos += n_strlen(d_dict+pos) + 1;
        }
    }
    else if(type == 14){
        /* H14: timestamp strings */
        if(threadIdx.x||blockIdx.x) return;
        uint8_t pk[32]; int pos=0;
        while(pos < DICT_MAX && d_dict[pos]){
            d_sha256((const uint8_t*)(d_dict+pos), n_strlen(d_dict+pos), pk);
            CHECK_RET(pk,found,found_key);
            pos += n_strlen(d_dict+pos) + 1;
        }
    }
    else if(type == 15){
        /* H15: date formats */
        if(threadIdx.x||blockIdx.x) return;
        uint8_t pk[32]; int pos=0;
        while(pos < DICT_MAX && d_dict[pos]){
            d_sha256((const uint8_t*)(d_dict+pos), n_strlen(d_dict+pos), pk);
            CHECK_RET(pk,found,found_key);
            pos += n_strlen(d_dict+pos) + 1;
        }
    }
    else if(type == 41){
        /* H41: leet words (plain + capitalized) */
        if(threadIdx.x||blockIdx.x) return;
        uint8_t pk[32]; int pos=0;
        while(pos < DICT_MAX && d_dict[pos]){
            d_sha256((const uint8_t*)(d_dict+pos), n_strlen(d_dict+pos), pk);
            CHECK_RET(pk,found,found_key);
            /* Capitalized */
            char b[128]; const char *s=d_dict+pos; int j=0;
            b[0]=(s[0]>='a'&&s[0]<='z')?(s[0]-32):s[0];
            while(s[++j]) b[j]=s[j];
            d_sha256((const uint8_t*)b,n_strlen(s),pk);
            CHECK_RET(pk,found,found_key);
            pos += n_strlen(s) + 1;
        }
    }
    else if(type == 4226 || type == 4230 || type == 4327 ||
            type == 4325 || type == 4333 || type == 4328 ||
            type == 4329 || type == 4330 || type == 4331 ||
            type == 4332 || type == 4334 || type == 4326){
        /* Generic: iterate dict, SHA256 each item */
        if(threadIdx.x||blockIdx.x) return;
        uint8_t pk[32]; int pos=0;
        while(pos < DICT_MAX && d_dict[pos]){
            d_sha256((const uint8_t*)(d_dict+pos), n_strlen(d_dict+pos), pk);
            CHECK_RET(pk,found,found_key);
            pos += n_strlen(d_dict+pos) + 1;
        }
    }
    else if(type == 4229){
        /* H29: bitcoin+suffix — use combined dict with prefix_suffix */
        if(threadIdx.x||blockIdx.x) return;
        uint8_t pk[32]; int pos=0;
        while(pos < DICT_MAX && d_dict[pos]){
            d_sha256((const uint8_t*)(d_dict+pos), n_strlen(d_dict+pos), pk);
            CHECK_RET(pk,found,found_key);
            pos += n_strlen(d_dict+pos) + 1;
        }
    }
    else if(type == 4235){
        /* H35: periodic patterns */
        if(threadIdx.x||blockIdx.x) return;
        uint8_t pk[32]; int pos=0;
        while(pos < DICT_MAX && d_dict[pos]){
            d_sha256((const uint8_t*)(d_dict+pos), n_strlen(d_dict+pos), pk);
            CHECK_RET(pk,found,found_key);
            pos += n_strlen(d_dict+pos) + 1;
        }
    }
    else if(type == 4228){
        /* H28: sequential SHA256(i) — multi-thread */
        uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
        if(tid>=count||*found) return;
        uint32_t v32=(uint32_t)(tid&0xFFFFFFFF);
        uint8_t m[4]={(uint8_t)(v32>>24),(uint8_t)(v32>>16),(uint8_t)(v32>>8),(uint8_t)v32};
        uint8_t pk[32]; d_sha256(m,4,pk);
        CHECK_RET(pk,found,found_key);
    }
    else if(type == 4220){
        /* H20: srand(time(NULL)) — use thread as seed */
        uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
        if(tid>=count||*found) return;
        uint32_t seed32=(uint32_t)(tid);
        uint8_t m[4]={(uint8_t)(seed32>>24),(uint8_t)(seed32>>16),(uint8_t)(seed32>>8),(uint8_t)seed32};
        uint8_t pk[32]; d_sha256(m,4,pk);
        CHECK_RET(pk,found,found_key);
    }
    else if(type == 4203){
        /* H03: timestamp+PID (sweep) */
        uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
        if(tid>=count||*found) return;
        /* Use tid as combined ts (high) + pid (low) */
        uint64_t ts = tid >> 15;
        uint16_t pid = (uint16_t)(tid & 0x7FFF);
        uint8_t m[10];
        for(int i=0;i<8;i++) m[7-i] = (uint8_t)(ts >> (i*8));
        m[8] = (uint8_t)(pid>>8); m[9] = (uint8_t)(pid);
        uint8_t pk[32]; d_sha256(m,10,pk);
        CHECK_RET(pk,found,found_key);
    }
    else if(type == 4236){
        /* H36: ms timestamp sweep */
        uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
        if(tid>=count||*found) return;
        uint64_t ms = tid;  /* start_ms offset added by caller */
        uint8_t m[8]; for(int i=0;i<8;i++) m[7-i] = (uint8_t)(ms>>(i*8));
        uint8_t pk[32]; d_sha256(m,8,pk);
        CHECK_RET(pk,found,found_key);
    }
}

#endif /* GPU_HYPO_SMALL_CUH */
