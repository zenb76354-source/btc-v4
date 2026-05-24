/* ================================================================
 *  KERNELS_API.H — Kernel declarations (extern) for main.cu
 *  Main.cu includes ONLY this header, NOT kernels.cuh (which has ECC tables)
 * ================================================================ */

#ifndef KERNELS_API_H
#define KERNELS_API_H

#include <cuda_runtime.h>
#include <stdint.h>

/* __constant__ arrays (allocated in main.cu, used in kernels_code.cu) */
extern __constant__ uint8_t d_targets[160];
extern __constant__ char d_dict[8192];
extern __constant__ char d_phrases[1048576];
extern __constant__ int d_num_phrases;
extern __constant__ uint8_t d_block_hashes[6400000];

/* Kernel declarations */
__global__ void k21(void *f, void *fk);
__global__ void k11(void *f, void *fk);
__global__ void k14(void *f, void *fk);
__global__ void k15(void *f, void *fk);
__global__ void k41(void *f, void *fk);
__global__ void k_gentxt(void *f, void *fk);
__global__ void k28(uint64_t st, uint64_t cn, void *f, void *fk);
__global__ void k3(uint64_t cn, void *f, void *fk);
__global__ void k20(uint64_t cn, void *f, void *fk);
__global__ void k36(uint64_t st, uint64_t cn, void *f, void *fk);
__global__ void k8(int nb, void *f, void *fk);
__global__ void k1(void *f, void *fk);
__global__ void k9(void *f, void *fk);
__global__ void k18(void *f, void *fk);

#endif /* KERNELS_API_H */
