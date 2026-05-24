/* DEBUG: Sanity check kernel - prints internals */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

/* We need the definitions from kernels.cuh */
#define FE_L 4
#define D_N0 0xBFD25E8CD0364141ULL
#define D_N1 0xBAAEDCE6AF48A03BULL
#define D_N2 0xFFFFFFFFFFFFFFFEULL
#define D_N3 0xFFFFFFFFFFFFFFFFULL

/* Forward declaration of GPU functions */
extern __device__ int d_pk2h160(const uint64_t *scalar, uint8_t h160[20]);

/* Debug kernel: test privkey=1 and print every step */
__global__ void k_debug_test(void) {
    if(threadIdx.x || blockIdx.x) return;
    
    uint64_t k[4];
    /* Big-endian: pk[31]=1, all others 0 */
    /* k[0] = lowest 64 bits = pk[24..31] */
    k[0] = 1;    /* pk[31]=1 -> lowest 64-bit word = 1 */
    k[1] = 0;    /* pk[16..23] = 0 */
    k[2] = 0;    /* pk[8..15]  = 0 */
    k[3] = 0;    /* pk[0..7]   = 0 (MSB) */
    
    printf("[DEBUG] k[0]=%016llx k[1]=%016llx k[2]=%016llx k[3]=%016llx\n",
        (unsigned long long)k[0], (unsigned long long)k[1],
        (unsigned long long)k[2], (unsigned long long)k[3]);
    
    uint8_t h160[20];
    int ok = d_pk2h160(k, h160);
    printf("[DEBUG] d_pk2h160 returned %d\n", ok);
    
    if(ok) {
        printf("[DEBUG] h160 = ");
        for(int i=0; i<20; i++) printf("%02x", h160[i]);
        printf("\n");
        
        /* Expected (compressed): 751e76e8199196d454941c45d1b3a323f1433bd6 */
        const uint8_t expected[20] = {0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
        int match = 1;
        for(int i=0; i<20; i++) if(h160[i] != expected[i]) { match=0; break; }
        printf("[DEBUG] Compressed match: %s\n", match ? "YES!" : "NO");
        
        /* Expected (uncompressed): 91b24bf9f5288532960ac687abb035127b1d28a5 */
        const uint8_t exp_uncomp[20] = {0x91,0xb2,0x4b,0xf9,0xf5,0x28,0x85,0x32,0x96,0x0a,0xc6,0x87,0xab,0xb0,0x35,0x12,0x7b,0x1d,0x28,0xa5};
        match = 1;
        for(int i=0; i<20; i++) if(h160[i] != exp_uncomp[i]) { match=0; break; }
        printf("[DEBUG] Uncompressed match: %s\n", match ? "YES!" : "NO");
        
        /* Also print full 64 bytes of pubkey */
        printf("[DEBUG] Neither matched - crypto is BUGGY\n");
    } else {
        printf("[DEBUG] d_pk2h160 returned 0 - SCALAR REJECTED\n");
        printf("[DEBUG] k[0]=%016llx check D_N0=%016llx\n",
            (unsigned long long)k[0], (unsigned long long)D_N0);
        printf("[DEBUG] k[0]==0 check: %s\n", 
            (k[0]==0) ? "YES (zero reject)" : "no");
    }
}
