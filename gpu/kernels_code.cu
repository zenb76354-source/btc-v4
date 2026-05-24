/* ================================================================
 *  KERNELS_CODE.CU — All GPU kernels for BTC recovery
 *  ملف منفصل لـ device compilation مع main.cu
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>

/* ننقل تعريف check.h هنا — مقارنة 8 targets */
__constant__ uint8_t d_targets[8*20];

/* We only include the device functions, no __global__ here yet */
#include "kernels.cuh"

/* Constant memory */
__constant__ char d_dict[8192];
__constant__ char d_phrases[4096*256];
__constant__ int d_num_phrases;
__constant__ uint8_t d_block_hashes[200000*32];

/* Device strlen */
__device__ static int n_strlen(const char *s) { int i=0; while(s[i]) i++; return i; }

/* CHECK MACRO */
#define CHK(pk,f,fk) do { \
    if(*(f)) break; \
    uint64_t __k[4]; for(int i=0;i<4;i++)__k[i]= \
        ((uint64_t)(pk)[i*8]<<56)|((uint64_t)(pk)[i*8+1]<<48)|((uint64_t)(pk)[i*8+2]<<40)|((uint64_t)(pk)[i*8+3]<<32)|\
        ((uint64_t)(pk)[i*8+4]<<24)|((uint64_t)(pk)[i*8+5]<<16)|((uint64_t)(pk)[i*8+6]<<8)|(pk)[i*8+7]; \
    uint8_t __h[20]; if(!d_pk2h160(__k,__h))break; \
    for(int t=0;t<8;t++){int m=1; for(int i=0;i<20;i++)if(__h[i]!=d_targets[t*20+i]){m=0;break;} \
        if(m){atomicExch((int*)f,1);for(int i=0;i<4;i++)fk[i]=__k[i];return;}} \
} while(0)

/* H21 */
__global__ void k21(void *f, void *fk) { if(threadIdx.x||blockIdx.x)return;
    uint8_t pk[32]; d_sha256((const uint8_t*)"",0,pk); CHK(pk,f,fk); }

/* H11 dict */
__global__ void k11(void *f, void *fk) { if(threadIdx.x||blockIdx.x)return;
    uint8_t pk[32]; int p=0; while(d_dict[p]){int sl=n_strlen(d_dict+p);
        d_sha256((const uint8_t*)(d_dict+p),sl,pk);CHK(pk,f,fk);p+=sl+1;} }

/* H14 */
__global__ void k14(void *f, void *fk) { if(threadIdx.x||blockIdx.x)return;
    uint8_t pk[32]; int p=0; while(d_dict[p]){int sl=n_strlen(d_dict+p);
        d_sha256((const uint8_t*)(d_dict+p),sl,pk);CHK(pk,f,fk);p+=sl+1;} }

/* H15 */
__global__ void k15(void *f, void *fk) { if(threadIdx.x||blockIdx.x)return;
    uint8_t pk[32]; int p=0; while(d_dict[p]){int sl=n_strlen(d_dict+p);
        d_sha256((const uint8_t*)(d_dict+p),sl,pk);CHK(pk,f,fk);p+=sl+1;} }

/* H41 dict + caps */
__global__ void k41(void *f, void *fk) { if(threadIdx.x||blockIdx.x)return;
    uint8_t pk[32]; int p=0; while(d_dict[p]){
        const char *s=d_dict+p;int sl=n_strlen(s);
        d_sha256((const uint8_t*)s,sl,pk);CHK(pk,f,fk);
        char b[128];b[0]=(s[0]>='a'&&s[0]<='z')?(s[0]-32):s[0];
        for(int j=1;j<sl;j++)b[j]=s[j];
        d_sha256((const uint8_t*)b,sl,pk);CHK(pk,f,fk);p+=sl+1;} }

/* H42-H35, H26, H27, H29, H25 — all generic dict */
__global__ void k_gentxt(void *f, void *fk) { if(threadIdx.x||blockIdx.x)return;
    uint8_t pk[32]; int p=0; while(d_dict[p]){int sl=n_strlen(d_dict+p);
        d_sha256((const uint8_t*)(d_dict+p),sl,pk);CHK(pk,f,fk);p+=sl+1;} }

/* H28: sequential */
__global__ void k28(uint64_t st,uint64_t cn,void *f,void *fk) {
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=cn||*(int*)f)return;
    uint32_t v=(uint32_t)(st+tid); uint8_t m[4]={(uint8_t)(v>>24),(uint8_t)(v>>16),(uint8_t)(v>>8),(uint8_t)v};
    uint8_t pk[32]; d_sha256(m,4,pk); CHK(pk,f,fk); }

/* H03: ts+PID */
__global__ void k3(uint64_t cn,void *f,void *fk) {
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=cn||*(int*)f)return;
    uint8_t m[10]; for(int i=0;i<8;i++) m[7-i]=(uint8_t)(tid>>(i*8));
    m[8]=(uint8_t)((tid>>15)&0xFF); m[9]=(uint8_t)((tid>>7)&0xFF);
    uint8_t pk[32]; d_sha256(m,10,pk); CHK(pk,f,fk); }

/* H20: srand(time) */
__global__ void k20(uint64_t cn,void *f,void *fk) {
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=cn||*(int*)f)return;
    uint32_t v=(uint32_t)(tid&0xFFFFFFFF); uint8_t m[4]={(uint8_t)(v>>24),(uint8_t)(v>>16),(uint8_t)(v>>8),(uint8_t)v};
    uint8_t pk[32]; d_sha256(m,4,pk); CHK(pk,f,fk); }

/* H36: timestamp ms sweep */
__global__ void k36(uint64_t st,uint64_t cn,void *f,void *fk) {
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=cn||*(int*)f)return;
    uint64_t ms=st+tid; uint8_t m[8]; for(int i=0;i<8;i++) m[7-i]=(uint8_t)(ms>>(i*8));
    uint8_t pk[32]; d_sha256(m,8,pk); CHK(pk,f,fk); }

/* H08: block hashes */
__global__ void k8(int nb,void *f,void *fk) {
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=(uint64_t)nb||*(int*)f)return;
    uint8_t pk[32]; d_sha256(d_block_hashes+tid*32,32,pk); CHK(pk,f,fk); }

/* H01: phrases × 7 variants */
__global__ void k1(void *f, void *fk) {
    int n=d_num_phrases, tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n||*(int*)f)return;
    const char *ph=d_phrases+(uint64_t)tid*256; int pl=n_strlen(ph);
    uint8_t pk[32]; char b[512]; for(int i=0;i<pl;i++)b[i]=ph[i];
    d_sha256((const uint8_t*)ph,pl,pk);CHK(pk,f,fk);
    b[pl]='1';b[pl+1]='2';b[pl+2]='3';b[pl+3]=0;
    d_sha256((const uint8_t*)b,pl+3,pk);CHK(pk,f,fk);
    b[pl]='1';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,pk);CHK(pk,f,fk);
    b[pl]='!';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,pk);CHK(pk,f,fk);
    b[pl]='@';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,pk);CHK(pk,f,fk);
    b[pl]='?';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,pk);CHK(pk,f,fk);
    b[pl]='.';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,pk);CHK(pk,f,fk);
}

/* H09: phrase + year */
__global__ void k9(void *f, void *fk) {
    int n=d_num_phrases, tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n||*(int*)f)return;
    const char *ph=d_phrases+(uint64_t)tid*256; int pl=n_strlen(ph);
    uint8_t pk[32]; char b[512]; for(int i=0;i<pl;i++)b[i]=ph[i];
    for(int yr=2009;yr<=2013;yr++){if(*(int*)f)return;
        int yy=yr; char y[8];int yl=0,t=yy;while(t){y[yl++]=t%10+'0';t/=10;}
        for(int i=0;i<yl/2;i++){char tp=y[i];y[i]=y[yl-1-i];y[yl-1-i]=tp;}
        for(int i=0;i<yl;i++)b[pl+i]=y[i];b[pl+yl]=0;
        d_sha256((const uint8_t*)b,pl+yl,pk);CHK(pk,f,fk);}
}

/* H18: phrase pairs */
__global__ void k18(void *f, void *fk) {
    int n=d_num_phrases; if(n<2||*(int*)f)return;
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t max=(uint64_t)n*(n-1)/2; if(tid>=max||*(int*)f)return;
    int i=0; uint64_t t=tid; while((uint64_t)(i+1)*i/2<=t)i++; i--;
    int j=(int)(t-(uint64_t)i*(i+1)/2)+i+1;
    if(j>=n||i>=n)return;
    const char *a=d_phrases+(uint64_t)i*256; const char *b=d_phrases+(uint64_t)j*256;
    char buf[512]; int al=n_strlen(a),bl=n_strlen(b);
    for(int x=0;x<al;x++)buf[x]=a[x];buf[al]=' ';
    for(int x=0;x<bl;x++)buf[al+1+x]=b[x];buf[al+1+bl]=0;
    uint8_t pk[32]; d_sha256((const uint8_t*)buf,al+1+bl,pk);CHK(pk,f,fk);
}

/* Dictionary wrapper: sends dict, runs correct kernel, returns found */
extern "C" int run_dict_gpu(int hypo_id, const char *host_dict, int dict_size,
                            volatile int *d_flag, volatile uint64_t *d_fk) {
    cudaMemcpyToSymbol(d_dict, host_dict, 8192);
    int h_flag=0; cudaMemcpy(d_flag,&h_flag,sizeof(int),cudaMemcpyHostToDevice);
    uint64_t hz[4]={0}; cudaMemcpy(d_fk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
    switch(hypo_id){
        case 21: k21<<<1,1>>>(d_flag,d_fk); break;
        case 11: k11<<<1,1>>>(d_flag,d_fk); break;
        case 14: k14<<<1,1>>>(d_flag,d_fk); break;
        case 15: k15<<<1,1>>>(d_flag,d_fk); break;
        case 41: k41<<<1,1>>>(d_flag,d_fk); break;
        default: k_gentxt<<<1,1>>>(d_flag,d_fk); break;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(&h_flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost);
    if(h_flag){ uint64_t h_fk[4]; cudaMemcpy(h_fk,d_fk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        return (int)(h_fk[0]&0xFFFFFFFF); /* placeholder — returns 1 if found */ }
    return 0;
}
