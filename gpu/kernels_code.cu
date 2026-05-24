/* ================================================================
 *  KERNELS_CODE.CU — All GPU kernels (separate file for -dc)
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>

/* extern __constant__ from main.cu */
extern __constant__ uint8_t d_targets[160];
extern __constant__ char d_dict[8192];
extern __constant__ char d_phrases[1048576];
extern __constant__ int d_num_phrases;
extern __constant__ uint8_t d_block_hashes[6400000];

#include "kernels.cuh"

/* Device strlen */
__device__ static int n_strlen(const char *s){int i=0;while(s[i])i++;return i;}

/* CHK: int* f, uint64_t* fk */
/* __device__ check function */
__device__ static void check_return(const uint8_t *pk, void *fp, void *fkp) {
    int *f = (int*)fp; uint64_t *fk = (uint64_t*)fkp;
    if(*f) return;
    uint64_t k[4];
    for(int i=0;i<4;i++) k[i]=((uint64_t)pk[i*8]<<56)|((uint64_t)pk[i*8+1]<<48)|((uint64_t)pk[i*8+2]<<40)|((uint64_t)pk[i*8+3]<<32)|((uint64_t)pk[i*8+4]<<24)|((uint64_t)pk[i*8+5]<<16)|((uint64_t)pk[i*8+6]<<8)|pk[i*8+7];
    uint8_t h[20]; if(!d_pk2h160(k,h)) return;
    for(int t=0;t<8;t++){int m=1; for(int i=0;i<20;i++) if(h[i]!=d_targets[t*20+i]){m=0;break;}
        if(m){atomicExch(f,1);for(int i=0;i<4;i++)fk[i]=k[i];return;}}
}

/* H21 */
__global__ void k21(void*f,void*fk){if(threadIdx.x||blockIdx.x)return;
    uint8_t p[32];d_sha256((const uint8_t*)"",0,p);check_return(p,f,fk);}

/* H11 dict */
__global__ void k11(void*f,void*fk){if(threadIdx.x||blockIdx.x)return;
    uint8_t p[32];int pos=0;while(d_dict[pos]){int sl=n_strlen(d_dict+pos);
    d_sha256((const uint8_t*)(d_dict+pos),sl,p);check_return(p,f,fk);pos+=sl+1;}}

/* H14 */
__global__ void k14(void*f,void*fk){if(threadIdx.x||blockIdx.x)return;
    uint8_t p[32];int pos=0;while(d_dict[pos]){int sl=n_strlen(d_dict+pos);
    d_sha256((const uint8_t*)(d_dict+pos),sl,p);check_return(p,f,fk);pos+=sl+1;}}

/* H15 */
__global__ void k15(void*f,void*fk){if(threadIdx.x||blockIdx.x)return;
    uint8_t p[32];int pos=0;while(d_dict[pos]){int sl=n_strlen(d_dict+pos);
    d_sha256((const uint8_t*)(d_dict+pos),sl,p);check_return(p,f,fk);pos+=sl+1;}}

/* H41 dict+caps */
__global__ void k41(void*f,void*fk){if(threadIdx.x||blockIdx.x)return;
    uint8_t p[32];int pos=0;while(d_dict[pos]){const char*s=d_dict+pos;int sl=n_strlen(s);
    d_sha256((const uint8_t*)s,sl,p);check_return(p,f,fk);
    char b[128];b[0]=(s[0]>='a'&&s[0]<='z')?(s[0]-32):s[0];
    for(int j=1;j<sl;j++)b[j]=s[j];d_sha256((const uint8_t*)b,sl,p);check_return(p,f,fk);pos+=sl+1;}}

/* Generic dict */
__global__ void k_gentxt(void*f,void*fk){if(threadIdx.x||blockIdx.x)return;
    uint8_t p[32];int pos=0;while(d_dict[pos]){int sl=n_strlen(d_dict+pos);
    d_sha256((const uint8_t*)(d_dict+pos),sl,p);check_return(p,f,fk);pos+=sl+1;}}

/* H28 seq */
__global__ void k28(uint64_t st,uint64_t cn,void*f,void*fk){
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=cn||*(int*)f)return;
    uint32_t v=(uint32_t)(st+tid);uint8_t m[4]={(uint8_t)(v>>24),(uint8_t)(v>>16),(uint8_t)(v>>8),(uint8_t)v};
    uint8_t p[32];d_sha256(m,4,p);check_return(p,f,fk);}

/* H03 ts+PID */
__global__ void k3(uint64_t cn,void*f,void*fk){
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=cn||*(int*)f)return;
    uint8_t m[10];for(int i=0;i<8;i++)m[7-i]=(uint8_t)(tid>>(i*8));
    m[8]=(uint8_t)((tid>>15)&0xFF);m[9]=(uint8_t)((tid>>7)&0xFF);
    uint8_t p[32];d_sha256(m,10,p);check_return(p,f,fk);}

/* H20 */
__global__ void k20(uint64_t cn,void*f,void*fk){
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=cn||*(int*)f)return;
    uint32_t v=(uint32_t)(tid&0xFFFFFFFF);uint8_t m[4]={(uint8_t)(v>>24),(uint8_t)(v>>16),(uint8_t)(v>>8),(uint8_t)v};
    uint8_t p[32];d_sha256(m,4,p);check_return(p,f,fk);}

/* H36 timestamp ms */
__global__ void k36(uint64_t st,uint64_t cn,void*f,void*fk){
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=cn||*(int*)f)return;
    uint64_t ms=st+tid;uint8_t m[8];for(int i=0;i<8;i++)m[7-i]=(uint8_t)(ms>>(i*8));
    uint8_t p[32];d_sha256(m,8,p);check_return(p,f,fk);}

/* H08 blocks */
__global__ void k8(int nb,void*f,void*fk){
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=(uint64_t)nb||*(int*)f)return;
    uint8_t p[32];d_sha256(d_block_hashes+tid*32,32,p);check_return(p,f,fk);}

/* H01 phrases × 7 */
__global__ void k1(void*f,void*fk){
    int n=d_num_phrases,tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n||*(int*)f)return;
    const char*ph=d_phrases+(uint64_t)tid*256;int pl=n_strlen(ph);
    uint8_t p[32];char b[512];for(int i=0;i<pl;i++)b[i]=ph[i];
    d_sha256((const uint8_t*)ph,pl,p);check_return(p,f,fk);
    b[pl]='1';b[pl+1]='2';b[pl+2]='3';b[pl+3]=0;d_sha256((const uint8_t*)b,pl+3,p);check_return(p,f,fk);
    b[pl]='1';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,p);check_return(p,f,fk);
    b[pl]='!';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,p);check_return(p,f,fk);
    b[pl]='@';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,p);check_return(p,f,fk);
    b[pl]='?';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,p);check_return(p,f,fk);
    b[pl]='.';b[pl+1]=0;d_sha256((const uint8_t*)b,pl+1,p);check_return(p,f,fk);}

/* H09 phrase+year */
__global__ void k9(void*f,void*fk){
    int n=d_num_phrases,tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n||*(int*)f)return;
    const char*ph=d_phrases+(uint64_t)tid*256;int pl=n_strlen(ph);
    uint8_t p[32];char b[512];for(int i=0;i<pl;i++)b[i]=ph[i];
    for(int yr=2009;yr<=2013;yr++){if(*(int*)f)return;
        int yy=yr;char y[8];int yl=0,t=yy;while(t){y[yl++]=t%10+'0';t/=10;}
        for(int i=0;i<yl/2;i++){char tp=y[i];y[i]=y[yl-1-i];y[yl-1-i]=tp;}
        for(int i=0;i<yl;i++)b[pl+i]=y[i];b[pl+yl]=0;
        d_sha256((const uint8_t*)b,pl+yl,p);check_return(p,f,fk);}}

/* H18 pairs */
__global__ void k18(void*f,void*fk){
    int n=d_num_phrases;if(n<2||*(int*)f)return;
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t max=(uint64_t)n*(n-1)/2;if(tid>=max||*(int*)f)return;
    int i=0;uint64_t t=tid;while((uint64_t)(i+1)*i/2<=t)i++;i--;
    int j=(int)(t-(uint64_t)i*(i+1)/2)+i+1;
    if(j>=n||i>=n)return;
    const char*a=d_phrases+(uint64_t)i*256;const char*b=d_phrases+(uint64_t)j*256;
    char buf[512];int al=n_strlen(a),bl=n_strlen(b);
    for(int x=0;x<al;x++)buf[x]=a[x];buf[al]=' ';
    for(int x=0;x<bl;x++)buf[al+1+x]=b[x];buf[al+1+bl]=0;
    uint8_t p[32];d_sha256((const uint8_t*)buf,al+1+bl,p);check_return(p,f,fk);}
