/* ================================================================
 *  KERNELS_CODE.CU — All GPU kernels (separate file for -dc)
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>

/* extern from main.cu */
extern __constant__ uint8_t d_targets[160];
extern __constant__ char d_dict[8192];
extern __constant__ int d_num_phrases;
extern __device__ char d_phrases[1048576];
extern __device__ uint8_t d_block_hashes[6400000];

#include "kernels.cuh"

/* Device strlen */
__device__ static int n_strlen(const char *s){int i=0;while(s[i])i++;return i;}

/* CHK: int* f, uint64_t* fk */
/* __device__ check function */
__device__ static void check_return(const uint8_t *pk, void *fp, void *fkp) {
    int *f = (int*)fp; uint64_t *fk = (uint64_t*)fkp;
    if(*f) return;
    uint64_t k[4];
    /* تحويل Big-endian 32-byte إلى uint64[4] Little-endian */
    /* pk[0]=MSB, pk[31]=LSB */
    /* bit0 من المفتاح = أقل بت في pk[31] */
    /* في uint64[4]: k[0] يحتوي على أقل 64 بت = pk[24..31] */
    for(int i=0;i<4;i++) k[3-i]=((uint64_t)pk[i*8]<<56)|((uint64_t)pk[i*8+1]<<48)|((uint64_t)pk[i*8+2]<<40)|((uint64_t)pk[i*8+3]<<32)|((uint64_t)pk[i*8+4]<<24)|((uint64_t)pk[i*8+5]<<16)|((uint64_t)pk[i*8+6]<<8)|pk[i*8+7];
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

/* H48: كل uint64 من 0 إلى 2^48 (SHA256(integer)) */
__global__ void k48(uint64_t st,uint64_t cn,void*f,void*fk){
    uint64_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=cn||*(int*)f)return;
    uint64_t val=st+tid; /* يولد 48-bit integer */
    uint8_t m[8];for(int i=0;i<8;i++)m[7-i]=(uint8_t)(val>>(i*8));
    uint8_t p[32];d_sha256(m,8,p);check_return(p,f,fk);}

/* H50: SHA256(address string) لكل target */
__global__ void k50(void*f,void*fk){
    if(threadIdx.x||blockIdx.x)return;
    /* العناوين كاملة */
    const char *addrs[]={
        "12rMpw5HnEvAw3nQqLmRBCQyuktfpa4eVw",
        "1HLvaTs3zR3oev9ya7Pzp3GB9Gqfg6XYJT",
        "1JA4MpuV8MMNYCDTFHdCQeXGyem7mqo4B4",
        "13GvAdkFeHFGVxTHzcA2rD2e5BD4cGkbBH",
        "1DTy9z4JvtqYsg44oagVpHqyQpF7ZLLs45",
        "1MVLP2kRPNqz8VJUy83LstUoMQzUjgq4Zg",
        "15QezNwA5ThiPf7wo89TTnfBwny93VQFTp",
        "198aMn6ZYAczwrE5NvNTUMyJ5qkfy4g3Hi",
        NULL
    };
    for(int i=0;addrs[i];i++){
        if(*(int*)f)return;
        int sl=n_strlen(addrs[i]);
        uint8_t p[32];d_sha256((const uint8_t*)addrs[i],sl,p);check_return(p,f,fk);
        /* address backwards */
        char rev[64];for(int j=0;j<sl;j++)rev[j]=addrs[i][sl-1-j];rev[sl]=0;
        d_sha256((const uint8_t*)rev,sl,p);check_return(p,f,fk);
    }
    /* RIPEMD160 hash itself as key */
    for(int t=0;t<8;t++){
        if(*(int*)f)return;
        uint8_t p[32];d_sha256(d_targets+t*20,20,p);check_return(p,f,fk);
    }
    /* SHA256(Hash160) */
    for(int t=0;t<8;t++){
        if(*(int*)f)return;
        uint8_t h[32];d_sha256(d_targets+t*20,20,h);
        d_sha256(h,32,h);check_return(h,f,fk);
    }
    /* Double SHA256 of address */
    for(int i=0;addrs[i];i++){
        if(*(int*)f)return;
        int sl=n_strlen(addrs[i]);
        uint8_t h[32];d_sha256((const uint8_t*)addrs[i],sl,h);
        d_sha256(h,32,h);check_return(h,f,fk);
        /* dSHA256 backwards */
        char rev[64];for(int j=0;j<sl;j++)rev[j]=addrs[i][sl-1-j];rev[sl]=0;
        d_sha256((const uint8_t*)rev,sl,h);d_sha256(h,32,h);check_return(h,f,fk);
    }
}

/* H51: reverse string لكل كلمة في d_dict */
__global__ void k51(void*f,void*fk){
    if(threadIdx.x||blockIdx.x)return;
    int pos=0;uint8_t p[32];
    while(d_dict[pos]){
        if(*(int*)f)return;
        int sl=n_strlen(d_dict+pos);
        /* كل مرة نص عكسي */
        char rev[256];for(int i=0;i<sl;i++)rev[i]=d_dict[pos+sl-1-i];rev[sl]=0;
        d_sha256((const uint8_t*)rev,sl,p);check_return(p,f,fk);
        pos+=sl+1;
    }
}

/* ================================================================
 *  SANITY KERNEL: يبحث عن مفتاح معروف privkey = 1
 *  الهدف: التحقق من صحة الخوارزميات (SHA256+RIPEMD160+ECC point mult)
 *  address الصحيح: 1EHNa6b34hmqoEgmcer8Kyo3Vs7NPre6MG
 * ================================================================ */
__global__ void k_sanity(void*f,void*fk){
    if(threadIdx.x||blockIdx.x)return;
    /* مفتاح خاص = 1 (0x000...001) */
    uint8_t pk[32];
    for(int i=0;i<32;i++) pk[i]=0;
    pk[31]=1;  /* Big-endian: آخر بايت = 1 */
    check_return(pk,f,fk);
}

/* DEBUG: تعمق في الفحص */
__global__ void k_debug_test(void){
    if(threadIdx.x||blockIdx.x)return;
    uint64_t k[4];
    /* ترجمة pk[31]=1 → k[0] = 1 (أقل 64 بت) */
    k[0]=1; k[1]=0; k[2]=0; k[3]=0;
    printf("[DBG] k[0]=%016llx k[1]=%016llx k[2]=%016llx k[3]=%016llx\n",
        (unsigned long long)k[0], (unsigned long long)k[1],
        (unsigned long long)k[2], (unsigned long long)k[3]);
    
    uint8_t h160[20];
    int ok = d_pk2h160(k, h160);
    printf("[DBG] d_pk2h160 ok=%d\n", ok);
    if(ok){
        printf("[DBG] h160="); for(int i=0;i<20;i++) printf("%02x",h160[i]); printf("\n");
        /* مقارنة مع compressed */
        uint8_t exp[]={0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
        int m=1; for(int i=0;i<20;i++) if(h160[i]!=exp[i]){m=0;break;}
        printf("[DBG] Compressed match: %s\n", m?"YES!":"NO");
        /* مقارنة مع uncompressed */
        uint8_t exp2[]={0x91,0xb2,0x4b,0xf9,0xf5,0x28,0x85,0x32,0x96,0x0a,0xc6,0x87,0xab,0xb0,0x35,0x12,0x7b,0x1d,0x28,0xa5};
        m=1; for(int i=0;i<20;i++) if(h160[i]!=exp2[i]){m=0;break;}
        printf("[DBG] Uncompressed match: %s\n", m?"YES!":"NO");
    } else {
        printf("[DBG] SCALAR REJECTED!\n");
    }
}

/* H52: كل target address كـ string بالكامل */
__global__ void k52(void*f,void*fk){
    if(threadIdx.x||blockIdx.x)return;
    const char*addrs[8]={
        "12rMpw5HnEvAw3nQqLmRBCQyuktfpa4eVw",
        "1HLvaTs3zR3oev9ya7Pzp3GB9Gqfg6XYJT",
        "1JA4MpuV8MMNYCDTFHdCQeXGyem7mqo4B4",
        "13GvAdkFeHFGVxTHzcA2rD2e5BD4cGkbBH",
        "1DTy9z4JvtqYsg44oagVpHqyQpF7ZLLs45",
        "1MVLP2kRPNqz8VJUy83LstUoMQzUjgq4Zg",
        "15QezNwA5ThiPf7wo89TTnfBwny93VQFTp",
        "198aMn6ZYAczwrE5NvNTUMyJ5qkfy4g3Hi"
    };
    uint8_t p[32];
    for(int i=0;i<8;i++){
        if(*(int*)f)return;
        int sl=n_strlen(addrs[i]);
        d_sha256((const uint8_t*)addrs[i],sl,p);check_return(p,f,fk);
        
        /* Double SHA256 of address */
        uint8_t h[32];
        d_sha256((const uint8_t*)addrs[i],sl,h);
        d_sha256(h,32,h);check_return(h,f,fk);
        
        /* SHA256 of hash160 */
        d_sha256(d_targets+i*20,20,p);check_return(p,f,fk);
        
        /* Double SHA256 hash160 */
        d_sha256(d_targets+i*20,20,h);
        d_sha256(h,32,h);check_return(h,f,fk);
        
        /* reverse address */
        char rev[64];
        for(int j=0;j<sl;j++) rev[j]=addrs[i][sl-1-j]; rev[sl]=0;
        d_sha256((const uint8_t*)rev,sl,p);check_return(p,f,fk);
    }
}
