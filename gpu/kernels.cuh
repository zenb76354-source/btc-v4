/* ================================================================
 *  KERNELS.CUH — Full ECC on GPU
 *  كل الـ tables داخل الـ functions مش static const (عشان constant memory)
 * ================================================================ */

#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

/* ================================================================
 *  SHA256
 * ================================================================ */

/* SHA-256 إعادة كتابة كاملة بأسماء متغيرات واضحة */
__device__ static void d_sha256(const uint8_t *msg, uint32_t len, uint8_t out[32]) {
    const uint32_t K256[64] = {
        0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
        0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
        0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
        0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
        0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
        0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
        0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
        0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
    };
    uint32_t H[8] = {0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,
                     0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19};
    uint64_t bitlen = (uint64_t)len * 8;
    
    for(uint32_t poff = 0; poff < len + 9; poff += 64) {
        uint8_t blk[64];
        for(int i=0;i<64;i++) blk[i]=0;
        for(uint32_t i=0; i<64 && (poff+i)<len; i++)
            blk[i] = msg[poff+i];
        if(poff + 64 > len) {
            blk[len - poff] = 0x80;
            if(poff + 56 < len) {
                /* bits في next block */
            } else {
                for(int i=0;i<8;i++) blk[63-i] = (uint8_t)(bitlen >> (i*8));
            }
        }
        uint32_t W[64];
        for(int i=0;i<16;i++)
            W[i] = ((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)|
                   ((uint32_t)blk[i*4+2]<<8)|blk[i*4+3];
        for(int i=16;i<64;i++) {
            uint32_t ss0 = ((W[i-15]>>7)|(W[i-15]<<25))^((W[i-15]>>18)|(W[i-15]<<14))^(W[i-15]>>3);
            uint32_t ss1 = ((W[i-2]>>17)|(W[i-2]<<15))^((W[i-2]>>19)|(W[i-2]<<13))^(W[i-2]>>10);
            W[i] = W[i-16] + ss0 + W[i-7] + ss1;
        }
        uint32_t A=H[0],B=H[1],C=H[2],D=H[3],E=H[4],F=H[5],G=H[6],Hh=H[7];
        for(int i=0;i<64;i++){
            uint32_t S1 = ((E>>6)|(E<<26))^((E>>11)|(E<<21))^((E>>25)|(E<<7));
            uint32_t ch = (E&F)^((~E)&G);
            uint32_t t1 = Hh + S1 + ch + K256[i] + W[i];
            uint32_t S0 = ((A>>2)|(A<<30))^((A>>13)|(A<<19))^((A>>22)|(A<<10));
            uint32_t maj = (A&B)^(A&C)^(B&C);
            uint32_t t2 = S0 + maj;
            Hh=G; G=F; F=E; E=D+t1; D=C; C=B; B=A; A=t1+t2;
        }
        H[0]+=A; H[1]+=B; H[2]+=C; H[3]+=D; H[4]+=E; H[5]+=F; H[6]+=G; H[7]+=Hh;
        if(poff + 64 > len) break;  /* آخر block */
    }
    out[0]=(uint8_t)(H[0]>>24);out[1]=(uint8_t)(H[0]>>16);out[2]=(uint8_t)(H[0]>>8);out[3]=(uint8_t)H[0];
    out[4]=(uint8_t)(H[1]>>24);out[5]=(uint8_t)(H[1]>>16);out[6]=(uint8_t)(H[1]>>8);out[7]=(uint8_t)H[1];
    out[8]=(uint8_t)(H[2]>>24);out[9]=(uint8_t)(H[2]>>16);out[10]=(uint8_t)(H[2]>>8);out[11]=(uint8_t)H[2];
    out[12]=(uint8_t)(H[3]>>24);out[13]=(uint8_t)(H[3]>>16);out[14]=(uint8_t)(H[3]>>8);out[15]=(uint8_t)H[3];
    out[16]=(uint8_t)(H[4]>>24);out[17]=(uint8_t)(H[4]>>16);out[18]=(uint8_t)(H[4]>>8);out[19]=(uint8_t)H[4];
    out[20]=(uint8_t)(H[5]>>24);out[21]=(uint8_t)(H[5]>>16);out[22]=(uint8_t)(H[5]>>8);out[23]=(uint8_t)H[5];
    out[24]=(uint8_t)(H[6]>>24);out[25]=(uint8_t)(H[6]>>16);out[26]=(uint8_t)(H[6]>>8);out[27]=(uint8_t)H[6];
    out[28]=(uint8_t)(H[7]>>24);out[29]=(uint8_t)(H[7]>>16);out[30]=(uint8_t)(H[7]>>8);out[31]=(uint8_t)H[7];
}

/* ================================================================
 *  RIPEMD-160 (local tables only — no static const)
 * ================================================================ */

/* RIPEMD-160 — كاملة من الصفر */
__device__ static void d_ripemd160(const uint8_t *msg, uint32_t len, uint8_t out[20]) {
    /* IV */
    uint32_t H0=0x67452301, H1=0xEFCDAB89, H2=0x98BADCFE, H3=0x10325476, H4=0xC3D2E1F0;
    /* K constants for right (parallel) line */
    const uint32_t KR[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
    /* K constants for left (original) line */
    const uint32_t KL[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
    /* RR: message order for right line */
    const uint32_t RR[5][16] = {
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
        { 7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8},
        { 3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12},
        { 1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2},
        { 4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13}};
    /* RL: message order for left line */
    const uint32_t RL[5][16] = {
        { 5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12},
        { 6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2},
        {15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13},
        { 8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14},
        {12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11}};
    /* SR: shift amounts for right line */
    const uint32_t SR[5][16] = {
        {11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8},
        {12,13,11,15, 6, 9, 9, 7,12,15,11,13, 7,15, 7,12},
        {13,15,14,11, 7, 7, 6,12,13,13,11,15, 9,11, 9, 7},
        { 9, 9,13,15, 6,14, 8,11,13,15,10,14,10,13, 9,13},
        {15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,14, 5,15,11}};
    /* SL: shift amounts for left line */
    const uint32_t SL[5][16] = {
        { 8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6},
        { 9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11},
        { 9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5},
        {15, 5, 8,11,14,14, 6,14, 6, 9,12, 9, 5,15,11,12},
        { 8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11}};
    /* functions */
    #define F1(x,y,z) (x^y^z)
    #define F2(x,y,z) ((x&y)|(~x&z))
    #define F3(x,y,z) ((x|~y)^z)
    #define F4(x,y,z) ((x&z)|(y&~z))
    #define F5(x,y,z) (x^(y|~z))
    #define RL32(x,n) (((x)<<(n))|((x)>>(32-(n))))
    
    /* padding */
    uint64_t bitlen = (uint64_t)len * 8;
    uint32_t padded = len + 9;
    uint32_t blkcnt = (padded + 63) / 64;
    for(uint32_t b=0; b<blkcnt; b++) {
        uint8_t blk[64];
        for(int i=0;i<64;i++) blk[i]=0;
        uint32_t poff = b*64;
        for(uint32_t i=0; i<64 && (poff+i)<len; i++)
            blk[i] = msg[poff+i];
        if(poff+64 > len) {
            blk[len-poff] = 0x80;
            if(poff+56 <= len) {
                for(int i=0;i<8;i++) blk[56+i] = (uint8_t)(bitlen >> (i*8));
            }
        }
        
        /* X: 16 little-endian words */
        uint32_t X[16];
        for(int i=0;i<16;i++)
            X[i] = blk[i*4] | ((uint32_t)blk[i*4+1]<<8) |
                   ((uint32_t)blk[i*4+2]<<16) | ((uint32_t)blk[i*4+3]<<24);
        
        /* state */
        uint32_t r0=H0, r1=H1, r2=H2, r3=H3, r4=H4;
        uint32_t l0=H0, l1=H1, l2=H2, l3=H3, l4=H4;
        
        for(int j=0; j<80; j++) {
            int rd = j/16;
            int pos = j%16;
            /* right line */
            uint32_t f = (rd==0?F1:rd==1?F2:rd==2?F3:rd==3?F4:F5)(r1,r2,r3);
            uint32_t t = RL32(r0 + f + X[RR[rd][pos]] + KR[rd], SR[rd][pos]) + r4;
            r0=r4; r4=r3; r3=RL32(r2,10); r2=r1; r1=t;
            /* left line */
            f = (rd==0?F5:rd==1?F4:rd==2?F3:rd==3?F2:F1)(l1,l2,l3);
            t = RL32(l0 + f + X[RL[rd][pos]] + KL[rd], SL[rd][pos]) + l4;
            l0=l4; l4=l3; l3=RL32(l2,10); l2=l1; l1=t;
        }
        
        /* combine */
        uint32_t t = H1 + r2 + l3;
        H1 = H2 + r3 + l4;
        H2 = H3 + r4 + l0;
        H3 = H4 + r0 + l1;
        H4 = H0 + r1 + l2;
        H0 = t;
    }
    
    /* output little-endian */
    out[0]=(uint8_t)H0; out[1]=(uint8_t)(H0>>8); out[2]=(uint8_t)(H0>>16); out[3]=(uint8_t)(H0>>24);
    out[4]=(uint8_t)H1; out[5]=(uint8_t)(H1>>8); out[6]=(uint8_t)(H1>>16); out[7]=(uint8_t)(H1>>24);
    out[8]=(uint8_t)H2; out[9]=(uint8_t)(H2>>8); out[10]=(uint8_t)(H2>>16); out[11]=(uint8_t)(H2>>24);
    out[12]=(uint8_t)H3; out[13]=(uint8_t)(H3>>8); out[14]=(uint8_t)(H3>>16); out[15]=(uint8_t)(H3>>24);
    out[16]=(uint8_t)H4; out[17]=(uint8_t)(H4>>8); out[18]=(uint8_t)(H4>>16); out[19]=(uint8_t)(H4>>24);
    
    #undef F1
    #undef F2
    #undef F3
    #undef F4
    #undef F5
    #undef RL32
}

/* ================================================================
 *  FIELD ELEMENT secp256k1
 *  p = 2²⁵⁶ - 2³² - 977
 *  4 × uint64_t little-endian
 * ================================================================ */

#define FE_L 4
typedef struct { uint64_t d[FE_L]; } d_fe;

/* P, N, Gx, Gy as macros to avoid __device__ const storage */
#define D_P0  0xFFFFFFFEFFFFFC2FULL
#define D_P1  0xFFFFFFFFFFFFFFFFULL
#define D_P2  0xFFFFFFFFFFFFFFFFULL
#define D_P3  0xFFFFFFFFFFFFFFFFULL

#define D_N0  0xBFD25E8CD0364141ULL
#define D_N1  0xBAAEDCE6AF48A03BULL
#define D_N2  0xFFFFFFFFFFFFFFFEULL
#define D_N3  0xFFFFFFFFFFFFFFFFULL

__device__ static void fe_zero(d_fe *r) { for(int i=0;i<FE_L;i++) r->d[i]=0; }
__device__ static void fe_one(d_fe *r) { fe_zero(r); r->d[0]=1; }
__device__ static void fe_copy(d_fe *r, const d_fe *a) { for(int i=0;i<FE_L;i++) r->d[i]=a->d[i]; }
__device__ static int fe_is_zero(const d_fe *a) { return (a->d[0]|a->d[1]|a->d[2]|a->d[3])==0; }

__device__ static int fe_ge_p(const d_fe *a) {
    /* compare >= P */
    if(a->d[3]<D_P3) return 0;
    if(a->d[3]>D_P3) return 1;
    if(a->d[2]<D_P2) return 0;
    if(a->d[2]>D_P2) return 1;
    if(a->d[1]<D_P1) return 0;
    if(a->d[1]>D_P1) return 1;
    return a->d[0]>=D_P0?1:0;
}

__device__ static void fe_add(d_fe *r, const d_fe *a, const d_fe *b) {
    uint64_t c=0;
    for(int i=0;i<FE_L;i++){
        uint64_t s=a->d[i]+b->d[i]+c;
        r->d[i]=s; c=(s<a->d[i])?1:((s==a->d[i])?c:0);
    }
    if(c||fe_ge_p(r)){
        uint64_t b2=0;
        for(int i=0;i<FE_L;i++){
            uint64_t sub=(i==0)?D_P0:0xFFFFFFFFFFFFFFFFULL;
            uint64_t v=r->d[i]-sub-b2;
            r->d[i]=v; b2=(v>r->d[i])?1:0;
        }
    }
}

__device__ static void fe_sub(d_fe *r, const d_fe *a, const d_fe *b) {
    uint64_t borrow=0;
    for(int i=0;i<FE_L;i++){
        uint64_t v=a->d[i]-b->d[i]-borrow;
        r->d[i]=v; borrow=(v>a->d[i])?1:0;
    }
    if(borrow){
        uint64_t carry=0;
        for(int i=0;i<FE_L;i++){
            uint64_t add=(i==0)?D_P0:0xFFFFFFFFFFFFFFFFULL;
            uint64_t v=r->d[i]+add+carry;
            r->d[i]=v; carry=(v<add||(v==add&&carry))?1:0;
        }
    }
}

__device__ static void fe_mul(d_fe *r, const d_fe *a, const d_fe *b) {
    uint64_t t[8]={0};
    for(int i=0;i<FE_L;i++){
        uint64_t carry=0;
        for(int j=0;j<FE_L;j++){
            __uint128_t prod=(__uint128_t)a->d[i]*b->d[j]+t[i+j]+carry;
            t[i+j]=(uint64_t)prod;
            carry=(uint64_t)(prod>>64);
        }
        t[i+FE_L]+=carry;
    }
    /* Lower 4 limbs = product mod 2^256 */
    d_fe r2;
    r2.d[0]=t[0]; r2.d[1]=t[1]; r2.d[2]=t[2]; r2.d[3]=t[3];
    if(fe_ge_p(&r2)){
        uint64_t bor=0;
        for(int i=0;i<FE_L;i++){
            uint64_t sub=(i==0)?D_P0:0xFFFFFFFFFFFFFFFFULL;
            uint64_t v=r2.d[i]-sub-bor;
            r2.d[i]=v; bor=(v>r2.d[i])?1:0;
        }
    }
    fe_copy(r,&r2);
}

__device__ static void fe_sqr(d_fe *r, const d_fe *a) {
    /* Same as fe_mul but with a=a */
    uint64_t t[8]={0};
    for(int i=0;i<FE_L;i++){
        uint64_t carry=0;
        for(int j=0;j<FE_L;j++){
            __uint128_t prod=(__uint128_t)a->d[i]*a->d[j]+t[i+j]+carry;
            t[i+j]=(uint64_t)prod;
            carry=(uint64_t)(prod>>64);
        }
        t[i+FE_L]+=carry;
    }
    d_fe r2;
    r2.d[0]=t[0]; r2.d[1]=t[1]; r2.d[2]=t[2]; r2.d[3]=t[3];
    if(fe_ge_p(&r2)){
        uint64_t bor=0;
        for(int i=0;i<FE_L;i++){
            uint64_t sub=(i==0)?D_P0:0xFFFFFFFFFFFFFFFFULL;
            uint64_t v=r2.d[i]-sub-bor;
            r2.d[i]=v; bor=(v>r2.d[i])?1:0;
        }
    }
    fe_copy(r,&r2);
}

__device__ static void fe_inv(d_fe *r, const d_fe *a) {
    /* a^(p-2) mod p — Fermat */
    d_fe base,res;
    fe_copy(&base,a);
    fe_one(&res);
    /* exponent = p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D */
    const uint64_t exp[4]={0xFFFFFFFEFFFFFC2DULL,0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
    for(int bit=0;bit<256;bit++){
        if(exp[bit/64]&(1ULL<<(bit%64))) fe_mul(&res,&res,&base);
        fe_sqr(&base,&base);
    }
    fe_copy(r,&res);
}

/* ================================================================
 *  JACOBIAN POINT
 * ================================================================ */

typedef struct { d_fe x,y,z; } d_jac;

__device__ static void jac_inf(d_jac *p) { fe_zero(&p->x); fe_zero(&p->y); fe_zero(&p->z); }
__device__ static int jac_is_inf(const d_jac *p) { return fe_is_zero(&p->z); }
__device__ static void jac_g(d_jac *p) {
    p->x.d[0]=0x79BE667EF9DCBBACULL; p->x.d[1]=0x55A06295CE870B07ULL;
    p->x.d[2]=0x029BFCDB2DCE28D9ULL; p->x.d[3]=0x59F2815B16F81798ULL;
    p->y.d[0]=0x483ADA7726A3C465ULL; p->y.d[1]=0x5DA4FBFC0E1108A8ULL;
    p->y.d[2]=0xFD17B448A6855419ULL; p->y.d[3]=0x9C47D08FFB10D4B8ULL;
    fe_one(&p->z);
}
__device__ static void jac_cp(d_jac *r, const d_jac *a) { fe_copy(&r->x,&a->x); fe_copy(&r->y,&a->y); fe_copy(&r->z,&a->z); }

__device__ static void jac_double(d_jac *r, const d_jac *p) {
    if(jac_is_inf(p)){jac_inf(r);return;}
    d_fe m,s,t,x3,y3,z3;
    fe_sqr(&t,&p->x);
    fe_add(&m,&t,&t); fe_add(&m,&m,&t);
    fe_sqr(&t,&p->y);
    fe_mul(&s,&p->x,&t);
    fe_add(&s,&s,&s); fe_add(&s,&s,&s);
    fe_sqr(&x3,&m);
    fe_sub(&x3,&x3,&s); fe_sub(&x3,&x3,&s);
    fe_sub(&t,&s,&x3);
    fe_mul(&y3,&m,&t);
    fe_sqr(&t,&t);
    fe_add(&t,&t,&t); fe_add(&t,&t,&t); fe_add(&t,&t,&t);
    fe_sub(&y3,&y3,&t);
    fe_mul(&z3,&p->y,&p->z); fe_add(&z3,&z3,&z3);
    fe_copy(&r->x,&x3); fe_copy(&r->y,&y3); fe_copy(&r->z,&z3);
}

__device__ static void jac_add_mixed(d_jac *r, const d_jac *a, const d_jac *b) {
    if(jac_is_inf(a)){jac_cp(r,b);return;}
    if(jac_is_inf(b)){jac_cp(r,a);return;}
    d_fe z1z1,u1,s1,u2,s2,h,hh,i,rr,vv;
    fe_sqr(&z1z1,&a->z);
    fe_copy(&u1,&a->x);
    fe_copy(&s1,&a->y);
    fe_mul(&u2,&b->x,&z1z1);
    fe_mul(&s2,&b->y,&a->z); fe_mul(&s2,&s2,&z1z1);
    fe_sub(&h,&u2,&u1);
    fe_sqr(&hh,&h);
    fe_mul(&i,&hh,&h);
    fe_sub(&rr,&s2,&s1); fe_add(&rr,&rr,&rr);
    fe_sqr(&r->x,&rr);
    fe_sub(&r->x,&r->x,&i);
    fe_add(&vv,&z1z1,&z1z1);
    fe_sub(&r->x,&r->x,&vv);
    fe_sub(&r->y,&vv,&r->x);
    fe_mul(&r->y,&r->y,&rr);
    fe_mul(&s1,&s1,&i);
    fe_sub(&r->y,&r->y,&s1);
    fe_mul(&r->z,&b->z,&a->z);
    fe_mul(&r->z,&r->z,&h);
}

/* ================================================================
 *  POINT MULT: Q = k * G
 * ================================================================ */

__device__ static void jac_mul_g(d_jac *r, const uint64_t k[FE_L]) {
    d_jac Q,T;
    jac_inf(&Q);
    jac_g(&T);
    for(int bit=0;bit<256;bit++){
        if(k[bit/64]&(1ULL<<(bit%64))){
            d_jac tmp; jac_cp(&tmp,&Q);
            jac_add_mixed(&Q,&tmp,&T);
        }
        d_jac T2; jac_cp(&T2,&T);
        jac_double(&T,&T2);
    }
    jac_cp(r,&Q);
}

/* ================================================================
 *  d_pk2h160: private key → hash160 (Full GPU)
 * ================================================================ */

__device__ static int d_pk2h160(const uint64_t *scalar, uint8_t h160[20]) {
    for(int i=3;i>=0;i--){
        if(scalar[i]> ((i==3)?D_N3:(i==2)?D_N2:(i==1)?D_N1:D_N0)) return 0;
        if(scalar[i]< ((i==3)?D_N3:(i==2)?D_N2:(i==1)?D_N1:D_N0)) break;
    }
    if((scalar[0]|scalar[1]|scalar[2]|scalar[3])==0) return 0;

    d_jac Q;
    jac_mul_g(&Q, scalar);

    d_fe zi,zi2,zi3,xa,ya;
    fe_inv(&zi,&Q.z);
    fe_sqr(&zi2,&zi);
    fe_mul(&zi3,&zi2,&zi);
    fe_mul(&xa,&Q.x,&zi2);
    fe_mul(&ya,&Q.y,&zi3);

    uint8_t pub[33];
    pub[0] = (ya.d[0] & 1) ? 0x03 : 0x02;
    for(int i=0;i<4;i++){
        pub[1+(3-i)*8+0] = (uint8_t)(xa.d[i]>>56);
        pub[1+(3-i)*8+1] = (uint8_t)(xa.d[i]>>48);
        pub[1+(3-i)*8+2] = (uint8_t)(xa.d[i]>>40);
        pub[1+(3-i)*8+3] = (uint8_t)(xa.d[i]>>32);
        pub[1+(3-i)*8+4] = (uint8_t)(xa.d[i]>>24);
        pub[1+(3-i)*8+5] = (uint8_t)(xa.d[i]>>16);
        pub[1+(3-i)*8+6] = (uint8_t)(xa.d[i]>>8);
        pub[1+(3-i)*8+7] = (uint8_t)(xa.d[i]);
    }

    uint8_t sha[32];
    d_sha256(pub, 33, sha);
    d_ripemd160(sha, 32, h160);
    return 1;
}

#endif /* KERNELS_CUH */
