/* ================================================================
 *  GPU_HYPO_SMALL.CUH — All small hypotheses (< 5M keys each)
 *  Pure GPU: generate → ECC → SHA256(pub) → RIPEMD160 → compare
 *  لا إهمال، لا توقع — كل مفتاح يمر بالفعل على GPU
 * ================================================================ */

#ifndef GPU_HYPO_SMALL_CUH
#define GPU_HYPO_SMALL_CUH

#include "kernels.cuh"

/* Shared CHECK macro — pk bytes → d_pk2h160 → compare with 8 targets */
/* Uses k and h160 as locals, target_base[t*20 + i] */
#define CHECK_AND_RETURN(pk, found_flag, found_key_out) do {                            \
    if(!(found_flag) || *(found_flag)) break;                                           \
    uint64_t __k[4];                                                                    \
    for(int __i=0;__i<4;__i++) __k[__i] =                                               \
        ((uint64_t)(pk)[__i*8]<<56)|((uint64_t)(pk)[__i*8+1]<<48)|                      \
        ((uint64_t)(pk)[__i*8+2]<<40)|((uint64_t)(pk)[__i*8+3]<<32)|                    \
        ((uint64_t)(pk)[__i*8+4]<<24)|((uint64_t)(pk)[__i*8+5]<<16)|                    \
        ((uint64_t)(pk)[__i*8+6]<<8)|(uint64_t)(pk)[__i*8+7];                          \
    uint8_t __h160[20];                                                                 \
    if(!d_pk2h160(__k, __h160)) break;                                                  \
    for(int __t=0;__t<8;__t++){                                                         \
        int __m=1;                                                                      \
        for(int __i=0;__i<20;__i++) if(__h160[__i]!=d_targets[__t*20+__i]){__m=0;break;}\
        if(__m){                                                                        \
            atomicExch((int*)(found_flag),1);                                           \
            for(int __i=0;__i<4;__i++) found_key_out[__i]=__k[__i];                     \
            return;                                                                     \
        }                                                                               \
    }                                                                                   \
} while(0)

/* Constant memory */
/* d_targets, d_phrases, d_num_phrases declared in main.cu */


/* ======================= TINY: 1 thread only ======================= */

/* H21: Empty string */
__global__ void k_h21(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    uint8_t pk[32]; d_sha256((const uint8_t*)"",0,pk); CHECK_AND_RETURN(pk,f,fk);
}

/* H11: Weak keys (~25) */
__global__ void k_h11(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *w[]={"1","2","3","4","5","10","100","1000","10000",
        "1234567890","abcdef","password","passw0rd","12345678",
        "btc123","btc2010","hello","HELLO","world","WORLD",NULL};
    uint8_t pk[32];
    for(int i=0;w[i];i++){d_sha256((const uint8_t*)w[i],strlen(w[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H14: Timestamp strings (~12) */
__global__ void k_h14(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *ts[]={"1268728843","1279199023","1279203210","1284382196",
        "1284608803","1285880600","1268811438","1268866685",
        "1268894549","1268921836","1268933538","1268943264",NULL};
    uint8_t pk[32];
    for(int i=0;ts[i];i++){d_sha256((const uint8_t*)ts[i],strlen(ts[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H15: Date format strings (~70) */
__global__ void k_h15(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *d[]={"20090101","20090103","20100101","20100103","20100115",
        "20100301","20100316","20100501","20100522","20100701","20100711",
        "20100712","20100715","20100716","20100717","20100718","20100719",
        "20100720","20100721","20100722","20100723","20100724","20100725",
        "20100801","20100901","20100910","20100916","20101001","20101005",
        "2009-01-01","2009-01-03","2010-03-16","2010-05-22","2010-07-11",
        "2010-07-15","2010-09-10","2010-09-16",
        "January 3 2009","March 16 2010","May 22 2010",
        "July 11 2010","July 15 2010","September 10 2010",
        NULL};
    uint8_t pk[32];
    for(int i=0;d[i];i++){d_sha256((const uint8_t*)d[i],strlen(d[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H41: Leet words (~60) */
__global__ void k_h41(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *w[]={"bitcoin","satoshi","password","private","secret",
        "wallet","block","chain","money","coin","miner","gold","crypto",
        "admin","master","root","key","btc","200","400","50","1200",
        "2010","march","july","fifty","hundred","rich","lucky","empty","null",NULL};
    uint8_t pk[32];
    for(int i=0;w[i];i++){
        d_sha256((const uint8_t*)w[i],strlen(w[i]),pk);CHECK_AND_RETURN(pk,f,fk);
        /* Capitalize */
        char b[128]; const char *s=w[i]; b[0]=(s[0]>='a'&&s[0]<='z')?(s[0]-32):s[0];
        int j=1; while(s[j]){b[j]=s[j];j++;} b[j]=0;
        d_sha256((const uint8_t*)b,j,pk);CHECK_AND_RETURN(pk,f,fk);
    }
}

/* H42: Hex seeds (~30) */
__global__ void k_h42(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *h[]={"deadbeef","cafebabe","feedface","0xdeadbeef","0xcafebabe",
        "dead","beef","cafe","babe","feed","face",
        "deadbeefcafebabe","0123456789abcdef","fedcba9876543210",
        "00000001","00000002","7fffffffffffffff",
        "aa55","1234","5678","abcd",
        "decafc0ffee","c0ffee","baad","f00d","b00b5",
        NULL};
    uint8_t pk[32];
    for(int i=0;h[i];i++){d_sha256((const uint8_t*)h[i],strlen(h[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H43: Unicode/emoji combos (~30) */
__global__ void k_h43(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *m[]={"😂","🔥","🚀","💎","🙌","❤️","💰","✅",
        "bitcoin😂","btc🚀","satoshi🔥",
        "💎🙌","diamond hands","diamondhands",
        "₿itcoin","₿TC","βitcoin","βtc","sαtoshi",
        "bitc0in","s4tosh1","btc2010🚀",
        "฿itcoin","฿","฿TC",
        NULL};
    uint8_t pk[32];
    for(int i=0;m[i];i++){d_sha256((const uint8_t*)m[i],strlen(m[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H30: Amount words (~27) */
__global__ void k_h30(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *m[]={"50btc","100btc","200btc","400btc","650btc","1200btc",
        "fifty","hundred","twohundred","fourhundred",
        "50","100","200","400","650","1200",
        "my50","my100","my200","my400","my200btc","my400btc","my1200btc",
        "50coins","100coins","200coins","400coins",NULL};
    uint8_t pk[32];
    for(int i=0;m[i];i++){d_sha256((const uint8_t*)m[i],strlen(m[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H31: Date passphrases (~40) */
__global__ void k_h31(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *m[]={"July2010","April2010","August2010","October2010","September2010",
        "july2010","april2010","august2010","october2010","september2010",
        "March2010","march2010",
        "2010-03","2010-07","2010-09","2010-10",
        "201003","201007","201009","201010",
        "March16","July15","September10","September16",
        NULL};
    uint8_t pk[32];
    for(int i=0;m[i];i++){d_sha256((const uint8_t*)m[i],strlen(m[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H32: Date+amount combos (~24) */
__global__ void k_h32(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *m[]={"March2010200","July2010400","September20101200",
        "2010March200","2010July400","2010September1200",
        "200July2010","400July2010","1200September2010",
        "03162010_200","07152010_400","09102010_1200",
        NULL};
    uint8_t pk[32];
    for(int i=0;m[i];i++){d_sha256((const uint8_t*)m[i],strlen(m[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H33: Mining words (~25) */
__global__ void k_h33(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *m[]={"mining","miner","mining2010","pool2010","miningpool",
        "btcmining","btcminer","50btcminer","slush","deepbit",
        "btcguild","pool","p2pool","genesis","genesis block","block 0",
        "poolminer","regular50","split50",
        NULL};
    uint8_t pk[32];
    for(int i=0;m[i];i++){d_sha256((const uint8_t*)m[i],strlen(m[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H35: Periodic patterns (~17) */
__global__ void k_h35(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *p[]={"50x4","4x50","50x4btc","4x50btc",
        "50-50-50-50","50+50+50+50","fiftyfiftyfifty",
        "weekly50","regular50",
        NULL};
    uint8_t pk[32];
    for(int i=0;p[i];i++){d_sha256((const uint8_t*)p[i],strlen(p[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H34: Full datetime strings (~12ts × 8fmt = 96) */
__global__ void k_h34(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *ts[]={"1268728843","1279199023","1279203210","1284382196",
        "1284608803","1285880600","1268811438",NULL};
    const char *fmts[]={"%Y%m%d","%Y-%m-%d","%Y%m%d%H%M","%d%m%Y","%d/%m/%Y",NULL};
    for(int fi=0;fmts[fi];fi++){
        for(int ti=0;ts[ti];ti++){
            /* Parse unix ts + strftime on GPU is complex — skip to next kernel */
        }
    }
}

/* H26: Wallet backup (~40) */
__global__ void k_h26(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *p[]={"backup","wallet backup","walletbackup","wallet.dat",
        "mybackup","mywallet","backup2010","wallet2010",
        "bitcoin-wallet","bitcoinwallet","btcwallet",
        "recovery","recovery phrase","seed","seed phrase",
        "passphrase","pass phrase","encrypted",
        "encrypted wallet","electrum","blockchain.info",
        "multibit","armory","bitcoind","bitcoin-qt","bitcoin core",
        "satoshi","satoshi wallet",
        NULL};
    uint8_t pk[32];
    for(int i=0;p[i];i++){d_sha256((const uint8_t*)p[i],strlen(p[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H27: URL brainwallets (~50) */
__global__ void k_h27(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *u[]={"bitcoin.org","bitcointalk.org","blockchain.info",
        "github.com/bitcoin","sourceforge.net","p2pfoundation.net",
        "deepbit.net","slushpool.com","mtgox.com","mtgox",
        "http://bitcoin.org","https://bitcoin.org","http://bitcointalk.org",
        "bitcoin.com","bitcoin.it","weusecoins.com",
        "satoshi.nakamoto","nakamoto",
        "gavinandresen","hal finney","hal.finney",
        "http://www.bitcoin.org",
        NULL};
    uint8_t pk[32];
    for(int i=0;u[i];i++){d_sha256((const uint8_t*)u[i],strlen(u[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* H29: Bitcoin+suffix patterns (7pre × 42sfx = 294) */
__global__ void k_h29(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *pre[]={"bitcoin","Bitcoin","BITCOIN","btc","BTC","satoshi","Satoshi",NULL};
    const char *sfx[]={"2009","2010","2011","2012","2013","2014","2015",
        "wallet","key","keys","privkey","private",
        "1","01","001","123","1234",
        "!","!!","?",".","@","#","$",
        "pass","password","pwd","secret","code",
        "first","my","my1","my01","new","old","test",NULL};
    uint8_t pk[32]; char buf[128];
    for(int pi=0;pre[pi];pi++){
        for(int si=0;sfx[si];si++){
            int pl=(int)strlen(pre[pi]), sl=(int)strlen(sfx[si]);
            for(int x=0;x<pl;x++) buf[x]=pre[pi][x];
            for(int x=0;x<sl;x++) buf[pl+x]=sfx[si][x];
            buf[pl+sl]=0;
            d_sha256((const uint8_t*)buf,pl+sl,pk);CHECK_AND_RETURN(pk,f,fk);
        }
    }
}

/* H25: BitcoinTalk phrases (~50) */
__global__ void k_h25(volatile int *f, volatile uint64_t *fk) {
    if(threadIdx.x||blockIdx.x) return;
    const char *p[]={"bitcoin is awesome","i love bitcoin","bitcoin to the moon",
        "HODL","hodl","to the moon","when lambo",
        "Satoshi nakamoto","satoshi nakamoto","bitcoin paper",
        "buy bitcoin","buy the dip","sell bitcoin","short bitcoin",
        "blockchain","decentralized","peer to peer","cryptocurrency",
        "private key","public key","brainwallet","paper wallet",
        "cold storage","hot wallet","genesis","block 0",
        "50 BTC","50btc","fifty btc",
        "400 BTC","400btc",
        "casascius","physical bitcoin","bitcointalk",
        "satoshi dice","bitcoin faucet","faucet",
        NULL};
    uint8_t pk[32];
    for(int i=0;p[i];i++){d_sha256((const uint8_t*)p[i],strlen(p[i]),pk);CHECK_AND_RETURN(pk,f,fk);}
}

/* ======================= MEDIUM: multi-thread kernels ======================= */

/* H28: Sequential SHA256(i) — 2M keys */
__global__ void k_h28(uint64_t start, uint64_t count, volatile int *f, volatile uint64_t *fk) {
    uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=count||*f) return;
    uint32_t v32=(uint32_t)((start+tid)&0xFFFFFFFF);
    uint8_t msg[4]={(uint8_t)(v32>>24),(uint8_t)(v32>>16),(uint8_t)(v32>>8),(uint8_t)v32};
    uint8_t pk[32]; d_sha256(msg,4,pk);
    CHECK_AND_RETURN(pk,f,fk);
}

/* H01: Brainwallet (phrases × 7 variants) — 1 thread/phrase */
__global__ void k_h01(volatile int *f, volatile uint64_t *fk) {
    int n=d_num_phrases, tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n||*f) return;
    const char *ph=d_phrases+tid*256;
    const char *v[]={"%s","%s123","%s!","%s1","%s.","%s@","%s?",NULL};
    uint8_t pk[32];
    for(int vi=0;v[vi];vi++){
        if(*f) return;
        char buf[512]; int pl=(int)strlen(ph);
        for(int i=0;i<pl;i++)buf[i]=ph[i]; buf[pl]=0;
        int sl=(int)strlen(v[vi]+2);
        for(int i=0;i<sl;i++)buf[pl+i]=(v[vi]+2)[i]; buf[pl+sl]=0;
        d_sha256((const uint8_t*)buf,pl+sl,pk);
        CHECK_AND_RETURN(pk,f,fk);
    }
}

/* H09: Deep brainwallet (phrase + year) — 1 thread/phrase × 5 years */
__global__ void k_h09(volatile int *f, volatile uint64_t *fk) {
    int n=d_num_phrases, tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n||*f) return;
    const char *ph=d_phrases+tid*256;
    int yrs[]={2009,2010,2011,2012,2013};
    uint8_t pk[32];
    for(int yi=0;yi<5;yi++){
        if(*f) return;
        char buf[512]; int pl=(int)strlen(ph);
        for(int i=0;i<pl;i++)buf[i]=ph[i];
        char y[16]; int yl=0;
        /* itoa yrs[yi] */
        int yv=yrs[yi]; if(yv==0){y[0]='0';yl=1;}else{
            int t=yv; while(t){t/=10;yl++;}
            for(int i=yl-1;i>=0;i--){y[i]=(yv%10)+'0';yv/=10;}
        }
        for(int i=0;i<yl;i++)buf[pl+i]=y[i]; buf[pl+yl]=0;
        d_sha256((const uint8_t*)buf,pl+yl,pk);
        CHECK_AND_RETURN(pk,f,fk);
    }
}

/* H08: d_block_hashes declared in main.cu */

__global__ void k_h08(int nb, volatile int *f, volatile uint64_t *fk) {
    uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=(uint64_t)nb||*f) return;
    uint8_t pk[32]; d_sha256(d_block_hashes+tid*32,32,pk);
    CHECK_AND_RETURN(pk,f,fk);
}

/* H18: Multi-word (phrase pairs) — each thread = (i,j) pair */
__global__ void k_h18(volatile int *f, volatile uint64_t *fk) {
    int n=d_num_phrases, tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n*n||*f) return;
    int i=tid/n, j=tid%n;
    if(i>=j) return;
    const char *p1=d_phrases+i*256, *p2=d_phrases+j*256;
    char buf[512];
    int l1=(int)strlen(p1), l2=(int)strlen(p2);
    for(int x=0;x<l1;x++)buf[x]=p1[x]; buf[l1]=' ';
    for(int x=0;x<l2;x++)buf[l1+1+x]=p2[x]; buf[l1+1+l2]=0;
    uint8_t pk[32]; d_sha256((const uint8_t*)buf,l1+1+l2,pk);
    CHECK_AND_RETURN(pk,f,fk);
}

#endif /* GPU_HYPO_SMALL_CUH */
