/* ================================================================
 *  MAIN.CU — BTC Recovery: ALL HYPOTHESES Pure GPU v2
 *  كل hypothesis = kernel منفصل في kernels_code.cu
 *  هذا الملف: host code فقط (إطلاق kernels + إدارة البيانات)
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpu/kernels_api.h"

#define DICT_MAX      8192
#define MAX_PHRASES   4096
#define MAX_PHRASE_LEN 256
#define H08_MAX       200000

/* ---------- H36: عصر البيتكوين المبكر (2008-10-08 → 2011-06-28) ---------- */
#define H36_START   1223424000000ULL   /* 2008-10-08 (Bitcoin v0.1 released) */
#define H36_END     1309219200000ULL   /* 2011-06-28 */
#define H36_TOTAL   (H36_END - H36_START)  /* ≈ 85.8 مليار */

/* ---------- H28: PID range كامل ---------- */
#define H28_TOTAL   2000000000ULL      /* 0 → 2 مليار (Linux PID + pseudo-random) */

/* ---------- تواريخ key لكل target (ms epoch) ---------- */
#define A1_MS 1268728843000ULL
#define A3_MS 1279199023000ULL
#define A4_MS 1279203210000ULL
#define A5_MS 1279412745000ULL
#define A6_MS 1284111956000ULL
#define A7_MS 1284608803000ULL

/* ---------- مدى ± لكل target ---------- */
#define TW 86400000ULL   /* ±24 ساعة = 48 ساعة بحث */
#define TW2 604800000ULL /* ±7 أيام */

/* ---------- __device__ & __constant__ ---------- */
__constant__ uint8_t d_targets[8*20];
__constant__ char d_dict[DICT_MAX];
__constant__ int d_num_phrases;
__device__ char d_phrases[MAX_PHRASES*MAX_PHRASE_LEN];
__device__ uint8_t d_block_hashes[200000*32];

/*========== HELPERS ==========*/

static void print_key(const uint64_t *k) {
    for(int i=0;i<4;i++) printf("%016llx",(unsigned long long)k[i]); printf("\n");
}
static uint64_t time_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000+ts.tv_nsec/1000000;
}
static void found(const char *nm, uint64_t *k, uint64_t ex) {
    printf("\n*** [%s] KEY FOUND *** key: ",nm); print_key(k);
    FILE *f=fopen("found_key.txt","a");
    if(f){if(ex)fprintf(f,"[%s] ex=%llu key=",nm,(unsigned long long)ex);
        else fprintf(f,"[%s] key=",nm);
        for(int i=0;i<4;i++)fprintf(f,"%016llx",(unsigned long long)k[i]);fprintf(f,"\n");fclose(f);}
}

static char host_dict[DICT_MAX];
static void send_dict(const char *w[]) {
    memset(host_dict,0,DICT_MAX); int pos=0;
    for(int i=0;w[i]&&pos<DICT_MAX-1;i++){
        int l=(int)strlen(w[i])+1; if(pos+l>=DICT_MAX)break;
        memcpy(host_dict+pos,w[i],l); pos+=l;
    }
    cudaMemcpyToSymbol(d_dict,host_dict,DICT_MAX);
}

static int run_dict(const char *nm, const char *w[], int h21flag) {
    send_dict(w);
    int *df; uint64_t *dfk;
    cudaMalloc(&df,sizeof(int)); cudaMalloc(&dfk,4*sizeof(uint64_t));
    int hf=0; cudaMemcpy(df,&hf,sizeof(int),cudaMemcpyHostToDevice);
    uint64_t hz[4]={0}; cudaMemcpy(dfk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
    printf("[%s] ...\n",nm);
    if(h21flag==21) k21<<<1,1>>>(df,dfk);
    else if(h21flag==11) k11<<<1,1>>>(df,dfk);
    else if(h21flag==14) k14<<<1,1>>>(df,dfk);
    else if(h21flag==15) k15<<<1,1>>>(df,dfk);
    else if(h21flag==41) k41<<<1,1>>>(df,dfk);
    else k_gentxt<<<1,1>>>(df,dfk);
    cudaDeviceSynchronize();
    cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
    if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        found(nm,hfk,0); cudaFree(df); cudaFree(dfk); return 1;}
    printf("[%s] Done.\n",nm); cudaFree(df); cudaFree(dfk); return 0;
}

static int run_seq(const char *nm, int type, uint64_t total, uint64_t batch) {
    int *df; uint64_t *dfk;
    cudaMalloc(&df,sizeof(int)); cudaMalloc(&dfk,4*sizeof(uint64_t));
    uint64_t t0=time_ms(); int threads=256;
    printf("[%s] %llu keys...\n",nm,(unsigned long long)total);
    for(uint64_t s=0;s<total;){
        uint64_t b=(s+batch>total)?(total-s):batch;
        int hf=0; cudaMemcpy(df,&hf,sizeof(int),cudaMemcpyHostToDevice);
        uint64_t hz[4]={0}; cudaMemcpy(dfk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
        int blk=(int)((b+threads-1)/threads);
        if(type==28) k28<<<blk,threads>>>(s,b,df,dfk);
        else if(type==3) k3<<<blk,threads>>>(total,df,dfk);
        else if(type==20) k20<<<blk,threads>>>(total,df,dfk);
        else if(type==36) k36<<<blk,threads>>>(H36_START+s,b,df,dfk);
        else if(type==48) k48<<<blk,threads>>>(s,b,df,dfk);
        cudaDeviceSynchronize();
        cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
        if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
            found(nm,hfk,s); cudaFree(df); cudaFree(dfk); return 1;}
        s+=b;
        if(s%100000000==0)printf("[%s] %llu/%llu (%.1f%%)\n",nm,(unsigned long long)s,(unsigned long long)total,100.0*s/total);
    }
    printf("[%s] Done. %llums\n",nm,(unsigned long long)(time_ms()-t0));
    cudaFree(df); cudaFree(dfk); return 0;
}

/* H36 حول timestamp محدد */
static int run_h36_range(const char *nm, uint64_t start_ms, uint64_t count, uint64_t batch) {
    int *df; uint64_t *dfk;
    cudaMalloc(&df,sizeof(int)); cudaMalloc(&dfk,4*sizeof(uint64_t));
    uint64_t t0=time_ms(); int threads=256;
    printf("[%s] %llu keys...\n",nm,(unsigned long long)count);
    for(uint64_t s=0;s<count;){
        uint64_t b=(s+batch>count)?(count-s):batch;
        int hf=0; cudaMemcpy(df,&hf,sizeof(int),cudaMemcpyHostToDevice);
        uint64_t hz[4]={0}; cudaMemcpy(dfk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
        int blk=(int)((b+threads-1)/threads);
        k36<<<blk,threads>>>(start_ms+s,b,df,dfk);
        cudaDeviceSynchronize();
        cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
        if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
            found(nm,hfk,start_ms+s); cudaFree(df); cudaFree(dfk); return 1;}
        s+=b;
        if(s%10000000==0)printf("[%s] %llu/%llu\n",nm,(unsigned long long)s,(unsigned long long)count);
    }
    printf("[%s] Done. %llums\n",nm,(unsigned long long)(time_ms()-t0));
    cudaFree(df); cudaFree(dfk); return 0;
}

/* H28 حول timestamp محدد (timestamp ms → kernel يخلط مع PID) */
static int run_h28_block(const char *nm, uint64_t block_id) {
    /* نرسل block_id كـ start للـ kernel (كل PID مرة) */
    return run_seq(nm,28,H28_TOTAL,1000000);
}

static int run_h08() {
    uint8_t *hh=(uint8_t*)malloc(H08_MAX*32);
    for(int i=0;i<H08_MAX;i++) for(int j=0;j<32;j++) hh[i*32+j]=(uint8_t)(i+j);
    cudaMemcpyToSymbol(d_block_hashes,hh,H08_MAX*32); free(hh);
    int *df; uint64_t *dfk;
    cudaMalloc(&df,sizeof(int)); cudaMalloc(&dfk,4*sizeof(uint64_t));
    int hf=0; cudaMemcpy(df,&hf,sizeof(int),cudaMemcpyHostToDevice);
    uint64_t hz[4]={0}; cudaMemcpy(dfk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
    printf("[H08] %d blocks...\n",H08_MAX);
    int threads=256; int blk=(H08_MAX+threads-1)/threads;
    k8<<<blk,threads>>>(H08_MAX,df,dfk);
    cudaDeviceSynchronize();
    cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
    if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        found("H08",hfk,0); cudaFree(df); cudaFree(dfk); return 1;}
    printf("[H08] Done.\n"); cudaFree(df); cudaFree(dfk); return 0;
}

static int run_phr(const char *nm, int type, int min) {
    int n; cudaMemcpyFromSymbol(&n,d_num_phrases,sizeof(int));
    if(n<min){printf("[%s] Need %d phrases\n",nm,min);return 0;}
    int *df; uint64_t *dfk;
    cudaMalloc(&df,sizeof(int)); cudaMalloc(&dfk,4*sizeof(uint64_t));
    int hf=0; cudaMemcpy(df,&hf,sizeof(int),cudaMemcpyHostToDevice);
    uint64_t hz[4]={0}; cudaMemcpy(dfk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
    printf("[%s] %d phrases...\n",nm,n);
    int threads=128; int blk=(n+threads-1)/threads;
    if(type==1) k1<<<blk,threads>>>(df,dfk);
    else k9<<<blk,threads>>>(df,dfk);
    cudaDeviceSynchronize();
    cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
    if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        found(nm,hfk,0); cudaFree(df); cudaFree(dfk); return 1;}
    printf("[%s] Done.\n",nm); cudaFree(df); cudaFree(dfk); return 0;
}
static int run_h18() {
    int n; cudaMemcpyFromSymbol(&n,d_num_phrases,sizeof(int));
    if(n<2){printf("[H18] Skip\n");return 0;}
    uint64_t tot=(uint64_t)n*(n-1)/2;
    printf("[H18] %llu pairs...\n",(unsigned long long)tot);
    int *df; uint64_t *dfk;
    cudaMalloc(&df,sizeof(int)); cudaMalloc(&dfk,4*sizeof(uint64_t));
    int hf=0; cudaMemcpy(df,&hf,sizeof(int),cudaMemcpyHostToDevice);
    uint64_t hz[4]={0}; cudaMemcpy(dfk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
    int threads=128; int blk=(int)((tot+threads-1)/threads);
    k18<<<blk,threads>>>(df,dfk);
    cudaDeviceSynchronize();
    cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
    if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        found("H18",hfk,0); cudaFree(df); cudaFree(dfk); return 1;}
    printf("[H18] Done.\n"); cudaFree(df); cudaFree(dfk); return 0;
}

static int load_phrases() {
    FILE *f=fopen("phrases.txt","r"); if(!f) return 0;
    char hp[MAX_PHRASES][MAX_PHRASE_LEN];
    int n=0;
    while(n<MAX_PHRASES && fgets(hp[n],MAX_PHRASE_LEN,f)){
        size_t sl=strlen(hp[n]);
        while(sl>0 && (hp[n][sl-1]=='\n'||hp[n][sl-1]=='\r')) hp[n][--sl]=0;
        if(sl>0) n++;
    }
    fclose(f);
    char *flat=(char*)malloc(n*256); memset(flat,0,n*256);
    for(int i=0;i<n;i++) memcpy(flat+i*256,hp[i],strlen(hp[i])+1);
    cudaMemcpyToSymbol(d_phrases,flat,n*256);
    cudaMemcpyToSymbol(d_num_phrases,&n,sizeof(int));
    free(flat); return n;
}

/*========== MAIN ==========*/
int main() {
    printf("\n===== BTC RECOVERY v2 — PURE GPU (corrected targets) =====\n\n");
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("Device: %s SM%d.%d (%d SMs)\n\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);

    /* -------------------- CORRECT TARGETS (from common/targets.h) -------------------- */
    static const uint8_t ht[8*20] = {
        0x14,0x4d,0xe4,0x97,0x1a,0x30,0x9f,0x65,0x6a,0x25,
        0x98,0xf9,0x74,0x63,0xe2,0x1f,0xc4,0xe6,0x0f,0xe1,  /* A1: 12rMpw5... 400 BTC */
        0xb3,0x46,0xa3,0xbc,0xe0,0xe6,0xf5,0xe8,0xd0,0x1b,
        0x6a,0x73,0x9c,0x05,0x01,0x49,0x2d,0xd5,0xf5,0xeb,  /* A2: 1HLvaTs... 9260 BTC */
        0xbc,0x30,0xaf,0x9c,0xfb,0xa5,0x5e,0xa6,0x13,0x74,
        0xf9,0x8b,0x3e,0xf3,0x18,0x55,0x70,0xb7,0x98,0x18,  /* A3: 1JA4Mpu... 400 BTC */
        0x18,0xf2,0xdf,0x2f,0x55,0xe0,0xdd,0x03,0x98,0x2b,
        0x35,0x8b,0x5f,0xb7,0x49,0x1d,0x98,0xae,0x94,0xaf,  /* A4: 13GvAdk... 200 BTC */
        0x88,0xbb,0x33,0x3d,0x5d,0xff,0xea,0x68,0x28,0xbd,
        0x86,0x8e,0x3a,0xe5,0x70,0x09,0x75,0xc8,0xfa,0x4c,  /* A5: 1DTy9z4... 200 BTC */
        0xe0,0xbe,0x57,0x0f,0x09,0x09,0xa4,0xee,0xdc,0x8e,
        0x82,0x65,0x2c,0x7f,0x39,0x10,0x38,0xf0,0x0c,0xcc,  /* A6: 1MVLP2k... 1200 BTC */
        0x30,0x59,0xc8,0x38,0x4e,0x7e,0xbf,0x41,0xe0,0x3c,
        0x0d,0xa3,0xfa,0x7e,0x69,0xfa,0xb4,0x07,0x64,0x9d,  /* A7: 15QezNw... 200 BTC */
        0x59,0x2f,0xc3,0x99,0x00,0x26,0x33,0x4c,0x8c,0x6f,
        0xb2,0xb9,0xda,0x45,0x71,0x79,0xcd,0xb5,0xc6,0x88,  /* E1: 198aMn6... 250 BTC */
    };
    cudaMemcpyToSymbol(d_targets,ht,8*20);

    int have_ph = load_phrases();

    /* ===== PHASE 0: قاموس صغير (لم يتغير) ===== */
    printf("===== PHASE 0: TINY DICTS =====\n");
    const char *h21w[]={ "", NULL }; if(run_dict("H21",h21w,21)) return 0;
    const char *h11w[]={"1","2","3","4","5","10","100","1000","10000","1234567890","abcdef",
        "password","passw0rd","12345678","btc123","btc2010","hello","HELLO","world","WORLD",NULL};
    if(run_dict("H11",h11w,11)) return 0;
    const char *h14w[]={"1268728843","1279199023","1279203210","1284382196","1284608803",
        "1285880600","1268811438","1268866685","1268894549","1268921836","1268933538","1268943264",NULL};
    if(run_dict("H14",h14w,14)) return 0;
    const char *h15w[]={"20090101","20090103","20100101","20100316","20100522","20100711",
        "20100715","20100910","20100916","2009-01-01","2010-03-16","2010-05-22","2010-07-11",
        "2010-07-15","2010-09-10","January 3 2009","March 16 2010","May 22 2010",NULL};
    if(run_dict("H15",h15w,15)) return 0;
    const char *h41w[]={"bitcoin","satoshi","password","private","secret","wallet","block",
        "chain","money","coin","miner","gold","crypto","admin","master","root","key","btc",
        "200","400","50","1200","2010","march","july","fifty","hundred","rich","lucky","empty","null",NULL};
    if(run_dict("H41",h41w,41)) return 0;
    const char *h33w[]={"mining","miner","mining2010","pool2010","miningpool","btcmining",
        "slush","deepbit","btcguild","pool","p2pool","genesis","genesis block","block 0",NULL};
    if(run_dict("H33",h33w,0)) return 0;
    const char *h26w[]={"backup","wallet backup","walletbackup","wallet.dat","mybackup",
        "mywallet","bitcoin-wallet","bitcoinwallet","btcwallet","recovery","recovery phrase",
        "seed","seed phrase","passphrase","encrypted","electrum","blockchain.info","multibit",
        "armory","bitcoind","bitcoin-qt","bitcoin core","satoshi","satoshi wallet",NULL};
    if(run_dict("H26",h26w,0)) return 0;
    const char *h27w[]={"bitcoin.org","bitcointalk.org","blockchain.info","github.com/bitcoin",
        "sourceforge.net","p2pfoundation.net","deepbit.net","slushpool.com","mtgox.com",
        "http://bitcoin.org","https://bitcoin.org","satoshi.nakamoto","nakamoto",
        "gavinandresen","hal finney","http://www.bitcoin.org",NULL};
    if(run_dict("H27",h27w,0)) return 0;
    const char *h29w[]={"bitcoin2009","bitcoin2010","bitcoinwallet","bitcoinkey","bitcoin123",
        "bitcoin!","bitcoinpass","Bitcoin2010","Bitcoin2009","BITCOIN2009","BITCOIN2010",
        "btc2009","btc2010","btcwallet","btckey","BTC2009","BTC2010","satoshi2009","satoshi2010",
        "satoshiwallet","Satoshi2009","Satoshi2010","bitcoin1","bitcoin01","bitcoin001",
        "bitcoin123","bitcoin1234","bitcoin!!","bitcoin?","bitcoin.","bitcoin@","bitcoin#",
        "bitcoin$","bitcoinfirst","bitcoinmy","bitcoinnew","bitcoinold","bitcointest",NULL};
    if(run_dict("H29",h29w,0)) return 0;
    const char *h25w[]={"bitcoin is awesome","i love bitcoin","bitcoin to the moon","HODL",
        "hodl","to the moon","when lambo","Satoshi nakamoto","satoshi nakamoto","bitcoin paper",
        "buy bitcoin","buy the dip","sell bitcoin","blockchain","decentralized","peer to peer",
        "cryptocurrency","private key","public key","brainwallet","paper wallet","cold storage",
        "50 BTC","50btc","fifty btc","400 BTC","400btc","casascius","bitcointalk",
        "satoshi dice","bitcoin faucet","faucet",NULL};
    if(run_dict("H25",h25w,0)) return 0;

    /* ===== PHASE 1: MEDIUM (2M keys أو أقل لكل kernel) ===== */
    printf("\n===== PHASE 1: MEDIUM SEQUENTIAL =====\n");

    /* H28: PID pseudo-random 0 → 2 مليار */
    if(run_seq("H28",28,H28_TOTAL,10000000)) return 0;

    /* H08: block hashes */
    if(run_h08()) return 0;

    /* phrases */
    if(have_ph>0 && run_phr("H01",1,1)) return 0;
    if(have_ph>0 && run_phr("H09",9,1)) return 0;
    if(have_ph>0 && run_h18()) return 0;

    /* H03: timestamp + PID/extra (full year range حول كل target) */
    if(run_seq("H03",3,31536000,500000)) return 0;

    /* H20: timestamp only */
    if(run_seq("H20",20,31536000,500000)) return 0;

    /* ===== PHASE 2: H36 الكامل (2008-10 → 2011-06) ===== */
    printf("\n===== PHASE 2: FULL H36 ms SWEEP =====\n");
    if(run_seq("H36",36,H36_TOTAL,50000000)) return 0;

    /* ===== PHASE 3: H36 حول التواريخ المحددة لكل target (±7 أيام) ===== */
    printf("\n===== PHASE 3: TARGET WINDOWS =====\n");

    /* A1: 2010-03-16 ±7 أيام */
    if(run_h36_range("H36-A1",A1_MS-TW2,TW2*2,50000000)) return 0;

    /* A3/A4: 2010-07-15 ±7 أيام (كلاهما نفس اليوم) */
    if(run_h36_range("H36-A3A4",A3_MS-TW2,TW2*2,50000000)) return 0;

    /* A5: 2010-07-17 */
    if(run_h36_range("H36-A5",A5_MS-TW2,TW2*2,50000000)) return 0;

    /* A6: 2010-09-10 */
    if(run_h36_range("H36-A6",A6_MS-TW2,TW2*2,50000000)) return 0;

    /* A7: 2010-09-16 */
    if(run_h36_range("H36-A7",A7_MS-TW2,TW2*2,50000000)) return 0;

    /* ===== PHASE 4: H36 ±24h (دقيق) ===== */
    printf("\n===== PHASE 4: TARGET WINDOWS ±24h =====\n");
    if(run_h36_range("H36-A1f",A1_MS-TW,TW*2,3000000)) return 0;
    if(run_h36_range("H36-A3f",A3_MS-TW,TW*2,3000000)) return 0;
    if(run_h36_range("H36-A4f",A4_MS-TW,TW*2,3000000)) return 0;
    if(run_h36_range("H36-A5f",A5_MS-TW,TW*2,3000000)) return 0;
    if(run_h36_range("H36-A6f",A6_MS-TW,TW*2,3000000)) return 0;
    if(run_h36_range("H36-A7f",A7_MS-TW,TW*2,3000000)) return 0;

    /* ===== PHASE 5: MEGA INTEGER — كل uint64 0 → 2^48 ===== */
    printf("\n===== PHASE 5: MEGA INTEGER (0 to 2^48) =====\n");
    if(run_seq("H48",48,(1ULL<<48),50000000)) return 0;

    /* ===== PHASE 6: ADDRESS/KEY DERIVED ===== */
    printf("\n===== PHASE 6: ADDRESS/KEY HASHES =====\n");
    {
        int *df; uint64_t *dfk;
        cudaMalloc(&df,sizeof(int)); cudaMalloc(&dfk,4*sizeof(uint64_t));
        int hf=0; cudaMemcpy(df,&hf,sizeof(int),cudaMemcpyHostToDevice);
        uint64_t hz[4]={0}; cudaMemcpy(dfk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
        printf("[H50] Address/key based keys...\n");
        k50<<<1,1>>>(df,dfk);
        cudaDeviceSynchronize();
        cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
        if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
            found("H50",hfk,0); cudaFree(df); cudaFree(dfk); return 0;}
        printf("[H50] Done.\n");
        cudaFree(df); cudaFree(dfk);
    }

    /* ===== PHASE 7: REVERSE DICT ===== */
    printf("\n===== PHASE 7: REVERSE DICT =====\n");
    {
        int *df; uint64_t *dfk;
        cudaMalloc(&df,sizeof(int)); cudaMalloc(&dfk,4*sizeof(uint64_t));
        int hf=0; cudaMemcpy(df,&hf,sizeof(int),cudaMemcpyHostToDevice);
        uint64_t hz[4]={0}; cudaMemcpy(dfk,hz,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
        printf("[H51] Reverse dict words...\n");
        k51<<<1,1>>>(df,dfk);
        cudaDeviceSynchronize();
        cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
        if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
            found("H51",hfk,0); cudaFree(df); cudaFree(dfk); return 0;}
        printf("[H51] Done.\n");
        cudaFree(df); cudaFree(dfk);
    }

    printf("\n===== ALL COMPLETE — No key found =====\n");
    return 0;
}
