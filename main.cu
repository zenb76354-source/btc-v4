/* ================================================================
 *  MAIN.CU — BTC Recovery: ALL HYPOTHESES Pure GPU
 *  كل hypothesis = kernel منفصل في kernels_code.cu
 *  هذا الملف: host code فقط
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpu/kernels_api.h"

#define DICT_MAX 8192
#define MAX_PHRASES 4096
#define MAX_PHRASE_LEN 256
#define START_MS 1230768000000ULL
#define END_MS   1325376000000ULL
#define TOTAL_H36_KEYS (END_MS - START_MS)
#define H28_MAX 2000000
#define H08_MAX 200000

/* Target definitions */
__constant__ uint8_t d_targets[8*20];
__constant__ char d_dict[DICT_MAX];
__constant__ char d_phrases[MAX_PHRASES*MAX_PHRASE_LEN];
__constant__ int d_num_phrases;
__constant__ uint8_t d_block_hashes[200000*32];

/* External kernel declarations (defined in kernels_code.cu) */
extern __global__ void k21(void*,void*);
extern __global__ void k11(void*,void*);
extern __global__ void k14(void*,void*);
extern __global__ void k15(void*,void*);
extern __global__ void k41(void*,void*);
extern __global__ void k_gentxt(void*,void*);
extern __global__ void k28(uint64_t,uint64_t,void*,void*);
extern __global__ void k3(uint64_t,void*,void*);
extern __global__ void k20(uint64_t,void*,void*);
extern __global__ void k36(uint64_t,uint64_t,void*,void*);
extern __global__ void k8(int,void*,void*);
extern __global__ void k1(void*,void*);
extern __global__ void k9(void*,void*);
extern __global__ void k18(void*,void*);

/* Helpers */
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

/* Pack dict → send to d_dict */
static char host_dict[DICT_MAX];
static void send_dict(const char *w[]) {
    memset(host_dict,0,DICT_MAX); int pos=0;
    for(int i=0;w[i]&&pos<DICT_MAX-1;i++){
        int l=(int)strlen(w[i])+1; if(pos+l>=DICT_MAX)break;
        memcpy(host_dict+pos,w[i],l); pos+=l;
    }
    cudaMemcpyToSymbol(d_dict,host_dict,DICT_MAX);
}

/* Run dictionary kernel */
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

/* Run sequential kernel (H28, H03, H20, H36) */
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
        else if(type==36) k36<<<blk,threads>>>(START_MS+s,b,df,dfk);
        cudaDeviceSynchronize();
        cudaMemcpy(&hf,df,sizeof(int),cudaMemcpyDeviceToHost);
        if(hf){uint64_t hfk[4]; cudaMemcpy(hfk,dfk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
            found(nm,hfk,s); cudaFree(df); cudaFree(dfk); return 1;}
        s+=b;
        if(s%1000000==0)printf("[%s] %llu/%llu (%.1f%%)\n",nm,(unsigned long long)s,(unsigned long long)total,100.0*s/total);
    }
    printf("[%s] Done. %llums\n",nm,(unsigned long long)(time_ms()-t0));
    cudaFree(df); cudaFree(dfk); return 0;
}

/* H08 runner */
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

/* Phrase kernels */
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
    char hp[MAX_PHRASES][MAX_PHRASE_LEN]; int n=0;
    while(n<MAX_PHRASES&&fgets(hp[n],MAX_PHRASE_LEN,f)){
        size_t sl=strlen(hp[n]); while(sl>0&&(hp[n][sl-1]=='\n'||hp[n][sl-1]=='\r'))hp[n][--sl]=0;
        if(sl>0)n++;
    }
    fclose(f);
    char *flat=(char*)malloc(n*256); memset(flat,0,n*256);
    for(int i=0;i<n;i++) memcpy(flat+i*256,hp[i],strlen(hp[i])+1);
    cudaMemcpyToSymbol(d_phrases,flat,n*256);
    cudaMemcpyToSymbol(d_num_phrases,&n,sizeof(int));
    free(flat); return n;
}

int main() {
    printf("\n===== BTC RECOVERY ALL HYPOTHESES PURE GPU =====\n\n");
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("Device: %s SM%d.%d (%d SMs)\n\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);

    /* Targets */
    static const uint8_t ht[8*20] = {
        0xc8,0xe5,0x09,0xee,0xe7,0xf7,0xbc,0xbc,0x11,0x1f,0x31,0x56,0xc0,0x4f,0x0b,0xc1,0xd7,0xb1,0xdb,0xf5,
        0x9d,0x9a,0x9b,0x77,0x5b,0x1b,0xbe,0x33,0xe1,0xf1,0xba,0x7b,0xd0,0x50,0xc5,0x75,0xf6,0x2d,0xb0,0x91,
        0xdb,0x4b,0x1a,0x77,0x39,0x45,0x6d,0x7d,0x43,0x98,0xc1,0xa7,0x1d,0x04,0x94,0x50,0x42,0x66,0x5c,0x3a,
        0x39,0x9a,0x4f,0x8f,0x8f,0x73,0xd3,0x2b,0x8d,0x52,0x0e,0x6a,0x54,0x74,0x05,0xea,0x06,0x09,0x2e,0x2a,
        0x3c,0x09,0x4b,0xb7,0x04,0x84,0xc3,0x15,0x7e,0x40,0xfd,0xa5,0x36,0xe6,0xfb,0x64,0x16,0x78,0x0e,0xe2,
        0x35,0x7a,0xd8,0x6e,0x87,0xf3,0x15,0xa8,0x25,0x2e,0xde,0x8b,0x6a,0xb4,0xe3,0xe0,0xa9,0x75,0x44,0xaa,
        0x28,0x4c,0x34,0x0f,0x0e,0xbf,0x7a,0x10,0x0b,0xc7,0x0c,0x44,0x2f,0x83,0x19,0x77,0xaa,0xd7,0xb3,0xb7,
        0x7a,0x05,0xa1,0x5e,0xaf,0xbe,0x19,0xec,0xff,0x63,0xbc,0x7a,0x3d,0x3b,0x9d,0x3a,0xfd,0x75,0x00,0xa7
    };
    cudaMemcpyToSymbol(d_targets,ht,8*20);

    int have_ph = load_phrases();

    /* Phase 0: Tiny */
    printf("===== PHASE 0: TINY =====\n");
    const char *h21w[]={ "", NULL }; if(run_dict("H21",h21w,21)) return 0;
    const char *h11w[]={"1","2","3","4","5","10","100","1000","10000","1234567890","abcdef","password","passw0rd","12345678","btc123","btc2010","hello","HELLO","world","WORLD",NULL}; if(run_dict("H11",h11w,11)) return 0;
    const char *h14w[]={"1268728843","1279199023","1279203210","1284382196","1284608803","1285880600","1268811438","1268866685","1268894549","1268921836","1268933538","1268943264",NULL}; if(run_dict("H14",h14w,14)) return 0;
    const char *h15w[]={"20090101","20090103","20100101","20100316","20100522","20100711","20100715","20100910","20100916","2009-01-01","2010-03-16","2010-05-22","2010-07-11","2010-07-15","2010-09-10","January 3 2009","March 16 2010","May 22 2010",NULL}; if(run_dict("H15",h15w,15)) return 0;
    const char *h41w[]={"bitcoin","satoshi","password","private","secret","wallet","block","chain","money","coin","miner","gold","crypto","admin","master","root","key","btc","200","400","50","1200","2010","march","july","fifty","hundred","rich","lucky","empty","null",NULL}; if(run_dict("H41",h41w,41)) return 0;
    const char *h42w[]={"deadbeef","cafebabe","feedface","0xdeadbeef","dead","beef","cafe","babe","feed","face","deadbeefcafebabe","0123456789abcdef","00000001","00000002","aa55","1234","5678","abcd","c0ffee","b00b5",NULL}; if(run_dict("H42",h42w,0)) return 0;
    const char *h43w[]={"😂","🔥","🚀","💎","🙌","❤️","💰","✅","bitcoin😂","btc🚀","satoshi🔥","💎🙌","diamond hands","₿itcoin","₿TC","βitcoin","bitc0in","s4tosh1","btc2010🚀","฿itcoin","฿",NULL}; if(run_dict("H43",h43w,0)) return 0;
    const char *h30w[]={"50btc","100btc","200btc","400btc","650btc","1200btc","fifty","hundred","50","100","200","400","650","1200","my50","my100","my200","my400","my200btc","my1200btc","50coins","100coins","200coins",NULL}; if(run_dict("H30",h30w,0)) return 0;
    const char *h31w[]={"July2010","july2010","March2010","march2010","2010-03","2010-07","2010-09","201003","201007","201009","March16","July15","September10",NULL}; if(run_dict("H31",h31w,0)) return 0;
    const char *h32w[]={"March2010200","July2010400","2010March200","2010July400","03162010_200","07152010_400",NULL}; if(run_dict("H32",h32w,0)) return 0;
    const char *h33w[]={"mining","miner","mining2010","pool2010","miningpool","btcmining","slush","deepbit","btcguild","pool","p2pool","genesis","genesis block","block 0",NULL}; if(run_dict("H33",h33w,0)) return 0;
    const char *h35w[]={"50x4","4x50","50-50-50-50","50+50+50+50","fiftyfiftyfifty","weekly50",NULL}; if(run_dict("H35",h35w,0)) return 0;
    const char *h26w[]={"backup","wallet backup","walletbackup","wallet.dat","mybackup","mywallet","bitcoin-wallet","bitcoinwallet","btcwallet","recovery","recovery phrase","seed","seed phrase","passphrase","encrypted","electrum","blockchain.info","multibit","armory","bitcoind","bitcoin-qt","bitcoin core","satoshi","satoshi wallet",NULL}; if(run_dict("H26",h26w,0)) return 0;
    const char *h27w[]={"bitcoin.org","bitcointalk.org","blockchain.info","github.com/bitcoin","sourceforge.net","p2pfoundation.net","deepbit.net","slushpool.com","mtgox.com","http://bitcoin.org","https://bitcoin.org","satoshi.nakamoto","nakamoto","gavinandresen","hal finney","http://www.bitcoin.org",NULL}; if(run_dict("H27",h27w,0)) return 0;
    const char *h29w[]={"bitcoin2009","bitcoin2010","bitcoinwallet","bitcoinkey","bitcoin123","bitcoin!","bitcoinpass","Bitcoin2010","Bitcoin2009","BITCOIN2009","BITCOIN2010","btc2009","btc2010","btcwallet","btckey","BTC2009","BTC2010","satoshi2009","satoshi2010","satoshiwallet","Satoshi2009","Satoshi2010","bitcoin1","bitcoin01","bitcoin001","bitcoin123","bitcoin1234","bitcoin!!","bitcoin?","bitcoin.","bitcoin@","bitcoin#","bitcoin$","bitcoinfirst","bitcoinmy","bitcoinnew","bitcoinold","bitcointest","Bitcoinwallet","Bitcoinkey","Bitcoinpass","BTCwallet","BTCkey","BTCpass","BTC2009","BTC2010",NULL}; if(run_dict("H29",h29w,0)) return 0;
    const char *h25w[]={"bitcoin is awesome","i love bitcoin","bitcoin to the moon","HODL","hodl","to the moon","when lambo","Satoshi nakamoto","satoshi nakamoto","bitcoin paper","buy bitcoin","buy the dip","sell bitcoin","blockchain","decentralized","peer to peer","cryptocurrency","private key","public key","brainwallet","paper wallet","cold storage","hot wallet","50 BTC","50btc","fifty btc","400 BTC","400btc","casascius","physical bitcoin","bitcointalk","satoshi dice","bitcoin faucet","faucet",NULL}; if(run_dict("H25",h25w,0)) return 0;

    /* Phase 1 */
    printf("\n===== PHASE 1: MEDIUM =====\n");
    if(run_seq("H28",28,H28_MAX,100000)) return 0;
    if(run_h08()) return 0;
    if(have_ph>0&&run_phr("H01",1,1)) return 0;
    if(have_ph>0&&run_phr("H09",9,1)) return 0;
    if(have_ph>0&&run_h18()) return 0;
    if(run_seq("H03",3,31536000,100000)) return 0;
    if(run_seq("H20",20,31536000,100000)) return 0;

    /* Phase 2: H36 */
    printf("\n===== PHASE 2: BIG =====\n");
    if(run_seq("H36",36,TOTAL_H36_KEYS,10000000)) return 0;

    printf("\n===== ALL COMPLETE — No key found =====\n");
    return 0;
}
