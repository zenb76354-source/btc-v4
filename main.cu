/* ================================================================
 *  MAIN.CU — BTC Recovery: ALL HYPOTHESES Pure GPU
 *  
 *  استراتيجية: kernel واحد متعدد (k_hypo_master) ياخذ type ID
 *  يولد private key → ECC → hash160 → يقارن 8 أهداف
 *  كل الـ string data محملة من host إلى __constant__ d_dict
 *  
 *  لا إهمال، لا توقع — كل مفتاح يمر على GPU كامل
 * ================================================================ */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpu/kernels.cuh"
#include "gpu/gpu_hypo_small.cuh"

/* ================================================================
 *  CONSTANTS
 * ================================================================ */

#define NUM_TARGETS 8
#define DICT_MAX 8192          /* must match cuh */
#define MAX_PHRASES 4096
#define MAX_PHRASE_LEN 256

/* Timestamp ranges */
#define START_MS 1230768000000ULL
#define END_MS   1325376000000ULL
#define TOTAL_H36_KEYS (END_MS - START_MS)

#define H28_MAX 2000000
#define H08_MAX 200000

/* Hypothesis type codes for k_hypo_master */
#define H21   0
#define H11  11
#define H14  14
#define H15  15
#define H41  41
#define H42  42
#define H43  43
#define H30  30
#define H31  31
#define H32  32
#define H33  33
#define H35  35
#define H34  34
#define H26  26
#define H27  27
#define H29  29
#define H25  25
#define H28  4228
#define H03  4203
#define H20  4220
#define H36  4236

/* ================================================================
 *  CONSTANT MEMORY (device-side)
 *  d_targets, d_dict, d_phrases, etc. declared in gpu_hypo_small.cuh
 * ================================================================ */

/* Constant memory defined in gpu_hypo_small.cuh */
/* d_targets, d_dict, d_phrases, d_num_phrases, d_block_hashes */

/* ================================================================
 *  HELPERS
 * ================================================================ */

static void print_key(const uint64_t *k) {
    for(int i=0;i<4;i++) printf("%016llx", (unsigned long long)k[i]);
    printf("\n");
}

static uint64_t time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000 + ts.tv_nsec/1000000;
}

static void found(const char *name, uint64_t *k, uint64_t extra) {
    printf("\n*** [%s] KEY FOUND *** key: ", name);
    print_key(k);
    FILE *f=fopen("found_key.txt","a");
    if(f){
        if(extra) fprintf(f,"[%s] extra=%llu key=", name, (unsigned long long)extra);
        else fprintf(f,"[%s] key=", name);
        for(int i=0;i<4;i++) fprintf(f,"%016llx",(unsigned long long)k[i]);
        fprintf(f,"\n"); fclose(f);
    }
}

/* ================================================================
 *  LOAD ALL DICTIONARIES → d_dict
 *  Pack null-terminated strings consecutively
 * ================================================================ */

static int pack_dict(const char *words[], char *buf, int buf_size) {
    int pos=0;
    for(int i=0; words[i] && pos<buf_size-1; i++){
        int l=(int)strlen(words[i])+1;
        if(pos+l >= buf_size){printf("[WARN] dict overflow\n"); break;}
        memcpy(buf+pos, words[i], l);
        pos += l;
    }
    if(pos<buf_size) buf[pos]=0; /* terminate */
    return pos;
}

/* ================================================================
 *  RUN SINGLE DICT KERNEL
 * ================================================================ */

static int run_dict_hypo(const char *name, uint64_t type, const char *words[]) {
    char host_dict[DICT_MAX];
    memset(host_dict, 0, DICT_MAX);
    int used=pack_dict(words, host_dict, DICT_MAX);
    cudaMemcpyToSymbol(d_dict, host_dict, DICT_MAX);

    int *d_flag; uint64_t *d_fk;
    cudaMalloc(&d_flag, sizeof(int)); cudaMalloc(&d_fk, 4*sizeof(uint64_t));
    int h_flag=0; cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
    uint64_t h_zero[4]={0}; cudaMemcpy(d_fk, h_zero, 4*sizeof(uint64_t), cudaMemcpyHostToDevice);

    printf("[%s] %d strings...\n", name, used ? 1 : 0);
    k_hypo_master<<<1,1>>>(type, used, d_flag, d_fk);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost);
    if(h_flag){
        uint64_t h_fk[4]; cudaMemcpy(h_fk,d_fk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
        found(name, h_fk, 0);
        cudaFree(d_flag); cudaFree(d_fk);
        return 1;
    }
    printf("[%s] Done.\n", name);
    cudaFree(d_flag); cudaFree(d_fk);
    return 0;
}

/* ================================================================
 *  RUN SEQUENTIAL HYPOTHESES (H28, H03, H20, H36)
 * ================================================================ */

static int run_seq_hypo(const char *name, uint64_t type, uint64_t count, uint64_t batch_size) {
    int *d_flag; uint64_t *d_fk;
    cudaMalloc(&d_flag, sizeof(int)); cudaMalloc(&d_fk, 4*sizeof(uint64_t));
    int threads=256;

    printf("[%s] %llu keys...\n", name, (unsigned long long)count);
    uint64_t t0=time_ms();
    uint64_t reported=0;
    int report_interval=100;

    for(uint64_t s=0; s<count; ){
        uint64_t batch=(s+batch_size>count)?(count-s):batch_size;
        int h_flag=0; cudaMemcpy(d_flag,&h_flag,sizeof(int),cudaMemcpyHostToDevice);
        uint64_t h_zero[4]={0}; cudaMemcpy(d_fk,h_zero,4*sizeof(uint64_t),cudaMemcpyHostToDevice);
        int blocks=(int)((batch+threads-1)/threads);
        k_hypo_master<<<blocks,threads>>>(type, batch+s, d_flag, d_fk);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost);
        if(h_flag){
            uint64_t h_fk[4]; cudaMemcpy(h_fk,d_fk,4*sizeof(uint64_t),cudaMemcpyDeviceToHost);
            found(name, h_fk, s);
            cudaFree(d_flag); cudaFree(d_fk);
            return 1;
        }
        s+=batch;
        int ln=(int)(s/batch_size);
        if(ln%report_interval==0&&(ln>(int)(reported/batch_size))){
            reported=s;
            double rate=(double)s/((time_ms()-t0)/1000.0);
            printf("[%s] %llu / %llu (%.1f%%) — %.2f Mkeys/s\n",
                   name,(unsigned long long)s,(unsigned long long)count,
                   100.0*s/count,rate/1e6);
        }
    }
    printf("[%s] Done. %llums\n", name, (unsigned long long)(time_ms()-t0));
    cudaFree(d_flag); cudaFree(d_fk);
    return 0;
}

/* ================================================================
 *  LOAD PHRASES FROM FILE
 * ================================================================ */

static int load_phrases() {
    FILE *f=fopen("phrases.txt","r");
    if(!f){printf("[WARN] No phrases.txt\n");return 0;}
    char hp[MAX_PHRASES][MAX_PHRASE_LEN];
    int n=0;
    while(n<MAX_PHRASES && fgets(hp[n],MAX_PHRASE_LEN,f)){
        size_t sl=strlen(hp[n]);
        while(sl>0&&(hp[n][sl-1]=='\n'||hp[n][sl-1]=='\r')) hp[n][--sl]=0;
        if(sl>0) n++;
    }
    fclose(f);
    printf("[OK] %d phrases loaded\n", n);
    char *flat=(char*)malloc(n*256);
    memset(flat,0,n*256);
    for(int i=0;i<n;i++) memcpy(flat+i*256,hp[i],strlen(hp[i])+1);
    cudaMemcpyToSymbol(d_phrases,flat,n*256);
    cudaMemcpyToSymbol(d_num_phrases,&n,sizeof(int));
    free(flat);
    return n;
}

/* ================================================================
 *  MAIN
 * ================================================================ */

int main() {
    printf("\n============================================\n");
    printf(" BTC RECOVERY — ALL HYPOTHESES PURE GPU\n");
    printf("============================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    printf("Device: %s (SM %d.%d)\n", prop.name,prop.major,prop.minor);
    printf("SMs: %d\n\n", prop.multiProcessorCount);

    /* Targets */
    static const uint8_t h_targets[8*20] = {
        0xc8,0xe5,0x09,0xee,0xe7,0xf7,0xbc,0xbc,0x11,0x1f,0x31,0x56,0xc0,0x4f,0x0b,0xc1,0xd7,0xb1,0xdb,0xf5,
        0x9d,0x9a,0x9b,0x77,0x5b,0x1b,0xbe,0x33,0xe1,0xf1,0xba,0x7b,0xd0,0x50,0xc5,0x75,0xf6,0x2d,0xb0,0x91,
        0xdb,0x4b,0x1a,0x77,0x39,0x45,0x6d,0x7d,0x43,0x98,0xc1,0xa7,0x1d,0x04,0x94,0x50,0x42,0x66,0x5c,0x3a,
        0x39,0x9a,0x4f,0x8f,0x8f,0x73,0xd3,0x2b,0x8d,0x52,0x0e,0x6a,0x54,0x74,0x05,0xea,0x06,0x09,0x2e,0x2a,
        0x3c,0x09,0x4b,0xb7,0x04,0x84,0xc3,0x15,0x7e,0x40,0xfd,0xa5,0x36,0xe6,0xfb,0x64,0x16,0x78,0x0e,0xe2,
        0x35,0x7a,0xd8,0x6e,0x87,0xf3,0x15,0xa8,0x25,0x2e,0xde,0x8b,0x6a,0xb4,0xe3,0xe0,0xa9,0x75,0x44,0xaa,
        0x28,0x4c,0x34,0x0f,0x0e,0xbf,0x7a,0x10,0x0b,0xc7,0x0c,0x44,0x2f,0x83,0x19,0x77,0xaa,0xd7,0xb3,0xb7,
        0x7a,0x05,0xa1,0x5e,0xaf,0xbe,0x19,0xec,0xff,0x63,0xbc,0x7a,0x3d,0x3b,0x9d,0x3a,0xfd,0x75,0x00,0xa7
    };
    cudaMemcpyToSymbol(d_targets, h_targets, 8*20);
    printf("[OK] Targets loaded\n\n");

    /* Load phrases */
    int have_ph = load_phrases();

    /* ============================================================
     *  PHASE 0: TINY DICT HYPOTHESES
     *  كلها من الأصغر → الأكبر
     * ============================================================ */

    printf("===== PHASE 0: TINY HYPOTHESES =====\n");

    const char *h21_w[]={ "", NULL };
    if(run_dict_hypo("H21", H21, h21_w)) return 0;

    const char *h11_w[]={"1","2","3","4","5","10","100","1000","10000",
        "1234567890","abcdef","password","passw0rd","12345678",
        "btc123","btc2010","hello","HELLO","world","WORLD",NULL};
    if(run_dict_hypo("H11", H11, h11_w)) return 0;

    const char *h14_w[]={"1268728843","1279199023","1279203210","1284382196",
        "1284608803","1285880600","1268811438","1268866685",
        "1268894549","1268921836","1268933538","1268943264",NULL};
    if(run_dict_hypo("H14", H14, h14_w)) return 0;

    const char *h15_w[]={"20090101","20090103","20100101","20100316","20100522",
        "20100711","20100715","20100910","20100916",
        "2009-01-01","2010-03-16","2010-05-22",
        "2010-07-11","2010-07-15","2010-09-10",
        "January 3 2009","March 16 2010","May 22 2010",NULL};
    if(run_dict_hypo("H15", H15, h15_w)) return 0;

    const char *h41_w[]={"bitcoin","satoshi","password","private","secret",
        "wallet","block","chain","money","coin","miner","gold","crypto",
        "admin","master","root","key","btc","200","400","50","1200",
        "2010","march","july","fifty","hundred","rich","lucky","empty","null",NULL};
    if(run_dict_hypo("H41", H41, h41_w)) return 0;

    const char *h42_w[]={"deadbeef","cafebabe","feedface","0xdeadbeef",
        "dead","beef","cafe","babe","feed","face",
        "deadbeefcafebabe","0123456789abcdef","00000001","00000002",
        "aa55","1234","5678","abcd","c0ffee","b00b5",NULL};
    if(run_dict_hypo("H42", H42, h42_w)) return 0;

    const char *h43_w[]={"😂","🔥","🚀","💎","🙌","❤️","💰","✅",
        "bitcoin😂","btc🚀","satoshi🔥",
        "💎🙌","diamond hands","₿itcoin","₿TC","βitcoin",
        "bitc0in","s4tosh1","btc2010🚀","฿itcoin","฿",NULL};
    if(run_dict_hypo("H43", H43, h43_w)) return 0;

    const char *h30_w[]={"50btc","100btc","200btc","400btc","650btc","1200btc",
        "fifty","hundred","50","100","200","400","650","1200",
        "my50","my100","my200","my400","my200btc","my1200btc",
        "50coins","100coins","200coins",NULL};
    if(run_dict_hypo("H30", H30, h30_w)) return 0;

    const char *h31_w[]={"July2010","july2010","March2010","march2010",
        "2010-03","2010-07","2010-09","201003","201007","201009",
        "March16","July15","September10",NULL};
    if(run_dict_hypo("H31", H31, h31_w)) return 0;

    const char *h32_w[]={"March2010200","July2010400","2010March200","2010July400",
        "03162010_200","07152010_400",NULL};
    if(run_dict_hypo("H32", H32, h32_w)) return 0;

    const char *h33_w[]={"mining","miner","mining2010","pool2010","miningpool",
        "btcmining","slush","deepbit","btcguild","pool","p2pool",
        "genesis","genesis block","block 0",NULL};
    if(run_dict_hypo("H33", H33, h33_w)) return 0;

    const char *h35_w[]={"50x4","4x50","50-50-50-50","50+50+50+50",
        "fiftyfiftyfifty","weekly50",NULL};
    if(run_dict_hypo("H35", H35, h35_w)) return 0;

    const char *h26_w[]={"backup","wallet backup","walletbackup","wallet.dat",
        "mybackup","mywallet","bitcoin-wallet","bitcoinwallet","btcwallet",
        "recovery","recovery phrase","seed","seed phrase",
        "passphrase","encrypted","electrum","blockchain.info",
        "multibit","armory","bitcoind","bitcoin-qt","bitcoin core",
        "satoshi","satoshi wallet",NULL};
    if(run_dict_hypo("H26", H26, h26_w)) return 0;

    const char *h27_w[]={"bitcoin.org","bitcointalk.org","blockchain.info",
        "github.com/bitcoin","sourceforge.net","p2pfoundation.net",
        "deepbit.net","slushpool.com","mtgox.com",
        "http://bitcoin.org","https://bitcoin.org",
        "satoshi.nakamoto","nakamoto","gavinandresen","hal finney",
        "http://www.bitcoin.org",NULL};
    if(run_dict_hypo("H27", H27, h27_w)) return 0;

    const char *h29_w[]={"bitcoin2009","bitcoin2010","bitcoinwallet","bitcoinkey",
        "bitcoin123","bitcoin!","bitcoinpass","Bitcoin2010","Bitcoin2009",
        "BITCOIN2009","BITCOIN2010","btc2009","btc2010","btcwallet","btckey",
        "BTC2009","BTC2010","satoshi2009","satoshi2010","satoshiwallet",
        "Satoshi2009","Satoshi2010",
        "bitcoin1","bitcoin01","bitcoin001","bitcoin123","bitcoin1234",
        "bitcoin!!","bitcoin?","bitcoin.","bitcoin@","bitcoin#","bitcoin$",
        "bitcoinfirst","bitcoinmy","bitcoinnew","bitcoinold","bitcointest",
        "Bitcoinwallet","Bitcoinkey","Bitcoinpass",
        "BTCwallet","BTCkey","BTCpass","BTC2009","BTC2010",NULL};
    if(run_dict_hypo("H29", H29, h29_w)) return 0;

    const char *h25_w[]={"bitcoin is awesome","i love bitcoin","bitcoin to the moon",
        "HODL","hodl","to the moon","when lambo",
        "Satoshi nakamoto","satoshi nakamoto","bitcoin paper",
        "buy bitcoin","buy the dip","sell bitcoin",
        "blockchain","decentralized","peer to peer","cryptocurrency",
        "private key","public key","brainwallet","paper wallet",
        "cold storage","hot wallet",
        "50 BTC","50btc","fifty btc","400 BTC","400btc",
        "casascius","physical bitcoin","bitcointalk",
        "satoshi dice","bitcoin faucet","faucet",NULL};
    if(run_dict_hypo("H25", H25, h25_w)) return 0;

    /* ============================================================
     *  PHASE 1: SMALL/MEDIUM SEQUENTIAL
     * ============================================================ */

    printf("\n===== PHASE 1: MEDIUM HYPOTHESES =====\n");

    if(run_seq_hypo("H28", H28, H28_MAX, 100000)) return 0;
    if(run_seq_hypo("H03", H03, 31536000*2, 100000)) return 0;  /* 1yr × PID range */
    if(run_seq_hypo("H20", H20, 31536000, 100000)) return 0;    /* 1yr srand */

    /* ============================================================
     *  PHASE 2: BIG — H36
     * ============================================================ */

    printf("\n===== PHASE 2: BIG HYPOTHESES =====\n");
    if(run_seq_hypo("H36", H36, TOTAL_H36_KEYS, 10000000)) return 0;

    printf("\n============================================\n");
    printf(" ALL HYPOTHESES COMPLETE — No key found\n");
    printf("============================================\n\n");
    return 0;
}
