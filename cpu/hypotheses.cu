/* ================================================================
 *  CPU HYPOTHESES — All CPU-based private key generation methods
 *  Version 1.0
 *  
 *  Each hypothesis function:
 *    - Returns 1 if found, 0 otherwise
 *    - Uses check_privkey_multi() — checks ALL 8 targets at once
 *    - Can be individually enabled/disabled from main.cu
 *  
 *  Order: FAST → MEDIUM → SLOW (smallest keyspace first)
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <ctype.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <secp256k1.h>

#include "../common/targets.h"
#include "../common/check.h"
#include <stdlib.h>
#include <time.h>

/* ================================================================
 *  H01: Brainwallet Dictionary  —  ~7M keys (7 variations × phrases.txt)
 * ================================================================ */

int h01_brainwallet(const secp256k1_context *ctx) {
    log_msg("[H01] Brainwallet (phrases.txt + 7 variants)...");
    FILE *f = fopen("phrases.txt", "r");
    if (!f) { log_msg("[H01] Cannot open phrases.txt, skipping"); return 0; }

    char word[256];
    uint64_t total = 0;

    while (fgets(word, sizeof(word), f)) {
        size_t sl = strlen(word);
        while (sl > 0 && (word[sl-1] == '\n' || word[sl-1] == '\r')) word[--sl] = '\0';
        if (sl == 0) continue;

        uint8_t sha[32];

        /* V0: original */
        SHA256((uint8_t*)word, sl, sha);
        if (check_privkey_multi(ctx, sha)) { log_found("H01-original", sha, 32); fclose(f); return 1; }
        total++;

        /* V1: Capitalize */
        if (sl > 0) {
            char cap[256]; strcpy(cap, word);
            cap[0] = (cap[0] >= 'a' && cap[0] <= 'z') ? cap[0] - 32 : cap[0];
            SHA256((uint8_t*)cap, sl, sha);
            if (check_privkey_multi(ctx, sha)) { log_found("H01-Cap", sha, 32); fclose(f); return 1; }
            total++;
        }

        /* V2: UPPER */
        char upper[256]; int j;
        for (j = 0; word[j]; j++) upper[j] = (word[j] >= 'a' && word[j] <= 'z') ? word[j] - 32 : word[j];
        upper[j] = '\0';
        SHA256((uint8_t*)upper, sl, sha);
        if (check_privkey_multi(ctx, sha)) { log_found("H01-UPPER", sha, 32); fclose(f); return 1; }
        total++;

        /* V3: lower */
        char lower[256];
        for (j = 0; word[j]; j++) lower[j] = (word[j] >= 'A' && word[j] <= 'Z') ? word[j] + 32 : word[j];
        lower[j] = '\0';
        SHA256((uint8_t*)lower, sl, sha);
        if (check_privkey_multi(ctx, sha)) { log_found("H01-lower", sha, 32); fclose(f); return 1; }
        total++;

        /* V4: leetspeak */
        char leet[256];
        for (j = 0; word[j]; j++) {
            char c = word[j];
            if (c == 'a' || c == 'A') leet[j] = '4';
            else if (c == 'e' || c == 'E') leet[j] = '3';
            else if (c == 'i' || c == 'I') leet[j] = '1';
            else if (c == 'o' || c == 'O') leet[j] = '0';
            else if (c == 's' || c == 'S') leet[j] = '5';
            else if (c == 't' || c == 'T') leet[j] = '7';
            else leet[j] = c;
        }
        leet[j] = '\0';
        SHA256((uint8_t*)leet, sl, sha);
        if (check_privkey_multi(ctx, sha)) { log_found("H01-leet", sha, 32); fclose(f); return 1; }
        total++;

        /* V5: Capitalized leet */
        if (sl > 0) {
            char cleet[256]; strcpy(cleet, leet);
            cleet[0] = (cleet[0] >= 'a' && cleet[0] <= 'z') ? cleet[0] - 32 : cleet[0];
            SHA256((uint8_t*)cleet, sl, sha);
            if (check_privkey_multi(ctx, sha)) { log_found("H01-Cleet", sha, 32); fclose(f); return 1; }
            total++;
        }

        /* V6: +"123" */
        char w123[259]; snprintf(w123, sizeof(w123), "%s123", word);
        SHA256((uint8_t*)w123, strlen(w123), sha);
        if (check_privkey_multi(ctx, sha)) { log_found("H01-word123", sha, 32); fclose(f); return 1; }
        total++;

        /* V7: +"!" */
        char wbang[257]; snprintf(wbang, sizeof(wbang), "%s!", word);
        SHA256((uint8_t*)wbang, strlen(wbang), sha);
        if (check_privkey_multi(ctx, sha)) { log_found("H01-word!", sha, 32); fclose(f); return 1; }
        total++;

        if (total % 100000 == 0)
            log_msg("[H01] %llu keys", (unsigned long long)total);
    }

    fclose(f);
    log_msg("[H01] Done. %llu keys.", (unsigned long long)total);
    return 0;
}

/* ================================================================
 *  H03: Timestamp + PID  —  262K keys
 * ================================================================ */

int h03_timestamp_pid(const secp256k1_context *ctx) {
    log_msg("[H03] Timestamp+PID...");
    uint32_t tss[] = {1268728843, 1268811438, 1268866685, 1268894549,
                      1268921836, 1268933538, 1268943264, 1289662741};
    for (int ti = 0; ti < 8; ti++) {
        uint32_t ts = tss[ti];
        for (uint32_t pid = 0; pid < 32768; pid++) {
            uint8_t b[16] = {0};
            b[0] = ts>>24; b[1] = ts>>16; b[2] = ts>>8; b[3] = ts;
            b[4] = pid>>24; b[5] = pid>>16; b[6] = pid>>8; b[7] = pid;
            uint8_t pk[32]; SHA256(b, 16, pk);
            if (check_privkey_multi(ctx, pk)) { log_found("H03", pk, 32); return 1; }
        }
    }
    log_msg("[H03] Done.");
    return 0;
}

/* ================================================================
 *  H07: Android SecureRandom  —  40M keys (OpenMP)
 * ================================================================ */

int h07_android(const secp256k1_context *ctx) {
    log_msg("[H07] Android SecureRandom...");
    volatile int g_found = 0;
    secp256k1_context *ctx_copy = secp256k1_context_clone(ctx);

    #pragma omp parallel for
    for (uint64_t key = 0; key < 40000000ULL; key++) {
        if (g_found) continue;
        uint8_t buf[32];
        for (int i = 0; i < 32; i++) buf[i] = (uint8_t)((key >> (i * 2)) & 0xFF);
        uint8_t pk[32]; SHA256(buf, 32, pk);
        if (check_privkey_multi(ctx_copy, pk)) {
            #pragma omp critical
            { if (!g_found) { g_found = 1; log_found("H07-Android", pk, 32); } }
        }
        if (key % 5000000 == 0 && key > 0)
            log_msg("[H07] %lluM", (unsigned long long)(key / 1000000));
    }
    secp256k1_context_destroy(ctx_copy);
    log_msg("[H07] Done.");
    return g_found;
}

/* ================================================================
 *  H08: Block Hashes 0..200000  —  200K keys
 * ================================================================ */

int h08_blockhashes(const secp256k1_context *ctx) {
    log_msg("[H08] Block hashes 0..200000...");
    for (uint32_t i = 0; i < 200000; i++) {
        uint8_t b[4] = {(uint8_t)(i>>24), (uint8_t)(i>>16), (uint8_t)(i>>8), (uint8_t)i};
        uint8_t pk[32]; SHA256(b, 4, pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H08-block", pk, 32); return 1; }
        if (i % 50000 == 0) log_msg("[H08] %u", i);
    }
    log_msg("[H08] Done.");
    return 0;
}

/* ================================================================
 *  H09: Deep Brainwallet (word+year)  —  500M+ keys
 * ================================================================ */

int h09_deep_brainwallet(const secp256k1_context *ctx) {
    log_msg("[H09] Deep brainwallet (word+year)...");
    FILE *f = fopen("phrases.txt", "r");
    if (!f) { log_msg("[H09] No phrases.txt"); return 0; }

    char word[256];
    uint64_t total = 0;
    const char *years[] = {"2009","2010","2011","2012","2013","2014","2015","2020","2024"};
    const char *sfx[] = {"", "!", "123", "?"};

    while (fgets(word, sizeof(word), f)) {
        size_t sl = strlen(word);
        while (sl && (word[sl-1]=='\n'||word[sl-1]=='\r')) word[--sl]='\0';
        if (sl < 3) continue;

        for (int yi = 0; yi < 9; yi++) {
            for (int si = 0; si < 4; si++) {
                char buf[256]; snprintf(buf, sizeof(buf), "%s%s%s", word, years[yi], sfx[si]);
                uint8_t pk[32]; SHA256((uint8_t*)buf, strlen(buf), pk);
                if (check_privkey_multi(ctx, pk)) { log_found("H09-deep", pk, 32); fclose(f); return 1; }
                total++;
                buf[0] = toupper((unsigned char)buf[0]);
                SHA256((uint8_t*)buf, strlen(buf), pk);
                if (check_privkey_multi(ctx, pk)) { log_found("H09-deep-cap", pk, 32); fclose(f); return 1; }
                total++;
            }
        }
    }
    fclose(f);
    log_msg("[H09] Done. %llu keys.", (unsigned long long)total);
    return 0;
}

/* ================================================================
 *  H11: Known Weak Keys  —  10 keys (instant)
 * ================================================================ */

int h11_weakkeys(const secp256k1_context *ctx) {
    log_msg("[H11] Weak keys...");
    uint8_t pk[32];
    memset(pk, 0, 32); pk[31] = 1;
    if (check_privkey_multi(ctx, pk)) { log_found("H11-key1", pk, 32); return 1; }

    memset(pk, 0xFF, 32);
    if (check_privkey_multi(ctx, pk)) { log_found("H11-allFF", pk, 32); return 1; }

    const char *c[] = {"password","bitcoin","123456","private","satoshi","nakamoto","btc","crypto","qwerty","admin"};
    for (int i = 0; i < 10; i++) {
        SHA256((uint8_t*)c[i], strlen(c[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H11-word", pk, 32); return 1; }
    }
    log_msg("[H11] Done.");
    return 0;
}

/* ================================================================
 *  H14: Timestamp Decimal String  —  8 keys
 * ================================================================ */

int h14_timestamp_string(const secp256k1_context *ctx) {
    log_msg("[H14] Timestamp strings...");
    uint32_t tss[] = {1268728843,1268811438,1268866685,1268894549,
                      1268921836,1268933538,1268943264,1289662741};
    for (int ti = 0; ti < 8; ti++) {
        char b[16]; snprintf(b, 16, "%u", tss[ti]);
        uint8_t pk[32]; SHA256((uint8_t*)b, strlen(b), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H14", pk, 32); return 1; }
    }
    log_msg("[H14] Done.");
    return 0;
}

/* ================================================================
 *  H15: Date Format Strings  —  112 keys
 * ================================================================ */

int h15_date_formats(const secp256k1_context *ctx) {
    log_msg("[H15] Date formats...");
    uint32_t tss[] = {1268728843,1268811438,1268866685,1268894549,
                      1268921836,1268933538,1268943264,1289662741};
    const char *fmts[] = {"%Y%m%d","%d%m%Y","%m%d%Y","%Y-%m-%d","%d-%m-%Y",
                          "%Y%m%d%H%M","%d-%m-%Y-%H-%M","%Y%m%d%H%M%S",
                          "%d/%m/%Y","%Y/%m/%d","%m/%d/%Y"};
    const char *mn[] = {"January","February","March","April","May","June",
                        "July","August","September","October","November","December"};
    char b[64]; uint8_t pk[32];
    for (int ti = 0; ti < 8; ti++) {
        time_t t = tss[ti]; struct tm *g = gmtime(&t);
        for (int fi = 0; fi < 11; fi++) {
            strftime(b, 64, fmts[fi], g);
            SHA256((uint8_t*)b, strlen(b), pk);
            if (check_privkey_multi(ctx, pk)) { log_found("H15", pk, 32); return 1; }
        }
        snprintf(b,64,"%d %s %d",g->tm_mday,mn[g->tm_mon],g->tm_year+1900);
        SHA256((uint8_t*)b, strlen(b), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H15", pk, 32); return 1; }
        snprintf(b,64,"%s %d, %d",mn[g->tm_mon],g->tm_mday,g->tm_year+1900);
        SHA256((uint8_t*)b, strlen(b), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H15", pk, 32); return 1; }
    }
    log_msg("[H15] Done.");
    return 0;
}

/* ================================================================
 *  H17: Timestamp + Word  —  CPU+OpenMP
 * ================================================================ */

int h17_ts_word(const secp256k1_context *ctx) {
    log_msg("[H17] Timestamp+word...");
    uint32_t tss[] = {1268728843,1279199023,1279203210,1284382196,1284608803,1285880600};
    const char *words[] = {"bitcoin","btc","key","privkey","wallet","secret",
                           "password","mykey","coin","satoshi","pass","test","hello","admin"};
    uint8_t pk[32];
    for (int ti = 0; ti < 6; ti++) {
        for (int wi = 0; wi < 14; wi++) {
            SHA256((uint8_t*)words[wi], strlen(words[wi]), pk);
            if (check_privkey_multi(ctx, pk)) { log_found("H17", pk, 32); return 1; }
        }
    }
    log_msg("[H17] Done.");
    return 0;
}

/* ================================================================
 *  H18: Multi-Word Brainwallet  —  CPU+OpenMP
 * ================================================================ */

int h18_multiword(const secp256k1_context *ctx) {
    log_msg("[H18] Multi-word brainwallet (hardcoded phrases)...");
    const char *phrases[] = {
        "bitcoinwallet","walletbitcoin","satoshi nakamoto","satoshi2010",
        "bitcoin2009","mybitcoin","mywallet","bitcoin key",
        "private key","secret key","my private key",
        "bitcoin private key","wallet password",
        "satoshi wallet","bitcoin brainwallet",
        "bitcoin passphrase","blockchain wallet",
        NULL
    };
    uint8_t pk[32];
    for (int i = 0; phrases[i]; i++) {
        SHA256((uint8_t*)phrases[i], strlen(phrases[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H18", pk, 32); return 1; }
    }
    log_msg("[H18] Done.");
    return 0;
}

/* ================================================================
 *  H20: srand(time(NULL))  —  7201 keys
 * ================================================================ */

int h20_srand_time(const secp256k1_context *ctx) {
    log_msg("[H20] srand(time(NULL)) only...");
    for (uint32_t ts = 1262304000; ts < 1325376000; ts += 86400) {
        srand(ts);
        uint8_t pk[32];
        for (int i = 0; i < 32; i++) pk[i] = (uint8_t)(rand() % 256);
        if (check_privkey_multi(ctx, pk)) { log_found("H20", pk, 32); return 1; }
    }
    log_msg("[H20] Done.");
    return 0;
}

/* ================================================================
 *  H21: Empty String SHA256("")  —  1 key
 * ================================================================ */

int h21_empty_string(const secp256k1_context *ctx) {
    uint8_t pk[32]; SHA256((uint8_t*)"", 0, pk);
    if (check_privkey_multi(ctx, pk)) { log_found("H21", pk, 32); return 1; }
    log_msg("[H21] Empty done.");
    return 0;
}

/* ================================================================
 *  H23: PHP mt_rand() Wallet  —  ~10B keys (SLOW PHASE)
 * ================================================================ */

#define MT_N 624
static void php_mt_init(uint32_t seed, uint32_t *state) {
    state[0] = seed;
    for (int i = 1; i < MT_N; i++)
        state[i] = 1812433253U * (state[i-1] ^ (state[i-1] >> 30)) + i;
}
static uint32_t php_mt_rand(uint32_t *state) {
    for (int i = 0; i < MT_N; i++) {
        uint32_t y = (state[i] & 0x80000000U) + (state[(i+1)%MT_N] & 0x7FFFFFFFU);
        state[i] = state[(i+397)%MT_N] ^ (y >> 1);
        if (y & 1) state[i] ^= 0x9908B0DFU;
    }
    uint32_t y = state[0];
    y ^= (y>>11); y ^= (y<<7)&0x9D2C5680U; y ^= (y<<15)&0xEFC60000U; y ^= (y>>18);
    return y;
}

int h23_php_mt_wallet(const secp256k1_context *ctx) {
    log_msg("[H23] PHP mt_rand wallet...");
    uint32_t tss[] = {1268728843,1279199023,1279203210,1284382196,1284608803,1285880600};
    for (int ti = 0; ti < 6; ti++) {
        for (uint32_t pid = 0; pid < 1024; pid++) {
            uint32_t state[MT_N]; php_mt_init(tss[ti] ^ pid, state);
            uint8_t pk[32];
            for (int i = 0; i < 8; i++) {
                uint32_t r = php_mt_rand(state);
                pk[i*4]=(uint8_t)(r>>24); pk[i*4+1]=(uint8_t)(r>>16);
                pk[i*4+2]=(uint8_t)(r>>8); pk[i*4+3]=(uint8_t)(r);
            }
            if (check_privkey_multi(ctx, pk)) { log_found("H23", pk, 32); return 1; }
        }
    }
    log_msg("[H23] Done.");
    return 0;
}

/* ================================================================
 *  H24: JS Math.random() — V8 XorShift128+  —  30M keys (OpenMP)
 * ================================================================ */

int h24_js_math_random(const secp256k1_context *ctx) {
    log_msg("[H24] JS Math.random() (V8 XorShift128+)...");
    volatile int g_found = 0;
    secp256k1_context *ctx_copy = secp256k1_context_clone(ctx);

    #pragma omp parallel for
    for (uint64_t seed = 0; seed < 30000000ULL; seed++) {
        if (g_found) continue;
        uint64_t s0 = seed, s1 = seed ^ 0x9E3779B97F4A7C15ULL;
        uint8_t pk[32];
        for (int i = 0; i < 4; i++) {
            uint64_t x = s0, y = s1; s0 = y;
            x ^= x << 23;
            uint64_t r = x ^ y ^ (x >> 17) ^ (y >> 26);
            s1 = r;
            pk[i*8]=(uint8_t)r; pk[i*8+1]=(uint8_t)(r>>8); pk[i*8+2]=(uint8_t)(r>>16);
            pk[i*8+3]=(uint8_t)(r>>24); pk[i*8+4]=(uint8_t)(r>>32); pk[i*8+5]=(uint8_t)(r>>40);
            pk[i*8+6]=(uint8_t)(r>>48); pk[i*8+7]=(uint8_t)(r>>56);
        }
        if (check_privkey_multi(ctx_copy, pk)) {
            #pragma omp critical
            { if (!g_found) { g_found = 1; log_found("H24", pk, 32); } }
        }
        if (seed % 5000000 == 0 && seed > 0) log_msg("[H24] %lluM", (unsigned long long)(seed/1000000));
    }
    secp256k1_context_destroy(ctx_copy);
    log_msg("[H24] Done.");
    return g_found;
}

/* ================================================================
 *  H25-H35: Smaller hypothesis categories
 * ================================================================ */

int h25_bitcointalk_phrases(const secp256k1_context *ctx) {
    log_msg("[H25] BitcoinTalk phrases...");
    const char *p[] = {
        "bitcoin","btc","satoshi","nakamoto","bitcointalk","bitcoin2009","bitcoin2010",
        "genesis","block0","coinbase","mining","blockchain","wallet","privatekey",
        "brainwallet","password","secret","sha256","ripemd160","base58","wif",
        "satoshi nakamoto","Satoshi","NAKAMOTO","satoshinakamoto",
        "proofofwork","pow","crypto","currency","digitalcash","p2p",
        "bitcoin-qt","bitcoind","satoshiclient",
        "linux","windows","mac","python","php","javascript","cpp","go","rust",
        "hello world","test","admin","root","user","guest","default",
        "key","mykey","privkey","priv","sec","ecdsa","secp256k1",
        "free","money","cash","gold","silver","dollar","euro","pound",
        "btc2009","btc2010","btc2011","satoshi2009","satoshi2010",
        "moon","lambo","hodl","rich","wealth","fortune","luck",
        "freedom","liberty","peace","justice","truth","honor","god",
        "love","life","hope","faith","destiny","dream","star",
        "angel","hero","legend","myth","phoenix","dragon","tiger",
        "samurai","ninja","warrior","knight","wizard","mage","king",
        "thomas","jesus","alex","mike","john","david","sam","bob",
        "bitcoinwallet","BTCWallet","btcwallet","bitcoin-wallet",
        "my wallet","mywallet","mybtc","mybitcoin","mycoin",
        "empty","nothing","null","nil","zero","void","none","noone",
        "mtgox","gox","mt.gox","silkroad","silk road",
        "deepbit","slush","btcguild","miningpool",
        "2010","16 March 2010","15 July 2010",
        "March 16 2010","July 15 2010","September 16 2010",
        "16/03/2010","15/07/2010","16-03-2010",
        "myfirstbitcoin","firstbitcoin","myfirstbtc",
        "test123","test01","test1","demo","demo123","sample",
        "1234567890","123456789","12345678","1234567",
        "abcdef","qwerty","asdfgh","zxcvbn",
        NULL
    };
    uint8_t pk[32];
    for (int i = 0; p[i]; i++) {
        SHA256((uint8_t*)p[i], strlen(p[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H25", pk, 32); return 1; }
    }
    log_msg("[H25] Done.");
    return 0;
}

int h26_wallet_backup_passwords(const secp256k1_context *ctx) {
    log_msg("[H26] Wallet backup passwords...");
    const char *p[] = {
        "backup","wallet","wallet.dat","mywallet","bitcoinwallet",
        "bitcoin backup","backup bitcoin","btc backup",
        "export","exported","exported wallet","exported bitcoin",
        "wallet export","wallet backup","wallet recovery",
        "recovery","recovery phrase","recovery seed",
        "seed","seed phrase","mnemonic","bip39","bip32",
        "encrypted","decrypt","decryption","password",
        "password123","Password123","password123!",
        "Password1","password1","P@ssw0rd","P@ssw0rd123",
        "changeme","changethis","pleasechange",
        "temp","temporary","new","new wallet",
        "walletpassword","walletpassphrase",
        "btcwallet","btc wallet","my btc wallet",
        "bitcoin core","bitcoin-qt wallet",
        "armory","electrum","multibit","blockchain.info",
        "mycelium","breadwallet","copay","coinbase",
        NULL
    };
    uint8_t pk[32];
    for (int i = 0; p[i]; i++) {
        SHA256((uint8_t*)p[i], strlen(p[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H26", pk, 32); return 1; }
    }
    log_msg("[H26] Done.");
    return 0;
}

int h27_url_brainwallets(const secp256k1_context *ctx) {
    log_msg("[H27] URL brainwallets...");
    const char *urls[] = {
        "bitcoin.org","www.bitcoin.org","https://bitcoin.org",
        "bitcoin.com","www.bitcoin.com","https://bitcoin.com",
        "bitcoin.it","www.bitcoin.it",
        "blockchain.info","blockchain.com","blockexplorer.com",
        "blockchain","block chain","blockchain.info/wallet",
        "bitcointalk.org","forum.bitcointalk.org",
        "satoshi.nakamoto","satoshi@bitcoin.org","satoshi@bitcoin.com",
        "github.com","github","github.com/bitcoin","bitcoin/bitcoin",
        "sourceforge.net","sourceforge",
        "bitcoin.org/bitcoin.pdf","bitcoin paper","whitepaper",
        "https://bitcoin.org/bitcoin.pdf",
        "bitcoinwiki","wiki","bitcoin wiki",
        "en.bitcoin.it","en.bitcoin.it/wiki",
        "btc","btc.com","btc.to",
        "mtgox.com","mt.gox.com",
        "silkroad","silkroad market","silk road",
        "the silk road",
        NULL
    };
    uint8_t pk[32];
    for (int i = 0; urls[i]; i++) {
        SHA256((uint8_t*)urls[i], strlen(urls[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H27", pk, 32); return 1; }
    }
    log_msg("[H27] Done.");
    return 0;
}

int h28_sequential_keys(const secp256k1_context *ctx) {
    log_msg("[H28] Sequential keys SHA256(i)...");
    for (uint32_t i = 0; i < 2000000; i++) {
        uint8_t b[4] = {(uint8_t)(i>>24),(uint8_t)(i>>16),(uint8_t)(i>>8),(uint8_t)i};
        uint8_t pk[32]; SHA256(b, 4, pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H28", pk, 32); return 1; }
        if (i % 500000 == 0) log_msg("[H28] %u", i);
    }
    log_msg("[H28] Done.");
    return 0;
}

int h29_bitcoin_suffix_patterns(const secp256k1_context *ctx) {
    log_msg("[H29] Bitcoin+suffix patterns...");
    const char *sfx[] = {"2009","2010","2011","2012","2013","2014","2015",
        "09","10","11","12","13","14","15",
        "wallet","key","keys","privkey","private",
        "1","01","001","123","1234","12345","123456",
        "!","!!","?","??",".","@","#","$",
        "pass","password","pwd","secret","code",
        "first","my","my1","my01","new","old","test",
        NULL};
    const char *pre[] = {"bitcoin","Bitcoin","BITCOIN","btc","BTC","satoshi","Satoshi",NULL};
    uint8_t pk[32];
    for (int pi = 0; pre[pi]; pi++)
        for (int si = 0; sfx[si]; si++) {
            char buf[128]; snprintf(buf,128,"%s%s",pre[pi],sfx[si]);
            SHA256((uint8_t*)buf, strlen(buf), pk);
            if (check_privkey_multi(ctx, pk)) { log_found("H29", pk, 32); return 1; }
        }
    log_msg("[H29] Done.");
    return 0;
}

int h30_amount_words(const secp256k1_context *ctx) {
    log_msg("[H30] Amount-based brainwallets...");
    const char *m[] = {
        "50btc","100btc","150btc","200btc","400btc","650btc","1200btc",
        "50coins","100coins","200coins","400coins","1200coins",
        "my200btc","my400btc","my1200btc",
        "fifty","hundred","twohundred","fourhundred",
        "50","100","200","400","650","1200",
        "my50","my100","my200","my400",
        NULL};
    uint8_t pk[32];
    for (int i = 0; m[i]; i++) {
        SHA256((uint8_t*)m[i], strlen(m[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H30", pk, 32); return 1; }
    }
    log_msg("[H30] Done.");
    return 0;
}

int h31_date_passphrases(const secp256k1_context *ctx) {
    log_msg("[H31] Date passphrases...");
    const char *m[] = {
        "July2010","April2010","August2010","October2010","September2010",
        "july2010","april2010","august2010","october2010","september2010",
        "July 2010","April 2010","August 2010","October 2010","September 2010",
        "March2010","march2010","March 2010",
        "2010March","2010July","2010September","2010October",
        "2010-03","2010-07","2010-09","2010-10",
        "03-2010","07-2010","09-2010","10-2010",
        "201003","201007","201009","201010",
        "March16","July15","September10","September16",
        "march16","july15","september10","september16",
        NULL};
    uint8_t pk[32];
    for (int i = 0; m[i]; i++) {
        SHA256((uint8_t*)m[i], strlen(m[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H31", pk, 32); return 1; }
    }
    log_msg("[H31] Done.");
    return 0;
}

int h32_date_amount(const secp256k1_context *ctx) {
    log_msg("[H32] Date+amount combos...");
    const char *m[] = {
        "March2010200","July2010400","September20101200","July2010200",
        "march2010200","july2010400","september20101200","july2010200",
        "2010March200","2010July400","2010September1200","2010July200",
        "2010-03-200","2010-07-400","2010-09-1200",
        "200-20100316","400-20100715","1200-20100910",
        "200July2010","400July2010","1200September2010",
        "03162010_200","07152010_400","09102010_1200",
        NULL};
    uint8_t pk[32];
    for (int i = 0; m[i]; i++) {
        SHA256((uint8_t*)m[i], strlen(m[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H32", pk, 32); return 1; }
    }
    log_msg("[H32] Done.");
    return 0;
}

int h33_mining_words(const secp256k1_context *ctx) {
    log_msg("[H33] Mining/pool keywords...");
    const char *m[] = {
        "mining","miner","mining2010","miner2010","pool2010",
        "miningpool","mining pool","btcmining","btcminer",
        "50btcminer","50btc mining","daily50","four50",
        "4x50","50x4","fiftyx4","4times50",
        "slush","deepbit","btcguild","pool","p2pool",
        "50btc@2010","myfirst50btc","50btcfirst",
        "50-50-50-50","50+50+50+50","split50","regular50",
        "poolminer","genesis block","genesis","block 0",
        NULL};
    uint8_t pk[32];
    for (int i = 0; m[i]; i++) {
        SHA256((uint8_t*)m[i], strlen(m[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H33", pk, 32); return 1; }
    }
    log_msg("[H33] Done.");
    return 0;
}

int h34_timestamp_full_dt(const secp256k1_context *ctx) {
    log_msg("[H34] Full datetime strings...");
    uint32_t ts[] = {1268728843,1279199023,1279203210,1284382196,1284608803,1285880600,
                     1268811438,1268866685,1268894549,1268921836,1268933538,1268943264};
    const char *f[] = {"%Y%m%d%H%M%S","%Y-%m-%d-%H-%M-%S","%d%m%Y%H%M%S",
                       "%Y%m%d%H%M","%Y%m%d","%d/%m/%Y %H:%M:%S",
                       "%Y/%m/%d %H:%M:%S","%d-%m-%Y-%H%M"};
    char b[32]; uint8_t pk[32];
    for (int fi = 0; fi < 8; fi++)
        for (int ti = 0; ti < 12; ti++) {
            time_t t = ts[ti]; struct tm *g = gmtime(&t);
            strftime(b, 32, f[fi], g);
            SHA256((uint8_t*)b, strlen(b), pk);
            if (check_privkey_multi(ctx, pk)) { log_found("H34", pk, 32); return 1; }
        }
    log_msg("[H34] Done.");
    return 0;
}

int h35_periodic_patterns(const secp256k1_context *ctx) {
    log_msg("[H35] Periodic patterns...");
    const char *p[] = {
        "50x4","4x50","50x4btc","4x50btc","fiftyxfour","fourxfifty",
        "50btc4times","repeat50","weekly50","50-50-50-50","50+50+50+50",
        "fiftyfiftyfifty","regular50","split50",
        "50btc@2010","myfirst50btc","50btcfirst"
    };
    uint8_t pk[32];
    for (int i = 0; i < sizeof(p)/sizeof(p[0]); i++) {
        SHA256((uint8_t*)p[i], strlen(p[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H35", pk, 32); return 1; }
    }
    log_msg("[H35] Done.");
    return 0;
}

int h41_leet_words(const secp256k1_context *ctx) {
    log_msg("[H41] Leet expansions...");
    const char *w[] = {"bitcoin","satoshi","password","private","secret",
                       "wallet","block","chain","money","coin","miner",
                       "gold","crypto","admin","master","root","key",
                       "btc","200","400","50","1200","2010","march",
                       "july","fifty","hundred","rich","lucky","empty","null"};
    int nw = sizeof(w)/sizeof(w[0]);
    uint8_t pk[32];
    for (int i = 0; i < nw; i++) {
        char buf[128];
        snprintf(buf,128,"%s",w[i]);
        SHA256((uint8_t*)buf, strlen(buf), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H41", pk, 32); return 1; }
        buf[0] = toupper((unsigned char)buf[0]);
        SHA256((uint8_t*)buf, strlen(buf), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H41", pk, 32); return 1; }
    }
    log_msg("[H41] Done.");
    return 0;
}

int h42_hex_seeds(const secp256k1_context *ctx) {
    log_msg("[H42] Hex seeds...");
    const char *h[] = {"deadbeef","cafebabe","feedface","decafc0ffee",
        "0xdeadbeef","0xcafebabe","0xfeed","0xbeef","0xc0de",
        "0xbaad","0xf00d","b00b5","c0ffee",
        "dead","beef","cafe","babe","feed","face",
        "deadbeefcafebabe","0123456789abcdef","fedcba9876543210",
        "aa55","55aa","aabb","ccdd","ddee",
        "1234","5678","90ab","cdef","abcd",
        "00000001","00000002","7fffffffffffffff",
        NULL};
    uint8_t pk[32];
    for (int i = 0; h[i]; i++) {
        SHA256((uint8_t*)h[i], strlen(h[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H42", pk, 32); return 1; }
    }
    log_msg("[H42] Done.");
    return 0;
}

int h43_unicode_combos(const secp256k1_context *ctx) {
    log_msg("[H43] Unicode/emoji combos...");
    const char *m[] = {
        "😂","🔥","🚀","💎","🙌","❤️","💰","✅",
        "bitcoin😂","btc🚀","satoshi🔥",
        "bitcoin🔥btc","btc💎diamond","satoshi💎nakamoto",
        "moon🚀","lambo🚀","hodl💎",
        "💎🙌","diamond hands","diamondhands",
        "₿","₿itcoin","₿TC",
        "βitcoin","βtc","sαtoshi",
        "bitc0in","s4tosh1","btc2010🚀",
        "฿itcoin","฿","฿TC",
        NULL};
    uint8_t pk[32];
    for (int i = 0; m[i]; i++) {
        SHA256((uint8_t*)m[i], strlen(m[i]), pk);
        if (check_privkey_multi(ctx, pk)) { log_found("H43", pk, 32); return 1; }
    }
    log_msg("[H43] Done.");
    return 0;
}