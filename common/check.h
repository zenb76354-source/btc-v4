#ifndef CHECK_H
#define CHECK_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <secp256k1.h>
#include "targets.h"

/* ================================================================
 *  LOGGING
 * ================================================================ */

static FILE *g_log = NULL;
static uint64_t grand_total_keys = 0;

static void log_init(void) {
    g_log = fopen("recovery.log", "a");
    if (g_log) {
        time_t t = time(NULL);
        fprintf(g_log, "\n===== RECOVERY START %s", ctime(&t));
        fflush(g_log);
    }
}

static void log_msg(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    printf("\n");
    fflush(stdout);
    if (g_log) {
        va_end(ap);
        va_start(ap, fmt);
        vfprintf(g_log, fmt, ap);
        fprintf(g_log, "\n");
        fflush(g_log);
    }
    va_end(ap);
}

static void log_found(const char *hyp, const uint8_t *key, uint32_t n) {
    log_msg("=== FOUND! %s ===", hyp);
    printf("privkey: ");
    for (uint32_t i = 0; i < n; i++) printf("%02x", key[i]);
    printf("\n");
    fflush(stdout);
    if (g_log) {
        fprintf(g_log, "privkey: ");
        for (uint32_t i = 0; i < n; i++) fprintf(g_log, "%02x", key[i]);
        fprintf(g_log, "\n");
        fflush(g_log);
    }
}

/* ================================================================
 *  CHECK_PRIVKEY_MULTI — against all 8 targets simultaneously
 *  Returns target index (1..8) if found, 0 otherwise
 *  When found, logs target address + private key
 * ================================================================ */

static int check_privkey_multi(const secp256k1_context *ctx, const uint8_t pk[32]) {
    secp256k1_pubkey pub;
    if (!secp256k1_ec_pubkey_create(ctx, &pub, pk)) return 0;

    uint8_t ser[65], sha[32], rmd[20];
    size_t l;

    /* Uncompressed (0x04 + X + Y) */
    l = 65;
    secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_UNCOMPRESSED);
    SHA256(ser, l, sha);
    RIPEMD160(sha, 32, rmd);

    for (int t = 0; t < NUM_TARGETS; t++) {
        if (!memcmp(rmd, TARGET_H160[t], 20)) {
            log_msg("=== FOUND! Target %d: %s (uncompressed) ===", t + 1, TARGET_ADDRS[t]);
            printf("privkey: ");
            for (int i = 0; i < 32; i++) printf("%02x", pk[i]);
            printf("\n");
            fflush(stdout);
            if (g_log) {
                fprintf(g_log, "privkey: ");
                for (int i = 0; i < 32; i++) fprintf(g_log, "%02x", pk[i]);
                fprintf(g_log, "\n");
                fflush(g_log);
            }
            return t + 1;
        }
    }

    /* Compressed (0x02/0x03 + X) */
    l = 33;
    secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_COMPRESSED);
    SHA256(ser, l, sha);
    RIPEMD160(sha, 32, rmd);

    for (int t = 0; t < NUM_TARGETS; t++) {
        if (!memcmp(rmd, TARGET_H160[t], 20)) {
            log_msg("=== FOUND! Target %d: %s (compressed) ===", t + 1, TARGET_ADDRS[t]);
            printf("privkey: ");
            for (int i = 0; i < 32; i++) printf("%02x", pk[i]);
            printf("\n");
            fflush(stdout);
            if (g_log) {
                fprintf(g_log, "privkey: ");
                for (int i = 0; i < 32; i++) fprintf(g_log, "%02x", pk[i]);
                fprintf(g_log, "\n");
                fflush(g_log);
            }
            return t + 1;
        }
    }

    return 0;
}

#endif /* CHECK_H */
