#ifndef TARGETS_H
#define TARGETS_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* ================================================================
 *  TARGET ADDRESSES — version 1.0
 *  All from 2009-2010 era
 * ================================================================
 *  Index | Label | First TX          | Balance     | Classification
 *  ------+-------+-------------------+-------------+-----------------
 *   0    | A1    | 2010-03-16 04:40  |  400.00 BTC | Exchange deposit
 *   1    | A2    | 2020-12-19        | 9260.00 BTC | ❌ NOT 2010!
 *   2    | A3    | 2010-07-15 13:03  |  400.02 BTC | Exchange withdrawal
 *   3    | A4    | 2010-07-15 14:13  |  200.03 BTC | Mining/exchange
 *   4    | A5    | 2010-07-17 22:25  |  200.00 BTC | Mining output (50×4)
 *   5    | A6    | 2010-09-10 11:45  | 1200.65 BTC | Mining pool/exchange
 *   6    | A7    | 2010-09-16 01:46  |  200.00 BTC | Mining output (50×4)
 *   7    | E1    | 2009              |  250.00 BTC | Extra (198aMn6...)
 * ================================================================ */

#define NUM_TARGETS 8  /* 7 primary + 1 extra */

static const char *TARGET_ADDRS[NUM_TARGETS] = {
    "12rMpw5HnEvAw3nQqLmRBCQyuktfpa4eVw",  /* A1: 400 BTC, 2010-03-16 */
    "1HLvaTs3zR3oev9ya7Pzp3GB9Gqfg6XYJT",  /* A2: 9260 BTC, 2020-12-19 (IGNORE!) */
    "1JA4MpuV8MMNYCDTFHdCQeXGyem7mqo4B4",  /* A3: 400 BTC, 2010-07-15 */
    "13GvAdkFeHFGVxTHzcA2rD2e5BD4cGkbBH",  /* A4: 200 BTC, 2010-07-15 */
    "1DTy9z4JvtqYsg44oagVpHqyQpF7ZLLs45",  /* A5: 200 BTC, 2010-07-17 */
    "1MVLP2kRPNqz8VJUy83LstUoMQzUjgq4Zg",  /* A6: 1200 BTC, 2010-09-10 */
    "15QezNwA5ThiPf7wo89TTnfBwny93VQFTp",  /* A7: 200 BTC, 2010-09-16 */
    "198aMn6ZYAczwrE5NvNTUMyJ5qkfy4g3Hi",  /* E1: 250 BTC, 2009 */
};

/* Hash160 for each target (20 bytes) */
static const uint8_t TARGET_H160[NUM_TARGETS][20] = {
    /* 12rMpw5HnEvAw3nQqLmRBCQyuktfpa4eVw */
    {0x14,0x4d,0xe4,0x97,0x1a,0x30,0x9f,0x65,0x6a,0x25,
     0x98,0xf9,0x74,0x63,0xe2,0x1f,0xc4,0xe6,0x0f,0xe1},
    /* 1HLvaTs3zR3oev9ya7Pzp3GB9Gqfg6XYJT */
    {0xb3,0x46,0xa3,0xbc,0xe0,0xe6,0xf5,0xe8,0xd0,0x1b,
     0x6a,0x73,0x9c,0x05,0x01,0x49,0x2d,0xd5,0xf5,0xeb},
    /* 1JA4MpuV8MMNYCDTFHdCQeXGyem7mqo4B4 */
    {0xbc,0x30,0xaf,0x9c,0xfb,0xa5,0x5e,0xa6,0x13,0x74,
     0xf9,0x8b,0x3e,0xf3,0x18,0x55,0x70,0xb7,0x98,0x18},
    /* 13GvAdkFeHFGVxTHzcA2rD2e5BD4cGkbBH */
    {0x18,0xf2,0xdf,0x2f,0x55,0xe0,0xdd,0x03,0x98,0x2b,
     0x35,0x8b,0x5f,0xb7,0x49,0x1d,0x98,0xae,0x94,0xaf},
    /* 1DTy9z4JvtqYsg44oagVpHqyQpF7ZLLs45 */
    {0x88,0xbb,0x33,0x3d,0x5d,0xff,0xea,0x68,0x28,0xbd,
     0x86,0x8e,0x3a,0xe5,0x70,0x09,0x75,0xc8,0xfa,0x4c},
    /* 1MVLP2kRPNqz8VJUy83LstUoMQzUjgq4Zg */
    {0xe0,0xbe,0x57,0x0f,0x09,0x09,0xa4,0xee,0xdc,0x8e,
     0x82,0x65,0x2c,0x7f,0x39,0x10,0x38,0xf0,0x0c,0xcc},
    /* 15QezNwA5ThiPf7wo89TTnfBwny93VQFTp */
    {0x30,0x59,0xc8,0x38,0x4e,0x7e,0xbf,0x41,0xe0,0x3c,
     0x0d,0xa3,0xfa,0x7e,0x69,0xfa,0xb4,0x07,0x64,0x9d},
    /* 198aMn6ZYAczwrE5NvNTUMyJ5qkfy4g3Hi */
    {0x59,0x2f,0xc3,0x99,0x00,0x26,0x33,0x4c,0x8c,0x6f,
     0xb2,0xb9,0xda,0x45,0x71,0x79,0xcd,0xb5,0xc6,0x88},
};

/* Approximate timestamp of first transaction for each target */
static const uint32_t TARGET_TS[NUM_TARGETS] = {
    1268728843U,  /* 2010-03-16 04:40:43 */
    1268928656U,  /* 2010-03-18 (but real activity is 2020) */
    1279199023U,  /* 2010-07-15 13:03:43 */
    1279203210U,  /* 2010-07-15 14:13:30 */
    1285880600U,  /* 2010-09-30 ~22:23 */
    1284382196U,  /* 2010-09-13 16:49:56 */
    1284608803U,  /* 2010-09-16 01:46:43 */
    1234567890U,  /* 2009 (approximate) */
};

/* True/first_seen timestamps from blockchain (2010-03-16 etc) */
static const uint32_t TARGET_FIRST_SEEN[NUM_TARGETS] = {
    1268728843U,  /* 2010-03-16 04:40:43 */
    1608388624U,  /* 2020-12-19 — NOT 2010! */
    1279199023U,  /* 2010-07-15 13:03:43 */
    1279203210U,  /* 2010-07-15 14:13:30 */
    1279412745U,  /* 2010-07-17 22:25:45 */
    1284111956U,  /* 2010-09-10 11:45:56 */
    1284608803U,  /* 2010-09-16 01:46:43 */
    1234567890U,  /* 2009 (approximate) */
};

#endif /* TARGETS_H */
