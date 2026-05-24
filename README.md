# BTC-Recovery v1.0

Bitcoin private key recovery through systematic weak-key hypothesis testing.

## Targets (8 addresses, ~$150M+ at peak)

| #  | Label | Address | Balance | First TX | Classification |
|:--:|:-----:|---------|:-------:|:--------:|---------------|
| 1  | A1    | `12rMpw5...` | 400 BTC | 2010-03-16 | Exchange deposit |
| 2  | A2*   | `1HLvaTs...` | 9260 BTC | **2020-12-19** | ❌ NOT 2010 |
| 3  | A3    | `1JA4Mpu...` | 400 BTC | 2010-07-15 | Exchange withdrawal |
| 4  | A4    | `13GvAdk...` | 200 BTC | 2010-07-15 | Mining/exchange |
| 5  | A5    | `1DTy9z4...` | 200 BTC | 2010-07-17 | Mining output (50×4) |
| 6  | A6    | `1MVLP2k...` | 1200 BTC | 2010-09-10 | Mining pool |
| 7  | A7    | `15QezNw...` | 200 BTC | 2010-09-16 | Mining output (50×4) |
| 8  | E1    | `198aMn6...` | 250 BTC | 2009 | Genesis era |

*A2 is a known dust collector active since 2020 — excluded from 2010 hypotheses.*

## Structure

```
btc-recovery/
├── Makefile              # Build targets: all, clean, run, deploy
├── README.md             # This file
├── common/
│   ├── targets.h         # All 8 targets (address, hash160, timestamps)
│   └── check.h           # check_privkey_multi + logging
├── cpu/
│   └── hypotheses.cu     # All 27 CPU hypothesis functions
├── gpu/
│   ├── kernels.cuh       # Device SHA256, RIPEMD160, secp256k1
│   └── timestamp_sweep.cu  # H36: GPU millisecond sweep (2009-2011)
└── main.cu               # Entry point — orchestrates phases
```

## Hypotheses

| ID | Name | Runner | Keys | Speed |
|:--:|------|:------:|:----:|:----:|
| H01 | Brainwallet Dictionary | CPU | ~7M | MED |
| H03 | Timestamp+PID | CPU+OMP | 262K | FAST |
| H07 | Android SecureRandom | CPU+OMP | 40M | MED |
| H08 | Block Hashes | CPU | 200K | FAST |
| H09 | Deep Brainwallet (word+year) | CPU | ~500M | MED |
| H11 | Known Weak Keys | CPU | 10 | ⚡ INSTANT |
| H14 | Timestamp Decimal String | CPU | 8 | ⚡ INSTANT |
| H15 | Date Format Strings | CPU | 112 | ⚡ INSTANT |
| H17 | Timestamp+Word | CPU+OMP | ~1M | SLOW |
| H18 | Multi-Word Brainwallet | CPU+OMP | ~500K | SLOW |
| H20 | srand(time(NULL)) | CPU | 7201 | FAST |
| H21 | Empty String SHA256("") | CPU | 1 | ⚡ INSTANT |
| H23 | PHP mt_rand() Wallet | CPU | ~10B | SLOW |
| H24 | JS Math.random() | CPU+OMP | 30M | MED |
| H25 | BitcoinTalk Phrases | CPU | ~5000 | FAST |
| H26 | Wallet Backup Passwords | CPU | ~300 | FAST |
| H27 | URL/Slug Brainwallets | CPU | ~300 | FAST |
| H28 | Sequential SHA256(i) | CPU | 2M | FAST |
| H29 | Bitcoin+Suffix Patterns | CPU | ~3000 | FAST |
| H30 | Amount-Based Brainwallets | CPU | ~40 | FAST |
| H31 | Date Passphrases | CPU | ~80 | FAST |
| H32 | Date+Amount Combos | CPU | ~30 | FAST |
| H33 | Mining/Pool Keywords | CPU | ~50 | FAST |
| H34 | Full Datetime Strings | CPU | ~96 | FAST |
| H35 | Periodic Pattern Matches | CPU | ~30 | FAST |
| **H36** | **GPU Timestamp ms Sweep** | **GPU** | **~95B** | ⚡~1s |
| H41 | Leet Word Expansions | CPU | ~60 | FAST |
| H42 | Hex Seeds | CPU | ~80 | FAST |
| H43 | Unicode/Emoji Combos | CPU | ~50 | FAST |

## Build

### On RunPod (RTX 5090)

```bash
# Clone fresh
git clone https://github.com/zenb76354-source/btc-v4 btc-recovery
cd btc-recovery

# Build
make

# Run
export LD_LIBRARY_PATH=/usr/local/lib && ./btc-recovery
```

### Other architectures

```bash
# RTX 4090
make arch89

# A100/A6000/RTX 3090
make arch80

# Custom
nvcc -O2 -arch=sm_XX -std=c++11 main.cu cpu/hypotheses.cu gpu/timestamp_sweep.cu \
     -lsecp256k1 -lssl -lcrypto -Xcompiler -fopenmp -o btc-recovery
```

## How H36 Works

Unlike the original buggy implementation (which multiplied seconds×1000 producing duplicates), H36 now:

1. **Millisecond precision**: `key = SHA256(ms_timestamp_be)` — one unique key per millisecond
2. **Focused range**: 2009-01-01 to 2012-01-01 (3 years, not 7)
3. **All targets at once**: GPU kernel compares against all 8 hash160s simultaneously
4. **~95B keys**: Completed in ~1 second on RTX 5090

## Adding New Hypotheses

1. Add function in `cpu/hypotheses.cu`
2. Declare in `main.cu`
3. Call it in the appropriate phase
4. Run `make deploy` to push

## License

Private project.
