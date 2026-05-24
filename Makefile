# ================================================================
#  BTC-RECOVERY Makefile
#
#  Targets:
#    all       — build everything
#    clean     — remove build artifacts
#    deploy    — push to GitHub
#    run       — build + run
#
#  Architecture notes:
#    RTX 5090: arch=sm_100
#    RTX 4090: arch=sm_89
#    A100/A6000/RTX 3090: arch=sm_80
#    RTX 2080: arch=sm_75
# ================================================================

ARCH    = sm_100
CC      = nvcc
CFLAGS  = -O2 -arch=$(ARCH) -std=c++11 -Xcompiler -fopenmp
LIBS    = -lsecp256k1 -lssl -lcrypto
SRCS    = main.cu cpu/hypotheses.cu gpu/timestamp_sweep.cu
TARGET  = btc-recovery
LOG     = recovery.log

.PHONY: all clean run deploy

all: $(TARGET)

$(TARGET): $(SRCS) common/targets.h common/check.h gpu/kernels.cuh
	$(CC) $(CFLAGS) $(SRCS) $(LIBS) -o $(TARGET)
	@echo ""
	@echo "=== Build complete: $(TARGET) ==="
	@echo "  Run: export LD_LIBRARY_PATH=/usr/local/lib && ./$(TARGET)"

clean:
	rm -f $(TARGET) $(LOG) *.o

run: $(TARGET)
	export LD_LIBRARY_PATH=/usr/local/lib && ./$(TARGET)

# For RTX 4090:
arch89:
	$(CC) -O2 -arch=sm_89 -std=c++11 $(SRCS) $(LIBS) -Xcompiler -fopenmp -o $(TARGET)

# For A100/RTX 3090:
arch80:
	$(CC) -O2 -arch=sm_80 -std=c++11 $(SRCS) $(LIBS) -Xcompiler -fopenmp -o $(TARGET)

# Push to GitHub
deploy:
	@echo "=== Deploying to GitHub ==="
	git add -A
	git commit -m "btc-recovery v1.0: clean structure, H36 fixed to milliseconds, 30 hypotheses"
	git push origin master

# Tail the log
log:
	tail -f $(LOG)
