CC = nvcc
CFLAGS = -O2 -arch=sm_100 -std=c++11
# LDFLAGS = -lsecp256k1 -lssl -lcrypto  # No longer needed — all ECC is on GPU now
LDFLAGS =
TARGET = btc-recovery

all: $(TARGET)

$(TARGET): main.cu
	$(CC) $(CFLAGS) main.cu $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o recovery.log found_key.txt

run: $(TARGET)
	export LD_LIBRARY_PATH=/usr/local/lib:$$LD_LIBRARY_PATH && ./$(TARGET)

.PHONY: all clean run
