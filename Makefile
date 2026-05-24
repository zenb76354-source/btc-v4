CC = nvcc
CFLAGS = -O2 -arch=sm_100 -std=c++11 -rdc=true
LDFLAGS =
TARGET = btc-recovery

# Files
KERNEL_OBJ = gpu/kernels_code.o
MAIN_OBJ = main.o
DLINK_OBJ = dlink.o

all: $(TARGET)

# Step 1: Compile kernels code (device)
$(KERNEL_OBJ): gpu/kernels_code.cu gpu/kernels.cuh
	$(CC) $(CFLAGS) -c $< -o $@


# Step 2: Compile main (host)
$(MAIN_OBJ): main.cu gpu/kernels.cuh
	$(CC) $(CFLAGS) -c $< -o $@

# Step 3: Device link
$(DLINK_OBJ): $(KERNEL_OBJ) $(MAIN_OBJ)
	$(CC) $(CFLAGS) -dlink $(KERNEL_OBJ) $(MAIN_OBJ) -o $@

# Step 4: Final link
$(TARGET): $(KERNEL_OBJ) $(MAIN_OBJ) $(DLINK_OBJ)
	$(CC) $(CFLAGS) $(KERNEL_OBJ) $(MAIN_OBJ) $(DLINK_OBJ) $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET) *.o gpu/*.o recovery.log found_key.txt

run: $(TARGET)
	export LD_LIBRARY_PATH=/usr/local/lib:$$LD_LIBRARY_PATH && ./$(TARGET)

.PHONY: all clean run
