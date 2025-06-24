# ==== Compiler & Flags ====
NVCC    = nvcc
CFLAGS  = -O2 -std=c++17          # ✅ Use C++17 for better features (GCC 11.4 fully supports it)
INCLUDES = -Iinclude

# ==== Files ====
TARGET = build/main
SRCS   = src/main.cu src/morton.cu src/utils.cu
OBJS   = $(SRCS:.cu=.o)

# ==== Default rule ====
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^

# ==== Clean ====
clean:
	rm -f $(TARGET) $(OBJS)
