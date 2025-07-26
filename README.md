# High-Performance Bin-Aware Stream Compaction of Morton-Ordered Data on CUDA GPUs

This repository explores how combining Morton-ordered data layout with several GPU stream-compaction techniques can shrink large 2-D wind-field point clouds and speed up downstream kernels. 
Each raw point stores **(x, y, wind direction, temperature)**. 
The pipeline:

1. **Morton sort** – reorder points along a Z-curve to maximize spatial locality.  
2. **Bin compaction** – group points into 2^k bins (k-bit Morton IDs) and keep only non-empty bins.  
3. **Stream-compaction kernels** – implementations include *naïve*, *shared memory*, *warp shuffle*, *bitmask*, and a *Thrust* reference.  
4. **Timing & speed-up report** – compare each GPU variant against a CPU baseline.



## Install

```bash
# Ubuntu / WSL2
sudo apt-get update
sudo apt-get install build-essential git \
                     nvidia-cuda-toolkit graphviz doxygen
make              # single-command build via the provided Makefile
```



## Requirements

C++17 compiler (gcc-10 or newer)

NVIDIA GPU with Compute Capability ≥ 7.0

CUDA Toolkit ≥ 11.4



## Usage

Run `./build/main -h` to see every flag. Key options are:

| Flag                                         | Meaning / Default                                           |
| -------------------------------------------- | ----------------------------------------------------------- |
| `-n <int>`                                   | number of points &mdash; **default 100**                    |
| `-r`                                         | use a random seed (otherwise a fixed seed is used)          |
| `-t`                                         | print timing: kernel-only and total wall-clock              |
| `-c`                                         | run **CPU only**                                            |
| `-g`                                         | run **GPU only**                                            |
| `--mode naive \| bin`                        | pipeline family &mdash; **default `naive`**                 |
| `--variant atomic \| partition`              | sub-variant for *bin* mode                                  |
| `--kernel shared \| warp \| bitmask \| auto` | GPU kernel style (partition mode only)                      |
| `-k <int>`                                   | Morton-ID bit-width (bin granularity) &mdash; **default 8** |
| `-h`, `--help`                               | show this help message                                      |

A mixed CPU + GPU run is the default when neither `-c` nor `-g` is supplied.



## Five quick examples

```bash
make clean && make -j

./build/main
# default: n = 100, CPU+GPU, naïve pipeline

./build/main -n 1000000 --mode bin --variant partition \
             --kernel shared -k 10 -g
# 1 M points, bin-partition pipeline, shared-memory kernel, GPU only

./build/main -n 10000000 --mode naive -c
# 10 M points, naïve pipeline, CPU baseline only

./build/main -n 5000000 --mode bin --variant atomic \
             --kernel warp -k 8 -g
# 5 M points, bin-atomic pipeline, warp-shuffle kernel, GPU only

./build/main -n 2000000 --mode bin --variant partition \
             --kernel auto -k 12
# 2 M points, bin-partition pipeline, automatically chooses fastest kernel
```


## Documentation

This project uses [Doxygen](https://www.doxygen.nl/) to generate browsable HTML documentation.

📄 Generate docs locally

```bash
# Step 1 – make sure Doxygen is installed
sudo apt-get install doxygen graphviz

# Step 2 – from project root
doxygen Doxyfile
```

Output will be in `docs/html/index.html`. You can open it with:

```bash
xdg-open docs/html/index.html       # on Linux
firefox docs/html/index.html        # or use your browser
```