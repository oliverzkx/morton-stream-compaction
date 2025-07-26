/*
  stream_compaction_bin.cu
  ---------------------------------------------------------------
  Bin-based stream compaction using Morton-code binning.
  Each bin is identified by the lower k bits of the Morton code.
  We build (binOffsets, binSizes) and compact every bin separately
  to improve shared-memory locality.

  Author : Kaixiang Zou
  Date   : 2025-07-14
*/

#include "stream_compaction_bin.h"
#include "stream_compaction.h"   // reuse compactNaiveGPU / compactSharedGPU
#include "benchmark_utils.h"
#include "bin_kernel.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>       // for thrust::lower_bound / upper_bound
#include <thrust/adjacent_difference.h> // for thrust::adjacent_difference


#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include <numeric>     // <-- add this

extern float d_threshold; 

/* -------------------------------------------------------------------------- */
/*  computeBinOffsets                                                         */
/* -------------------------------------------------------------------------- */
void computeBinOffsets(const uint32_t* d_codes,
                       int             N,
                       int             kBits,
                       int*            d_binOffsets,
                       int*            d_binSizes)
{
    /* 把 Morton 码拷到 host，做一次线性扫描 */
    std::vector<uint32_t> h_codes(N);
    cudaMemcpy(h_codes.data(), d_codes, N * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    const int numBins  = 1 << kBits;
    std::vector<int> h_offsets(numBins + 1, 0);
    std::vector<int> h_sizes  (numBins,     0);

    int curBin = (h_codes.empty() ? 0 :
                  (h_codes[0] & ((1 << kBits) - 1)));

    for (int i = 0; i < N; ++i) {  
        int binID = h_codes[i] & ((1 << kBits) - 1);
        if (binID != curBin) {
            /* 记录上一个 bin 的终点 / 大小 */
            h_offsets[curBin + 1] = i;
            h_sizes[curBin]       = i - h_offsets[curBin];
            /* 填充可能空缺的 bin（如果数据稀疏） */
            for (int b = curBin + 1; b < binID; ++b) {
                h_offsets[b + 1] = i;
                h_sizes[b]       = 0;
            }
            curBin = binID;
            h_offsets[curBin] = i;   // 起点
        }
    }
    /* 最后一个 bin */
    h_offsets[numBins]           = N;
    h_sizes[curBin]              = N - h_offsets[curBin];

    /* 拷回 device */
    cudaMemcpy(d_binOffsets, h_offsets.data(),
               (numBins + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binSizes,   h_sizes.data(),
               numBins      * sizeof(int), cudaMemcpyHostToDevice);
}


/* -------------------------------------------------------------------------- */
/*  compactWithBinsGPU  – compact each bin on the GPU                         */
/* -------------------------------------------------------------------------- */
void compactWithBinsGPU(const Point2D*  d_in,
                        Point2D*        d_out,
                        const uint32_t* mortonCodes,
                        int             numPoints,
                        int             kBits,
                        int*            d_outCount)
{
    const int numBins = 1 << kBits;

    /* --- build bin metadata on device --- */
    thrust::device_vector<int> d_offsets(numBins + 1);
    thrust::device_vector<int> d_sizes  (numBins);

    computeBinOffsets(mortonCodes, numPoints, kBits,
                      thrust::raw_pointer_cast(d_offsets.data()),
                      thrust::raw_pointer_cast(d_sizes.data()));

    /* --- copy to host for a simple loop --- */
    std::vector<int> h_offsets(numBins + 1);
    std::vector<int> h_sizes  (numBins);
    thrust::copy(d_offsets.begin(), d_offsets.end(), h_offsets.begin());
    thrust::copy(d_sizes.begin(),   d_sizes.end(),   h_sizes.begin());

    int totalCompacted = 0;

    for (int bin = 0; bin < numBins; ++bin) {
        int offsetIn = h_offsets[bin];
        int sizeIn   = h_sizes[bin];
        if (sizeIn == 0) continue;            // skip empty bin

        const Point2D* binIn  = d_in  + offsetIn;
        Point2D*       binOut = d_out + totalCompacted;

        int h_count = 0;                      // host-side counter
        compactNaiveGPU(binIn, binOut, sizeIn, h_count);

        totalCompacted += h_count;
    }

    /* --- final total back to device --- */
    cudaMemcpy(d_outCount, &totalCompacted,
               sizeof(int), cudaMemcpyHostToDevice);
}



/* -------------------------------------------------------------------------- */
/*  runBitmaskBenchmarkWithBins                                               */
/* -------------------------------------------------------------------------- */
void runBitmaskBenchmarkWithBins(int               size,
                                 int               blockSize,
                                 const std::string precision,
                                 float&            time_ms,
                                 float&            error)
{
    // Placeholder: integrate with your existing benchmark utilities.
    // Steps you’ll likely need:
    //   1. Allocate / generate input data (points + Morton codes)
    //   2. cudaEvent_t start/stop around compactWithBinsGPU()
    //   3. Compute error vs. CPU reference if desired
    //   4. Release resources
    std::cout << "[bin-mode] benchmark stub (size = " << size
              << ", block = " << blockSize
              << ", precision = " << precision << ")\n";
    time_ms = 0.0f;
    error   = 0.0f;
}

/* -------------------------------------------------------------------------- */
/*  host wrapper similar to testNaiveGPUCompaction                            */
/* -------------------------------------------------------------------------- */
void testBinGPUCompaction(const std::vector<Point2D>& input,
                          float                       threshold,
                          int                         kBits,
                          std::vector<Point2D>&       output) {
    const int N = static_cast<int>(input.size());

    // 1. allocate and copy input
    Point2D* d_in  = nullptr;
    Point2D* d_out = nullptr;
    uint32_t* d_codes = nullptr;         // Morton codes on device
    int* d_outCount   = nullptr;

    cudaMalloc(&d_in,  N * sizeof(Point2D));
    cudaMalloc(&d_out, N * sizeof(Point2D));
    cudaMalloc(&d_codes, N * sizeof(uint32_t));
    cudaMalloc(&d_outCount, sizeof(int));

    cudaMemcpy(d_in, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);

    // 1.1 prepare Morton codes (host → compute & copy，或在 device 上 kernel 计算)
    std::vector<uint32_t> codes(N);
    for (int i = 0; i < N; ++i) codes[i] = morton2D_encode((int)input[i].x, (int)input[i].y);
    cudaMemcpy(d_codes, codes.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // 2. call bin compaction
    compactWithBinsGPU(d_in, d_out, d_codes, N, kBits, d_outCount);

    // 3. copy back result
    int h_outCount = 0;
    cudaMemcpy(&h_outCount, d_outCount, sizeof(int), cudaMemcpyDeviceToHost);

    output.resize(h_outCount);
    cudaMemcpy(output.data(), d_out, h_outCount * sizeof(Point2D), cudaMemcpyDeviceToHost);

    // 4. free
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_codes);
    cudaFree(d_outCount);
}


/* -------------------------------------------------------------------------- */
/*  compactBinAtomic  ——  one-pass compaction with atomics                    */
/* -------------------------------------------------------------------------- */
// __global__ void compactBinAtomic(const Point2D* __restrict__ in,
//                                  Point2D*       __restrict__ out,
//                                  const uint32_t*__restrict__ codes,
//                                  int*           binCursor,
//                                  int            N,
//                                  int            mask,
//                                  float          threshold)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= N) return;

//     Point2D p = in[idx];
//     if (p.temp <= threshold) return;            // filter condition

//     int binId = codes[idx] & mask;              // low-k bits
//     int pos   = atomicAdd(&binCursor[binId], 1);
//     out[pos]  = p;                              // write directly
// }

// __global__ void compactBinAtomic(const Point2D*  in,
//                                  Point2D*        out,
//                                  int*            binCursor,
//                                  const int*      binOffsets,
//                                  const uint32_t* codes,
//                                  int             N,
//                                  int             mask,
//                                  float           thr)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= N) return;

//     Point2D p = in[idx];
//     if (p.temp > thr) {
//         int bin       = codes[idx] & mask;
//         int posInBin  = atomicAdd(&binCursor[bin], 1);
//         int globalPos = binOffsets[bin] + posInBin;
//         out[globalPos] = p;
//     }
// }

/* ---------------------------------------------------------------------------
   compactBinAtomic  –  Plan-B, one-pass atomic compaction
   把满足谓词的点直接写到输出，全局递增计数器保证写入唯一
--------------------------------------------------------------------------- */
__global__ void compactBinAtomic(const Point2D*  in,
                                 Point2D*        out,
                                 int*            globalCnt,     // 唯一计数器
                                 const uint32_t* mortonCodes,
                                 int             N,
                                 int             mask,          // 低 k 位 (未用，可保留)
                                 float           thr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Point2D p = in[idx];
    if (p.temp > thr) {                         // 谓词：温度 > threshold
        int pos = atomicAdd(globalCnt, 1);      // 获得全局写位置
        out[pos] = p;                           // 写入
    }
}




// void testBinGPUCompaction_atomic(const std::vector<Point2D>& input,
//                                  float                       threshold,
//                                  int                         kBits,
//                                  std::vector<Point2D>&       output,
//                                  float&                      t_kernel_ms,
//                                  float&                      t_total_ms)
// {
//     const int N       = static_cast<int>(input.size());
//     const int numBins = 1 << kBits;
//     const int mask    = numBins - 1;

//     /* ---------- CUDA events ---------- */
//     cudaEvent_t t0, t1, k0, k1;
//     cudaEventCreate(&t0); cudaEventCreate(&t1);
//     cudaEventCreate(&k0); cudaEventCreate(&k1);
//     cudaEventRecord(t0);

//     /* ---------- raw buffers ---------- */
//     Point2D*  d_in  = nullptr;
//     Point2D*  d_out = nullptr;
//     uint32_t* d_codes = nullptr;
//     cudaMalloc(&d_in,  N * sizeof(Point2D));
//     cudaMalloc(&d_out, N * sizeof(Point2D));
//     cudaMalloc(&d_codes, N * sizeof(uint32_t));
//     cudaMemcpy(d_in, input.data(), N*sizeof(Point2D), cudaMemcpyHostToDevice);

//     /* ---------- device vectors for Plan-B ---------- */
//     thrust::device_vector<int> d_binSizes  (numBins,   0);
//     thrust::device_vector<int> d_binOffsets(numBins+1, 0);
//     thrust::device_vector<int> d_binCursor (numBins,   0);   // atomic counter

//     /* ---------- build Morton codes on host ---------- */
//     std::vector<uint32_t> h_codes(N);
//     for (int i = 0; i < N; ++i)
//         h_codes[i] = morton2D_encode(static_cast<int>(input[i].x),
//                                      static_cast<int>(input[i].y));
//     cudaMemcpy(d_codes, h_codes.data(), N*sizeof(uint32_t),
//                cudaMemcpyHostToDevice);

//     /* ---------- pass-1 histogram ---------- */
//     const int threads = 256;
//     const int blocks  = (N + threads - 1) / threads;
//     histogramBins<<<blocks,threads>>>(d_codes,
//         thrust::raw_pointer_cast(d_binSizes.data()),
//         N, mask);

//     /* ---------- exclusive scan → offsets ---------- */
//     thrust::exclusive_scan(d_binSizes.begin(), d_binSizes.end(),
//                            d_binOffsets.begin());
//     d_binOffsets[numBins] = N;   // last offset = N

//     /* ---------- kernel launch (atomic Plan-B) ------ */
//     cudaEventRecord(k0);
//     compactBinAtomic<<<blocks,threads>>>(
//         d_in, d_out,
//         thrust::raw_pointer_cast(d_binCursor.data()),
//         thrust::raw_pointer_cast(d_binOffsets.data()),
//         d_codes, N, mask, threshold);
//     cudaEventRecord(k1);
//     cudaEventSynchronize(k1);

//     /* ---------- gather counts & copy back ---------- */
//     std::vector<int> h_cursor(numBins);
//     cudaMemcpy(h_cursor.data(), thrust::raw_pointer_cast(d_binCursor.data()),
//                numBins*sizeof(int), cudaMemcpyDeviceToHost);
//     int total = std::accumulate(h_cursor.begin(), h_cursor.end(), 0);

//     output.resize(total);
//     cudaMemcpy(output.data(), d_out,
//                total*sizeof(Point2D), cudaMemcpyDeviceToHost);

//     cudaEventRecord(t1); cudaEventSynchronize(t1);
//     cudaEventElapsedTime(&t_kernel_ms, k0, k1);
//     cudaEventElapsedTime(&t_total_ms,  t0, t1);

//     /* ---------- cleanup ---------- */
//     cudaEventDestroy(k0); cudaEventDestroy(k1);
//     cudaEventDestroy(t0); cudaEventDestroy(t1);
//     cudaFree(d_in); cudaFree(d_out); cudaFree(d_codes);
// }

/* --------------------------------------------------------------------------
   testBinGPUCompaction_atomic  –  Plan-B host driver
   不再做直方图 / binOffsets，直接使用全局计数器
--------------------------------------------------------------------------- */
void testBinGPUCompaction_atomic(const std::vector<Point2D>& input,
                                 float                       threshold,
                                 int                         kBits,
                                 std::vector<Point2D>&       output,
                                 float&                      t_kernel_ms,
                                 float&                      t_total_ms)
{
    const int N = static_cast<int>(input.size());

    /* ---------- CUDA events ---------- */
    cudaEvent_t t0, t1, k0, k1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    cudaEventRecord(t0);

    /* ---------- allocate buffers ---------- */
    Point2D*  d_in  = nullptr;
    Point2D*  d_out = nullptr;
    uint32_t* d_codes = nullptr;
    int*      d_globalCnt = nullptr;          // 全局计数器

    cudaMalloc(&d_in,  N * sizeof(Point2D));
    cudaMalloc(&d_out, N * sizeof(Point2D));
    cudaMalloc(&d_codes, N * sizeof(uint32_t));
    cudaMalloc(&d_globalCnt, sizeof(int));
    cudaMemset(d_globalCnt, 0, sizeof(int));  // 计数器清零

    /* ---------- copy input points ---------- */
    cudaMemcpy(d_in, input.data(),
               N * sizeof(Point2D), cudaMemcpyHostToDevice);

    /* ---------- build Morton codes on host ---------- */
    std::vector<uint32_t> h_codes(N);
    for (int i = 0; i < N; ++i)
        h_codes[i] = morton2D_encode(
                        static_cast<int>(input[i].x),
                        static_cast<int>(input[i].y));
    cudaMemcpy(d_codes, h_codes.data(),
               N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* ---------- launch kernel ---------- */
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    cudaEventRecord(k0);

    compactBinAtomic<<<blocks, threads>>>(d_in, d_out,
                                          d_globalCnt,
                                          d_codes, N,
                                          /*mask*/ (1 << kBits) - 1,
                                          threshold);

    cudaEventRecord(k1);
    cudaEventSynchronize(k1);

    /* ---------- copy result count ---------- */
    int total = 0;
    cudaMemcpy(&total, d_globalCnt,
               sizeof(int), cudaMemcpyDeviceToHost);

    /* ---------- copy compacted points back ---------- */
    output.resize(total);
    if (total > 0) {
        cudaMemcpy(output.data(), d_out,
                   total * sizeof(Point2D), cudaMemcpyDeviceToHost);
    }

    /* ---------- timing ---------- */
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&t_kernel_ms, k0, k1);
    cudaEventElapsedTime(&t_total_ms,  t0, t1);

    /* ---------- cleanup ---------- */
    cudaEventDestroy(k0);  cudaEventDestroy(k1);
    cudaEventDestroy(t0);  cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_codes); cudaFree(d_globalCnt);
}



/* pass-1: build histogram (binSizes) */
__global__ void histogramBins(const uint32_t* codes,
                              int*           binSizes,
                              int            N,
                              int            mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int id = codes[idx] & mask;
    atomicAdd(&binSizes[id], 1);
}

/* pass-2: scatter points into tmp so each bin is contiguous */
__global__ void scatterToBins(const Point2D* __restrict__ in,
                              Point2D*       __restrict__ tmp,
                              const uint32_t*__restrict__ codes,
                              int*           binCursor,   // init = binOffsets
                              int            N,
                              int            mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int id  = codes[idx] & mask;
    int pos = atomicAdd(&binCursor[id], 1);   // unique slot in that bin
    tmp[pos] = in[idx];
}

/* -------------------------------------------------------------------------- */
/*  testBinGPUCompaction_partition  (Plan-A)                                  */
/* -------------------------------------------------------------------------- */
void testBinGPUCompaction_partition(const std::vector<Point2D>& input,
                                    float                       threshold,
                                    int                         kBits,
                                    std::vector<Point2D>&       output,
                                    float&                      t_kernel_ms,
                                    float&                      t_total_ms,
                                    BinKernel kernelKind)
{
    const int N       = static_cast<int>(input.size());
    const int numBins = 1 << kBits;
    const int mask    = numBins - 1;

    /* ---------- CUDA events ---------- */
    cudaEvent_t t0, t1, k0, k1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    cudaEventRecord(t0);

    /* ---------- allocate & H2D copy ---------- */
    Point2D*  d_in;      cudaMalloc(&d_in,  N * sizeof(Point2D));
    Point2D*  d_tmp;     cudaMalloc(&d_tmp, N * sizeof(Point2D));   // scatter buffer
    Point2D*  d_out;     cudaMalloc(&d_out, N * sizeof(Point2D));
    uint32_t* d_codes;   cudaMalloc(&d_codes, N * sizeof(uint32_t));

    cudaMemcpy(d_in, input.data(), N*sizeof(Point2D), cudaMemcpyHostToDevice);

    thrust::device_vector<int>  d_binSizes (numBins, 0);
    thrust::device_vector<int>  d_binOffsets(numBins+1, 0);

    /* ---------- prepare Morton codes ---------- */
    std::vector<uint32_t> h_codes(N);
    for (int i=0;i<N;++i)
        h_codes[i] = morton2D_encode((int)input[i].x, (int)input[i].y);
    cudaMemcpy(d_codes, h_codes.data(), N*sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* ---------- pass-1 histogram ---------- */
    int threads = 256, blocks = (N+threads-1)/threads;
    histogramBins<<<blocks,threads>>>(d_codes,
                                      thrust::raw_pointer_cast(d_binSizes.data()),
                                      N, mask);
    /* ---------- exclusive scan ---------- */
    thrust::exclusive_scan(d_binSizes.begin(), d_binSizes.end(),
                           d_binOffsets.begin());
    // 把最后一个 offset 设为 N
    d_binOffsets[numBins] = N;

    /* ---------- pass-2 scatter ---------- */
    // binCursor = binOffsets (拷贝一份)
    thrust::device_vector<int> d_binCursor = d_binOffsets;
    cudaEventRecord(k0);         // kernel timer start
    scatterToBins<<<blocks,threads>>>(d_in, d_tmp, d_codes,
                                      thrust::raw_pointer_cast(d_binCursor.data()),
                                      N, mask);
    cudaEventRecord(k1);         // kernel timer end
    cudaEventSynchronize(k1);

    /* ---------- copy offsets & sizes to host ---------- */
    std::vector<int> h_offsets(numBins+1);
    std::vector<int> h_sizes  (numBins);
    thrust::copy(d_binOffsets.begin(), d_binOffsets.end(), h_offsets.begin());
    thrust::copy(d_binSizes.begin(),   d_binSizes.end(),   h_sizes.begin());

    /* ---------- per-bin compaction (shared / warp) ---------- */
    int totalOut = 0;
    for (int b=0;b<numBins;++b) {
        int off = h_offsets[b];
        int sz  = h_sizes[b];
        if (sz==0) continue;

        Point2D* binIn  = d_tmp + off;
        Point2D* binOut = d_out + totalOut;
        int      h_cnt  = 0;
        // TODO: replace with your optimized kernel
        //compactNaiveGPU(binIn, binOut, sz, h_cnt, threshold);
        //compactNaiveGPU(binIn, binOut, sz, h_cnt);
        //compactSharedGPU(binIn, binOut, sz, threshold, h_cnt); 
        compactOneBin(binIn, binOut, sz, threshold, h_cnt, kernelKind);
        
        totalOut += h_cnt;
    }

    /* ---------- copy results back ---------- */
    output.resize(totalOut);
    cudaMemcpy(output.data(), d_out,
               totalOut*sizeof(Point2D), cudaMemcpyDeviceToHost);

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&t_kernel_ms, k0, k1);
    cudaEventElapsedTime(&t_total_ms,  t0, t1);

    /* ---------- cleanup ---------- */
    cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_codes);
    cudaEventDestroy(k0); cudaEventDestroy(k1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

// -----------------------------------------------------------------------------
//  compactWarpGPU – per-bin warp-shuffle stream compaction
//  每个 bin 使用 warp-shuffle 版本做压缩
// -----------------------------------------------------------------------------
void compactWarpGPU(const Point2D* d_in,
                    Point2D*       d_out,
                    int            N,
                    float          threshold,
                    int&           h_outCount)
{
    /* 0. Push predicate threshold to constant memory
       0. 把阈值写入 device constant memory */
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    /* 1. Device counter for warp kernel
       1. 为 warp kernel 分配设备端计数器 */
    int* d_cnt = nullptr;
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMemset(d_cnt, 0, sizeof(int));

    /* 2. Launch warp-shuffle compaction
       2. 启动 warp-shuffle 压缩 kernel */
    compact_points_warp(const_cast<Point2D*>(d_in), d_out, d_cnt, N);

    /* 3. Copy compacted count back
       3. 拷回压缩后元素个数 */
    cudaMemcpy(&h_outCount, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_cnt);
}


/* ---------------------------------------------------------------------------
   compactOneBin – unified per-bin launcher
   根据指定策略调用 Shared / Warp / Bitmask kernel
--------------------------------------------------------------------------- */
void compactOneBin(Point2D* d_in,      // bin input (contiguous)
                   Point2D* d_out,     // bin output base
                   int      N,         // elements in this bin
                   float    threshold, // predicate value
                   int&     h_outCnt,  // host-side result count
                   BinKernel kind)     // strategy
{
    /* ---------------- threshold 常量区 ---------------- */
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    /* ---------------- device counter ----------------- */
    int* d_cnt = nullptr;
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMemset(d_cnt, 0, sizeof(int));

    /* ---------------- strategy select ---------------- */
    if (kind == BinKernel::Auto)        // simple heuristic
        kind = (N <= 32)   ? BinKernel::Bitmask :
               (N <= 1024) ? BinKernel::Warp
                            : BinKernel::Shared;

    switch (kind) {
      case BinKernel::Shared:
        compactSharedGPU(d_in, d_out, N, threshold, h_outCnt);
        break;

      case BinKernel::Warp:
        // kernel expects device counter ptr
        compact_points_warp(d_in, d_out, d_cnt, N);
        cudaMemcpy(&h_outCnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        break;

      case BinKernel::Bitmask:
        compact_points_bitmask(d_in, d_out, d_cnt, N);
        cudaMemcpy(&h_outCnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        break;

      default: break;
    }
    cudaFree(d_cnt);
}