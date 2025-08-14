/**
 * @file stream_compaction_bin.cu
 * @brief GPU stream-compaction with Morton-code partitioning (Plan A & Plan B).
 *
 * Pipeline overview
 * -----------------
 *  • Plan B (atomic): one-pass kernel writes directly with a global counter.
 *  • Plan A (partition):
 *      1. Histogram -> prefix-scan -> offsets
 *      2. Scatter points so each bin is contiguous
 *      3. Compact every bin with a selected ::BinKernel (shared / warp / bitmask)
 *
 * Host wrappers offer benchmark-ready entry points and timing statistics.
 *
 * @author  Kaixiang Zou
 * @version 1.3
 * @date    2025-07-26
 */

#include "stream_compaction_bin.h"
#include "stream_compaction.h"    // Naive / Shared helpers
#include "benchmark_utils.h"
#include "bin_kernel.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <numeric>

extern float d_threshold;   ///< device-side predicate threshold

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { \
  cudaError_t err__ = (x); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
  } \
} while(0)
#endif

// ────────────────────────────────────────────────────────────────
// computeBinOffsets
// ────────────────────────────────────────────────────────────────

/**
 * @brief Build exclusive offsets and per-bin sizes from Morton codes.
 *
 * The algorithm runs on the host for simplicity: copy codes → linear scan →
 * fill gaps for empty bins → copy results back to the device.
 *
 * @param d_codes       Device pointer to Morton codes.
 * @param N             Total number of elements.
 * @param kBits         Low-bit width used as the bin ID (numBins = 2^kBits).
 * @param d_binOffsets  Device array (numBins + 1) to receive start indices.
 * @param d_binSizes    Device array (numBins)     to receive element counts.
 */
void computeBinOffsets(const uint32_t* d_codes,
                       int             N,
                       int             kBits,
                       int*            d_binOffsets,
                       int*            d_binSizes)
{
    // Copy codes to host so we can use a simple CPU scan.
    std::vector<uint32_t> h_codes(N);
    cudaMemcpy(h_codes.data(), d_codes,
               N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    const int numBins = 1 << kBits;
    std::vector<int> h_offsets(numBins + 1, 0);  // inclusive start for each bin
    std::vector<int> h_sizes  (numBins,     0);

    // Current bin ID derived from the first Morton code
    int curBin = h_codes.empty() ? 0
                                 : (h_codes[0] & (numBins - 1));

    for (int i = 0; i < N; ++i) {
        int binID = h_codes[i] & (numBins - 1);
        if (binID != curBin) {
            // Close previous bin
            h_offsets[curBin + 1] = i;
            h_sizes  [curBin]     = i - h_offsets[curBin];

            // Fill any skipped empty bins
            for (int b = curBin + 1; b < binID; ++b) {
                h_offsets[b + 1] = i;
                h_sizes  [b]     = 0;
            }
            // Start new bin
            curBin            = binID;
            h_offsets[curBin] = i;
        }
    }
    // Final bin
    h_offsets[numBins] = N;
    h_sizes  [curBin]  = N - h_offsets[curBin];

    cudaMemcpy(d_binOffsets, h_offsets.data(),
               (numBins + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binSizes,   h_sizes.data(),
               numBins * sizeof(int),       cudaMemcpyHostToDevice);
}

// ────────────────────────────────────────────────────────────────
// compactWithBinsGPU  – reference implementation (per-bin Naive)
// ────────────────────────────────────────────────────────────────

/**
 * @brief Compact every bin on the GPU using the Naïve kernel.
 *
 * Intended for correctness measurements rather than peak performance.
 *
 * @param d_in         Device array of input points.
 * @param d_out        Device array to receive compacted points.
 * @param mortonCodes  Device array of Morton codes.
 * @param numPoints    Total input size.
 * @param kBits        Bin ID bit-width.
 * @param d_outCount   Device pointer receiving total valid count.
 */
void compactWithBinsGPU(const Point2D*  d_in,
                        Point2D*        d_out,
                        const uint32_t* mortonCodes,
                        int             numPoints,
                        int             kBits,
                        int*            d_outCount)
{
    const int numBins = 1 << kBits;

    // Allocate device vectors for offsets & sizes
    thrust::device_vector<int> d_offsets(numBins + 1);
    thrust::device_vector<int> d_sizes  (numBins);

    computeBinOffsets(mortonCodes, numPoints, kBits,
                      thrust::raw_pointer_cast(d_offsets.data()),
                      thrust::raw_pointer_cast(d_sizes.data()));

    // Copy metadata to host so we can iterate in a simple for-loop
    std::vector<int> h_offsets(numBins + 1);
    std::vector<int> h_sizes  (numBins);
    thrust::copy(d_offsets.begin(), d_offsets.end(), h_offsets.begin());
    thrust::copy(d_sizes.begin(),   d_sizes.end(),   h_sizes.begin());

    int totalCompacted = 0;

    for (int bin = 0; bin < numBins; ++bin) {
        int offsetIn = h_offsets[bin];
        int sizeIn   = h_sizes[bin];
        if (sizeIn == 0) continue;      // skip empty bins

        const Point2D* binIn  = d_in  + offsetIn;
        Point2D*       binOut = d_out + totalCompacted;

        int h_count = 0;
        compactNaiveGPU(binIn, binOut, sizeIn, h_count);
        totalCompacted += h_count;
    }
    cudaMemcpy(d_outCount, &totalCompacted,
               sizeof(int), cudaMemcpyHostToDevice);
}

// ────────────────────────────────────────────────────────────────
// runBitmaskBenchmarkWithBins  (stub for future work)
// ────────────────────────────────────────────────────────────────

/**
 * @brief Placeholder benchmark entry point (to be integrated later).
 */
void runBitmaskBenchmarkWithBins(int               size,
                                 int               blockSize,
                                 const std::string precision,
                                 float&            time_ms,
                                 float&            error)
{
    std::cout << "[bin-mode] benchmark stub (size=" << size
              << ", block="  << blockSize
              << ", precision=" << precision << ")\n";
    time_ms = 0.0f;
    error   = 0.0f;
}

// ────────────────────────────────────────────────────────────────
// testBinGPUCompaction  – naïve per-bin reference
// ────────────────────────────────────────────────────────────────

/**
 * @brief Host helper that uses ::compactWithBinsGPU for correctness checks.
 */
// void testBinGPUCompaction(const std::vector<Point2D>& input,
//                           float                       threshold,
//                           int                         kBits,
//                           std::vector<Point2D>&       output)
// {
//     (void)threshold;  // threshold is only used by kernels; Naïve path ignores it
//     const int N = static_cast<int>(input.size());

//     /* Allocate device buffers ------------------------------------------------*/
//     Point2D*  d_in        = nullptr;
//     Point2D*  d_out       = nullptr;
//     uint32_t* d_codes     = nullptr;
//     int*      d_outCount  = nullptr;

//     cudaMalloc(&d_in,  N * sizeof(Point2D));
//     cudaMalloc(&d_out, N * sizeof(Point2D));
//     cudaMalloc(&d_codes, N * sizeof(uint32_t));
//     cudaMalloc(&d_outCount, sizeof(int));

//     cudaMemcpy(d_in, input.data(),
//                N * sizeof(Point2D), cudaMemcpyHostToDevice);

//     /* Build Morton codes on host --------------------------------------------*/
//     std::vector<uint32_t> codes(N);
//     for (int i = 0; i < N; ++i)
//         codes[i] = morton2D_encode(static_cast<int>(input[i].x),
//                                    static_cast<int>(input[i].y));
//     cudaMemcpy(d_codes, codes.data(),
//                N * sizeof(uint32_t), cudaMemcpyHostToDevice);

//     /* Run compaction --------------------------------------------------------*/
//     compactWithBinsGPU(d_in, d_out, d_codes, N, kBits, d_outCount);

//     /* Copy results back -----------------------------------------------------*/
//     int h_outCount = 0;
//     cudaMemcpy(&h_outCount, d_outCount,
//                sizeof(int), cudaMemcpyDeviceToHost);

//     output.resize(h_outCount);
//     cudaMemcpy(output.data(), d_out,
//                h_outCount * sizeof(Point2D), cudaMemcpyDeviceToHost);

//     /* Cleanup ---------------------------------------------------------------*/
//     cudaFree(d_in); cudaFree(d_out);
//     cudaFree(d_codes); cudaFree(d_outCount);
// }

// ────────────────────────────────────────────────────────────────
// testBinGPUCompaction  – naïve per-bin reference  (TIMED)
// 计时版：同时统计 kernel-only 与 end-to-end（不含 alloc/free 与主机侧生成）
// ────────────────────────────────────────────────────────────────
/**
 * @brief Host helper that uses ::compactWithBinsGPU for correctness checks,
 *        now with timing.
 * @param ms_kernel  (out, optional) kernel-only time in milliseconds
 * @param ms_e2e     (out, optional) end-to-end time (H2D + kernels + D2H)
 *
 * Notes:
 * - End-to-end starts right before the first H2D and ends after copying back output.
 * - Morton code generation on host is intentionally excluded from E2E (dataset prep).
 * - All ops use the default stream (0) so events bracket the same timeline as kernels.
 */
void testBinGPUCompaction(const std::vector<Point2D>& input,
                          float                       threshold,
                          int                         kBits,
                          std::vector<Point2D>&       output,
                          float*                      ms_kernel /*= nullptr*/,
                          float*                      ms_e2e    /*= nullptr*/)
{
    (void)threshold;  // threshold is only used by kernels; Naïve path ignores it
    const int N = static_cast<int>(input.size());
    const size_t in_bytes   = N * sizeof(Point2D);
    const size_t code_bytes = N * sizeof(uint32_t);

    /* Allocate device buffers ------------------------------------------------*/
    Point2D*  d_in        = nullptr;
    Point2D*  d_out       = nullptr;
    uint32_t* d_codes     = nullptr;
    int*      d_outCount  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, in_bytes));       // out alloc N for worst-case
    CUDA_CHECK(cudaMalloc(&d_codes, code_bytes));
    CUDA_CHECK(cudaMalloc(&d_outCount, sizeof(int)));

    // Events (default stream 0) / 事件（默认流0）
    cudaEvent_t eStart, eStop, kStart, kStop, countReady;
    CUDA_CHECK(cudaEventCreate(&eStart));
    CUDA_CHECK(cudaEventCreate(&eStop));
    CUDA_CHECK(cudaEventCreate(&kStart));
    CUDA_CHECK(cudaEventCreate(&kStop));
    CUDA_CHECK(cudaEventCreate(&countReady));

    /* End-to-end begins: first H2D ------------------------------------------*/
    // E2E 计时从首次 H2D 开始（不含主机端构造 Morton codes）
    CUDA_CHECK(cudaEventRecord(eStart, 0));

    // H2D input (async on stream 0)
    CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(),
                               in_bytes, cudaMemcpyHostToDevice, 0));

    /* Build Morton codes on host (EXCLUDED from E2E) ------------------------*/
    // 主机侧生成 Morton codes（E2E 不计入）
    std::vector<uint32_t> codes(N);
    for (int i = 0; i < N; ++i)
        codes[i] = morton2D_encode(static_cast<int>(input[i].x),
                                   static_cast<int>(input[i].y));

    // H2D codes (async on stream 0)
    CUDA_CHECK(cudaMemcpyAsync(d_codes, codes.data(),
                               code_bytes, cudaMemcpyHostToDevice, 0));

    /* Kernel-only timing: bracket the GPU work ------------------------------*/
    // Kernel-only 计时：紧贴 GPU 工作的首尾
    CUDA_CHECK(cudaEventRecord(kStart, 0));
    // NOTE: compactWithBinsGPU should launch kernels on default stream (0).
    // 注意：假定 compactWithBinsGPU 在默认流0上发射 kernel。
    compactWithBinsGPU(d_in, d_out, d_codes, N, kBits, d_outCount);
    CUDA_CHECK(cudaEventRecord(kStop, 0));

    /* D2H: first fetch outCount, then fetch the compacted data --------------*/
    int h_outCount = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_outCount, d_outCount,
                               sizeof(int), cudaMemcpyDeviceToHost, 0));
    // Event after count is available / 计数可用后打点
    CUDA_CHECK(cudaEventRecord(countReady, 0));
    CUDA_CHECK(cudaEventSynchronize(countReady));   // ensure h_outCount ready

    output.resize(h_outCount);
    if (h_outCount > 0) {
        CUDA_CHECK(cudaMemcpyAsync(output.data(), d_out,
                                   h_outCount * sizeof(Point2D),
                                   cudaMemcpyDeviceToHost, 0));
    }

    // End-to-end stop AFTER enqueuing final D2H / 最终 D2H 入队后再打 eStop
    CUDA_CHECK(cudaEventRecord(eStop, 0));
    CUDA_CHECK(cudaEventSynchronize(eStop));  // wait for all queued GPU work

    /* Read timings ----------------------------------------------------------*/
    float tKernelMs = 0.f, tE2EMs = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&tKernelMs, kStart, kStop));
    CUDA_CHECK(cudaEventElapsedTime(&tE2EMs,    eStart, eStop));

    if (ms_kernel) *ms_kernel = tKernelMs;
    if (ms_e2e)    *ms_e2e    = tE2EMs;

    /* Cleanup ---------------------------------------------------------------*/
    cudaEventDestroy(eStart); cudaEventDestroy(eStop);
    cudaEventDestroy(kStart); cudaEventDestroy(kStop);
    cudaEventDestroy(countReady);
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_codes); cudaFree(d_outCount);
}


// ────────────────────────────────────────────────────────────────
// compactBinAtomic  – Plan B device kernel
// ────────────────────────────────────────────────────────────────

/**
 * @brief One-pass atomic compaction kernel (Plan B).
 *
 * Every thread that satisfies the predicate performs an atomicAdd on the
 * global counter and writes the element directly to the computed slot.
 *
 * @param in          Input points.
 * @param out         Output buffer.
 * @param globalCnt   Device counter for the next free slot.
 * @param mortonCodes Morton codes (unused, kept for symmetry).
 * @param N           Elements in the bin.
 * @param mask        Low-bit mask (unused in this kernel).
 * @param thr         Predicate threshold.
 */
__global__ void compactBinAtomic(const Point2D*  in,
                                 Point2D*        out,
                                 int*            globalCnt,
                                 const uint32_t* mortonCodes,
                                 int             N,
                                 int             mask,
                                 float           thr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const Point2D p = in[idx];
    if (p.temp > thr) {
        int pos = atomicAdd(globalCnt, 1);  // unique slot
        out[pos] = p;
    }
    (void)mortonCodes; (void)mask;          // silence unused-param warnings
}

// ────────────────────────────────────────────────────────────────
// testBinGPUCompaction_atomic  – Plan B host driver
// ────────────────────────────────────────────────────────────────

/**
 * @brief Host wrapper for the single-pass atomic kernel (Plan B).
 *
 * Measures both kernel-only and end-to-end timings using CUDA events.
 */
void testBinGPUCompaction_atomic(const std::vector<Point2D>& input,
                                 float                       threshold,
                                 int                         kBits,
                                 std::vector<Point2D>&       output,
                                 float&                      t_kernel_ms,
                                 float&                      t_total_ms)
{
    const int N = static_cast<int>(input.size());
    (void)kBits;  // mask is pre-computed but unused by this kernel

    /* Create CUDA events for timing ----------------------------------------*/
    cudaEvent_t t0, t1, k0, k1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    cudaEventRecord(t0);

    /* Allocate buffers -----------------------------------------------------*/
    Point2D*  d_in        = nullptr;
    Point2D*  d_out       = nullptr;
    uint32_t* d_codes     = nullptr;
    int*      d_globalCnt = nullptr;

    cudaMalloc(&d_in,  N * sizeof(Point2D));
    cudaMalloc(&d_out, N * sizeof(Point2D));
    cudaMalloc(&d_codes, N * sizeof(uint32_t));
    cudaMalloc(&d_globalCnt, sizeof(int));
    cudaMemset(d_globalCnt, 0, sizeof(int));

    cudaMemcpy(d_in, input.data(),
               N * sizeof(Point2D), cudaMemcpyHostToDevice);

    /* Build Morton codes on host ------------------------------------------*/
    std::vector<uint32_t> h_codes(N);
    for (int i = 0; i < N; ++i)
        h_codes[i] = morton2D_encode(static_cast<int>(input[i].x),
                                     static_cast<int>(input[i].y));
    cudaMemcpy(d_codes, h_codes.data(),
               N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Launch kernel --------------------------------------------------------*/
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    cudaEventRecord(k0);

    compactBinAtomic<<<blocks, threads>>>(d_in, d_out, d_globalCnt,
                                          d_codes, N, 0, threshold);

    cudaEventRecord(k1);
    cudaEventSynchronize(k1);

    /* Copy result count & elements back ------------------------------------*/
    int total = 0;
    cudaMemcpy(&total, d_globalCnt,
               sizeof(int), cudaMemcpyDeviceToHost);

    output.resize(total);
    if (total > 0)
        cudaMemcpy(output.data(), d_out,
                   total * sizeof(Point2D), cudaMemcpyDeviceToHost);

    /* Timings --------------------------------------------------------------*/
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&t_kernel_ms, k0, k1);
    cudaEventElapsedTime(&t_total_ms,  t0, t1);

    /* Cleanup --------------------------------------------------------------*/
    cudaEventDestroy(k0); cudaEventDestroy(k1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_codes); cudaFree(d_globalCnt);
}

// ────────────────────────────────────────────────────────────────
// Device utility kernels (Plan A)
// ────────────────────────────────────────────────────────────────

/**
 * @brief Pass 1: build a per-bin histogram of element counts.
 *
 * @param codes     Morton codes.
 * @param binSizes  Global histogram array (initialised to zero).
 * @param N         Total elements.
 * @param mask      (1 << kBits)-1 — extracts the bin ID.
 */
__global__ void histogramBins(const uint32_t* codes,
                              int*            binSizes,
                              int             N,
                              int             mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int id = codes[idx] & mask;
    atomicAdd(&binSizes[id], 1);
}


// (A) 为每个输出位置 dst 记录来源下标 src：srcIndexForDest[dst] = src
__global__ void buildDestMapKernel(
    const uint32_t* __restrict__ d_codes,       // [N] morton codes
    const int*      __restrict__ d_binOffsets,  // [numBins] exclusive scan
    int*            __restrict__ d_binCursor,   // [numBins] 运行时计数器(launch前清零)
    int*            __restrict__ d_srcIndexForDest, // [N]
    int N, int mask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int b = static_cast<int>(d_codes[i] & mask);
    int dst = atomicAdd(&d_binCursor[b], 1) + d_binOffsets[b];
    d_srcIndexForDest[dst] = i;   // 随机写，但只有4字节
}

// // 共享内存不够时的退回实现：每元素一次 atomicAdd（6参版本）
// __global__ void buildDestMapKernel_naive(
//     const uint32_t* __restrict__ d_codes,        // [N]
//     const int*      __restrict__ d_binOffsets,   // [numBins]
//     int*            __restrict__ d_binCursor,    // [numBins]
//     int*            __restrict__ d_srcIndexForDest, // [N]
//     int N, int mask)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;
//     int b   = static_cast<int>(d_codes[i] & mask);
//     int dst = atomicAdd(&d_binCursor[b], 1) + d_binOffsets[b];
//     d_srcIndexForDest[dst] = i;   // 随机写4B索引
// }

// __global__ void buildDestMapKernel(
//     const uint32_t* __restrict__ d_codes,        // [N] morton codes
//     const int*      __restrict__ d_binOffsets,   // [numBins] exclusive scan
//     int*            __restrict__ d_binCursor,    // [numBins] global running counters (0-inited)
//     int*            __restrict__ d_srcIndexForDest, // [N] out: dst -> src
//     int N, int mask, int numBins)
// {
//     extern __shared__ int smem[];               // 动态共享内存
//     int* shHist    = smem;                      // [numBins]
//     int* shBase    = shHist + numBins;          // [numBins]
//     int* shCounter = shBase + numBins;          // [numBins]

//     // --- 0) 清零共享内存直方图
//     for (int b = threadIdx.x; b < numBins; b += blockDim.x) {
//         shHist[b] = 0;
//     }
//     __syncthreads();

//     // --- 1) 本块统计：对本块覆盖的元素做 256-bin 直方图（共享内存原子）
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) {
//         int b = static_cast<int>(d_codes[i] & mask);
//         atomicAdd(&shHist[b], 1);
//     }
//     __syncthreads();

//     // --- 2) 为本块每个出现的 bin 预留全局段（每 bin 一次全局 atomic）
//     for (int b = threadIdx.x; b < numBins; b += blockDim.x) {
//         int cnt = shHist[b];
//         if (cnt > 0) {
//             int base = atomicAdd(&d_binCursor[b], cnt); // 预留 cnt 个位置
//             shBase[b] = base;                           // 记录块内该 bin 的全局起点增量
//         } else {
//             shBase[b] = 0;
//         }
//         shCounter[b] = 0; // 用于下一阶段的块内顺序分配
//     }
//     __syncthreads();

//     // --- 3) 二次遍历：给本块的每个元素分配最终 dst（仅共享原子）
//     if (i < N) {
//         int b      = static_cast<int>(d_codes[i] & mask);
//         int local  = atomicAdd(&shCounter[b], 1);                 // 块内局部偏移
//         int dst    = d_binOffsets[b] + shBase[b] + local;         // 全局 dst
//         d_srcIndexForDest[dst] = i;                               // 写 4B 索引
//     }
// }

// (B) 聚集拷贝：随机读 in[src]，按 dst=0..N-1 顺序写 out[dst]（store 完全合并）
__global__ void gatherCopyKernel(
    const Point2D*  __restrict__ d_in,           // [N]
    Point2D*        __restrict__ d_out,          // [N]
    const int*      __restrict__ d_srcIndexForDest, // [N]
    int N)
{
    int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= N) return;

    int src = d_srcIndexForDest[dst];
    // Plain load is fine; stores are fully coalesced which is the big win.
    Point2D v = d_in[src];
    d_out[dst] = v;
}

/**
 * @brief Pass 2: scatter points so each bin occupies a contiguous slice.
 *
 * @param in         Input points.
 * @param tmp        Scatter buffer (length = N).
 * @param codes      Morton codes.
 * @param binCursor  Per-bin cursor initialised with exclusive offsets.
 * @param N          Total elements.
 * @param mask       (1 << kBits)-1 — extracts the bin ID.
 */
__global__ void scatterToBins(const Point2D*  in,
                              Point2D*        tmp,
                              const uint32_t* codes,
                              int*            binCursor,
                              int             N,
                              int             mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int id  = codes[idx] & mask;
    int pos = atomicAdd(&binCursor[id], 1);  // unique slot in that bin
    tmp[pos] = in[idx];
}

// ────────────────────────────────────────────────────────────────
// testBinGPUCompaction_partition  –  Plan A pipeline
// ────────────────────────────────────────────────────────────────

/**
 * @brief Full Plan A pipeline: histogram → scan → scatter → per-bin compaction.
 *
 * @param input       Host-side input points.
 * @param threshold   Predicate threshold.
 * @param kBits       Bin ID bit-width.
 * @param output      Host-side vector receiving compacted points.
 * @param t_kernel_ms Returns histogram+scatter kernel time.
 * @param t_total_ms  Returns end-to-end GPU time.
 * @param kernelKind  Per-bin kernel strategy.
 */
// void testBinGPUCompaction_partition(const std::vector<Point2D>& input,
//                                     float                       threshold,
//                                     int                         kBits,
//                                     std::vector<Point2D>&       output,
//                                     float&                      t_kernel_ms,
//                                     float&                      t_total_ms,
//                                     BinKernel                   kernelKind)
// {
//     const int N       = static_cast<int>(input.size());
//     const int numBins = 1 << kBits;
//     const int mask    = numBins - 1;

//     /* CUDA events ----------------------------------------------------------*/
//     cudaEvent_t t0, t1, k0, k1;
//     cudaEventCreate(&t0); cudaEventCreate(&t1);
//     cudaEventCreate(&k0); cudaEventCreate(&k1);
//     cudaEventRecord(t0);

//     /* Allocate raw buffers -------------------------------------------------*/
//     Point2D*  d_in;  cudaMalloc(&d_in,  N * sizeof(Point2D));
//     Point2D*  d_tmp; cudaMalloc(&d_tmp, N * sizeof(Point2D));   // scatter buffer
//     Point2D*  d_out; cudaMalloc(&d_out, N * sizeof(Point2D));
//     uint32_t* d_codes; cudaMalloc(&d_codes, N * sizeof(uint32_t));

//     cudaMemcpy(d_in, input.data(),
//                N * sizeof(Point2D), cudaMemcpyHostToDevice);

//     thrust::device_vector<int> d_binSizes  (numBins,   0);
//     thrust::device_vector<int> d_binOffsets(numBins+1, 0);

//     /* Build Morton codes ---------------------------------------------------*/
//     std::vector<uint32_t> h_codes(N);
//     for (int i = 0; i < N; ++i)
//         h_codes[i] = morton2D_encode(static_cast<int>(input[i].x),
//                                      static_cast<int>(input[i].y));
//     cudaMemcpy(d_codes, h_codes.data(),
//                N * sizeof(uint32_t), cudaMemcpyHostToDevice);

//     /* Pass 1: histogram ----------------------------------------------------*/
//     const int threads = 256;
//     const int blocks  = (N + threads - 1) / threads;
//     histogramBins<<<blocks,threads>>>(d_codes,
//         thrust::raw_pointer_cast(d_binSizes.data()), N, mask);

//     /* Exclusive scan → offsets --------------------------------------------*/
//     thrust::exclusive_scan(d_binSizes.begin(), d_binSizes.end(),
//                            d_binOffsets.begin());
//     d_binOffsets[numBins] = N;   // sentinel

//     /* Pass 2: scatter ------------------------------------------------------*/
//     thrust::device_vector<int> d_binCursor = d_binOffsets;
//     cudaEventRecord(k0);
//     scatterToBins<<<blocks,threads>>>(d_in, d_tmp, d_codes,
//                                       thrust::raw_pointer_cast(d_binCursor.data()),
//                                       N, mask);
//     cudaEventRecord(k1);
//     cudaEventSynchronize(k1);

//     /* Copy metadata to host -----------------------------------------------*/
//     std::vector<int> h_offsets(numBins+1);
//     std::vector<int> h_sizes  (numBins);
//     thrust::copy(d_binOffsets.begin(), d_binOffsets.end(), h_offsets.begin());
//     thrust::copy(d_binSizes.begin(),   d_binSizes.end(),   h_sizes.begin());

//     /* Per-bin compaction ---------------------------------------------------*/
//     int totalOut = 0;
//     for (int b = 0; b < numBins; ++b) {
//         int off = h_offsets[b];
//         int sz  = h_sizes[b];
//         if (sz == 0) continue;

//         Point2D* binIn  = d_tmp + off;
//         Point2D* binOut = d_out + totalOut;
//         int      h_cnt  = 0;

//         compactOneBin(binIn, binOut, sz, threshold, h_cnt, kernelKind);
//         totalOut += h_cnt;
//     }

//     /* Copy final output ----------------------------------------------------*/
//     output.resize(totalOut);
//     cudaMemcpy(output.data(), d_out,
//                totalOut * sizeof(Point2D), cudaMemcpyDeviceToHost);

//     /* Timing ---------------------------------------------------------------*/
//     cudaEventRecord(t1); cudaEventSynchronize(t1);
//     cudaEventElapsedTime(&t_kernel_ms, k0, k1);
//     cudaEventElapsedTime(&t_total_ms,  t0, t1);

//     /* Cleanup --------------------------------------------------------------*/
//     cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_codes);
//     cudaEventDestroy(k0); cudaEventDestroy(k1);
//     cudaEventDestroy(t0); cudaEventDestroy(t1);
// }

void testBinGPUCompaction_partition(const std::vector<Point2D>& input,
                                    float                       threshold,
                                    int                         kBits,
                                    std::vector<Point2D>&       output,
                                    float&                      t_kernel_ms,
                                    float&                      t_total_ms,
                                    BinKernel                   kernelKind)
{
    const int N       = static_cast<int>(input.size());
    const int numBins = 1 << kBits;
    const int mask    = numBins - 1;

    /* CUDA events ----------------------------------------------------------*/
    cudaEvent_t t0, t1, k0, k1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    cudaEventRecord(t0);

    /* Allocate raw buffers -------------------------------------------------*/
    Point2D*  d_in;   cudaMalloc(&d_in,  N * sizeof(Point2D));
    Point2D*  d_tmp;  cudaMalloc(&d_tmp, N * sizeof(Point2D));   // 仍作为“分桶后”的缓冲
    Point2D*  d_out;  cudaMalloc(&d_out, N * sizeof(Point2D));
    uint32_t* d_codes; cudaMalloc(&d_codes, N * sizeof(uint32_t));

    cudaMemcpy(d_in, input.data(),
               N * sizeof(Point2D), cudaMemcpyHostToDevice);

    thrust::device_vector<int> d_binSizes  (numBins,   0);
    thrust::device_vector<int> d_binOffsets(numBins+1, 0);

    /* Build Morton codes ---------------------------------------------------*/
    std::vector<uint32_t> h_codes(N);
    for (int i = 0; i < N; ++i)
        h_codes[i] = morton2D_encode(static_cast<int>(input[i].x),
                                     static_cast<int>(input[i].y));
    cudaMemcpy(d_codes, h_codes.data(),
               N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Pass 1: histogram ----------------------------------------------------*/
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    histogramBins<<<blocks,threads>>>(d_codes,
        thrust::raw_pointer_cast(d_binSizes.data()), N, mask);

    /* Exclusive scan → offsets --------------------------------------------*/
    thrust::exclusive_scan(d_binSizes.begin(), d_binSizes.end(),
                           d_binOffsets.begin());
    d_binOffsets[numBins] = N;   // sentinel

    /* Pass 2: scatter  (REPLACED) -----------------------------------------*/
    // 原先是：thrust::device_vector<int> d_binCursor = d_binOffsets; // 作为起始写指针
    // 现在我们改为：先建“dst->src”索引映射，然后做一次顺序写的聚集拷贝。

    // 2.1 分配临时区：每bin计数器(清零) + 目标索引映射 [N]
    thrust::device_vector<int> d_binCursor(numBins, 0);
    thrust::device_vector<int> d_srcIndexForDest(N, -1);

    cudaEventRecord(k0);

    //2.2 为每个元素分配它在“分桶后数组”里的 dst 位置，并只写入其来源 src 下标
    buildDestMapKernel<<<blocks, threads>>>(
        d_codes,
        thrust::raw_pointer_cast(d_binOffsets.data()),
        thrust::raw_pointer_cast(d_binCursor.data()),
        thrust::raw_pointer_cast(d_srcIndexForDest.data()),
        N, mask);
    // size_t smemBytes = static_cast<size_t>(numBins) * 3 * sizeof(int);
    // buildDestMapKernel<<<blocks, threads, smemBytes>>>(
    //     d_codes,
    //     thrust::raw_pointer_cast(d_binOffsets.data()),
    //     thrust::raw_pointer_cast(d_binCursor.data()),
    //     thrust::raw_pointer_cast(d_srcIndexForDest.data()),
    //     N, mask, numBins);

    // 2.3 聚集拷贝：把 d_in 按 d_srcIndexForDest 的映射搬到 d_tmp（顺序写，写入合并）
    gatherCopyKernel<<<blocks, threads>>>(
        d_in, d_tmp,
        thrust::raw_pointer_cast(d_srcIndexForDest.data()),
        N);

    cudaEventRecord(k1);
    cudaEventSynchronize(k1);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // shared memory blocked version (ncu-safe)  – 目前不启用，因为 blocked 版本在 ncu 上会有 LaunchFailed
    ////////////////////////////////////////////////////////////////////////////////////////////////
    //  /* Pass 2: scatter (ncu-safe) -----------------------------------------*/

    // // 2.1 临时区：每bin计数器(清零) + 目标索引映射 [N]
    // thrust::device_vector<int> d_binCursor(numBins, 0);
    // thrust::device_vector<int> d_srcIndexForDest(N, -1);

    // // 2.2 blocked 版本需要的动态共享内存（三个 numBins 的 int 数组）
    // const size_t smemBytes = static_cast<size_t>(numBins) * 3 * sizeof(int);

    // // 2.3 设备默认共享内存上限（通常 48KB）
    // int dev = 0;
    // cudaGetDevice(&dev);
    // int maxDefault = 0;
    // cudaDeviceGetAttribute(&maxDefault, cudaDevAttrMaxSharedMemoryPerBlock, dev);

    // // ⚠️ 为了让 ncu 一定稳定：只在 smemBytes ≤ 默认上限（~48KB）时使用 blocked 版本
    // const bool useBlocked = (smemBytes <= static_cast<size_t>(maxDefault));

    // // ⏱ 计时开始（包含 buildDestMap + gatherCopy）
    // cudaEventRecord(k0);

    // if (useBlocked) {
    //     // 不走 opt-in，直接按默认上限跑 blocked
    //     buildDestMapKernel<<<blocks, threads, smemBytes>>>(
    //         d_codes,
    //         thrust::raw_pointer_cast(d_binOffsets.data()),
    //         thrust::raw_pointer_cast(d_binCursor.data()),
    //         thrust::raw_pointer_cast(d_srcIndexForDest.data()),
    //         N, mask, numBins);
    // } else {
    //     // 超过 48KB 就退回 naive，保证 ncu 不会 LaunchFailed
    //     buildDestMapKernel_naive<<<blocks, threads>>>(
    //         d_codes,
    //         thrust::raw_pointer_cast(d_binOffsets.data()),
    //         thrust::raw_pointer_cast(d_binCursor.data()),
    //         thrust::raw_pointer_cast(d_srcIndexForDest.data()),
    //         N, mask);
    // }

    // // 2.4 聚集拷贝（顺序写，合并良好）
    // gatherCopyKernel<<<blocks, threads>>>(
    //     d_in, d_tmp,
    //     thrust::raw_pointer_cast(d_srcIndexForDest.data()),
    //     N);

    // // ⏱ 计时结束
    // cudaEventRecord(k1);
    // cudaEventSynchronize(k1);

    /* Copy metadata to host -----------------------------------------------*/
    std::vector<int> h_offsets(numBins+1);
    std::vector<int> h_sizes  (numBins);
    thrust::copy(d_binOffsets.begin(), d_binOffsets.end(), h_offsets.begin());
    thrust::copy(d_binSizes.begin(),   d_binSizes.end(),   h_sizes.begin());

    /* Per-bin compaction ---------------------------------------------------*/
    int totalOut = 0;
    for (int b = 0; b < numBins; ++b) {
        int off = h_offsets[b];
        int sz  = h_sizes[b];
        if (sz == 0) continue;

        Point2D* binIn  = d_tmp + off;        // 注意：仍然从 d_tmp 的分段读
        Point2D* binOut = d_out + totalOut;
        int      h_cnt  = 0;

        compactOneBin(binIn, binOut, sz, threshold, h_cnt, kernelKind);
        totalOut += h_cnt;
    }

    /* Copy final output ----------------------------------------------------*/
    output.resize(totalOut);
    cudaMemcpy(output.data(), d_out,
               totalOut * sizeof(Point2D), cudaMemcpyDeviceToHost);

    /* Timing ---------------------------------------------------------------*/
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&t_kernel_ms, k0, k1);   // 现在包括 buildDestMap + gatherCopy 两个 kernel
    cudaEventElapsedTime(&t_total_ms,  t0, t1);

    /* Cleanup --------------------------------------------------------------*/
    cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_codes);
    cudaEventDestroy(k0); cudaEventDestroy(k1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

// ────────────────────────────────────────────────────────────────
// compactWarpGPU  – convenience micro-benchmark
// ────────────────────────────────────────────────────────────────

/**
 * @brief Warp-shuffle compaction of a single, contiguous bin.
 *
 * The routine performs a micro-benchmark:
 *   1. Upload the predicate threshold to device constant memory.
 *   2. Allocate a device-side counter initialised to 0.
 *   3. Launch the warp-shuffle kernel (::compact_points_warp).
 *   4. Copy the final element count back to the host.
 *
 * @param d_in        Device pointer to bin input (contiguous slice).
 * @param d_out       Device pointer to output buffer (same slice size).
 * @param N           Number of elements in the bin.
 * @param threshold   Temperature (or other) predicate threshold.
 * @param h_outCount  Host-side integer that receives the valid-element count.
 *
 * @note This helper is intended for profiling individual kernels rather than
 *       full pipelines; it assumes @p d_in holds one bin’s data only.
 */
void compactWarpGPU(const Point2D* d_in,
                    Point2D*       d_out,
                    int            N,
                    float          threshold,
                    int&           h_outCount)
{
    // ── 1. Push predicate threshold to constant memory (device global symbol)
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // ── 2. Allocate & zero a device-side counter used by the warp kernel
    int* d_cnt = nullptr;
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMemset(d_cnt, 0, sizeof(int));

    // ── 3. Launch warp-shuffle compaction kernel
    //     (kernel implementation writes the number of valid elements to d_cnt)
    compact_points_warp(const_cast<Point2D*>(d_in), d_out, d_cnt, N);

    // ── 4. Retrieve the final count back to the host
    cudaMemcpy(&h_outCount, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_cnt);   // tidy up
}


// ────────────────────────────────────────────────────────────────
// compactOneBin  – unified dispatcher
// ────────────────────────────────────────────────────────────────

/**
 * @brief Dispatch a chosen ::BinKernel variant to compact one bin in place.
 *
 * Workflow
 * --------
 *  1. Copy the predicate @p threshold into the device‐side constant symbol
 *     (`d_threshold`) so every kernel can access it quickly.
 *  2. Allocate a device counter (@p d_cnt) initialised to zero.
 *     • Shared-memory kernel fills @p h_outCnt internally (no counter needed).
 *     • Warp / Bitmask kernels update @p d_cnt atomically.
 *  3. Select the kernel:
 *        • ::BinKernel::Shared   →  per-block shared-memory scan
 *        • ::BinKernel::Warp     →  warp-shuffle implementation
 *        • ::BinKernel::Bitmask  →  ballot + popc prefix-sum
 *        • ::BinKernel::Auto     →  simple heuristic based on @p N
 *  4. After kernel completion, copy the final element count (either
 *     `h_outCnt` or the value in @p d_cnt) back to the host.
 *
 * @param d_in       Device pointer to the bin’s input slice (contiguous).
 * @param d_out      Device pointer to the bin’s output base slice.
 * @param N          Number of elements in the bin.
 * @param threshold  Predicate threshold applied inside the kernels.
 * @param h_outCnt   Host-side integer that receives the valid-element count.
 * @param kind       Kernel strategy (Shared / Warp / Bitmask / Auto-select).
 *
 * @note  The caller must ensure @p d_out has at least @p N slots available,
 *        because worst-case (all elements valid) the output equals the input size.
 */
void compactOneBin(Point2D*  d_in,
                   Point2D*  d_out,
                   int       N,
                   float     threshold,
                   int&      h_outCnt,
                   BinKernel kind)
{
    // ── 1. Upload predicate threshold to constant memory
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // ── 2. Allocate a device counter (only needed for Warp / Bitmask kernels)
    int* d_cnt = nullptr;
    cudaMalloc(&d_cnt, sizeof(int));
    cudaMemset(d_cnt, 0, sizeof(int));

    // ── 3. If Auto, pick a kernel heuristically based on bin size
    if (kind == BinKernel::Auto)
        kind = (N <= 32)   ? BinKernel::Bitmask :    // tiny bins
               (N <= 1024) ? BinKernel::Warp    :    // medium bins
                             BinKernel::Shared;      // large bins

    // ── 4. Launch the selected kernel
    switch (kind) {

        case BinKernel::Shared:
            // Shared-memory version fills h_outCnt directly
            compactSharedGPU(d_in, d_out, N, threshold, h_outCnt);
            break;

        case BinKernel::Warp:
            // Warp kernel writes its element count to d_cnt
            compact_points_warp(d_in, d_out, d_cnt, N);
            cudaMemcpy(&h_outCnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            break;

        case BinKernel::Bitmask:
            // Bitmask kernel also uses d_cnt
            compact_points_bitmask(d_in, d_out, d_cnt, N);
            cudaMemcpy(&h_outCnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            break;

        default:
            // Should never hit this path; treat as no-op
            h_outCnt = 0;
            break;
    }

    // ── 5. Clean-up
    cudaFree(d_cnt);
}
