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
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>  // par.on(stream)

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

extern float d_threshold;   ///< device-side predicate threshold

int g_block_size = BLOCK_SIZE;  // default 256

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { \
  cudaError_t err__ = (x); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
  } \
} while(0)
#endif

// ===== Q6 toggle: Plan A Pass-2 use gatherCopy (baseline) or direct scatter =====
static bool g_planA_use_gather = true;   // 默认 baseline: buildDestMap + gatherCopy

// 对外暴露一个 setter，供 driver 调用：set_planA_use_gather(false) => 走 scatterToBins
void set_planA_use_gather(bool v) { g_planA_use_gather = v; }

// 命中率控制：把 threshold 选成 (1 - h) 分位点，使 “temp > threshold” ≈ h
float choose_threshold_for_rate(const std::vector<Point2D>& input, double target_hit_rate) {
    std::vector<float> temps; temps.reserve(input.size());
    for (const auto& p : input) temps.push_back(p.temp);
    // k = (1 - h) * N 处的元素作为 threshold
    const size_t k = static_cast<size_t>((1.0 - target_hit_rate) * temps.size());
    std::nth_element(temps.begin(), temps.begin() + std::min(k, temps.size()-1), temps.end());
    return temps[std::min(k, temps.size()-1)];
}

// RAII：把已有 std::vector 的内存“临时注册”为 pinned，析构时自动取消注册
struct PinnedGuard {
    void*  p{nullptr};
    size_t n{0};
    bool   active{false};
    PinnedGuard(void* ptr, size_t bytes) : p(ptr), n(bytes), active(false) {
        if (p && n) {
            cudaError_t e = cudaHostRegister(p, n, 0);
            if (e == cudaSuccess) active = true;
            else if (e != cudaErrorHostMemoryAlreadyRegistered) {
                fprintf(stderr, "cudaHostRegister failed: %s\n", cudaGetErrorString(e));
                std::exit(1);
            } else {
                active = true; // 已被注册也当作成功使用
            }
        }
    }
    ~PinnedGuard() {
        if (active && p) {
            cudaHostUnregister(p); // 同步在外面做，这里只是收尾
        }
    }
    // 不允许拷贝
    PinnedGuard(const PinnedGuard&) = delete;
    PinnedGuard& operator=(const PinnedGuard&) = delete;
    // 允许移动
    PinnedGuard(PinnedGuard&& o) noexcept { p=o.p; n=o.n; active=o.active; o.p=nullptr; o.n=0; o.active=false; }
    PinnedGuard& operator=(PinnedGuard&& o) noexcept {
        if (this!=&o) { if(active&&p) cudaHostUnregister(p); p=o.p; n=o.n; active=o.active; o.p=nullptr; o.n=0; o.active=false; }
        return *this;
    }
};


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
// void testBinGPUCompaction_atomic(const std::vector<Point2D>& input,
//                                  float                       threshold,
//                                  int                         kBits,
//                                  std::vector<Point2D>&       output,
//                                  float&                      t_kernel_ms,
//                                  float&                      t_total_ms)
// {
//     const int N = static_cast<int>(input.size());
//     (void)kBits;  // mask is pre-computed but unused by this kernel

//     /* Create CUDA events for timing ----------------------------------------*/
//     cudaEvent_t t0, t1, k0, k1;
//     cudaEventCreate(&t0); cudaEventCreate(&t1);
//     cudaEventCreate(&k0); cudaEventCreate(&k1);
//     cudaEventRecord(t0);

//     /* Allocate buffers -----------------------------------------------------*/
//     Point2D*  d_in        = nullptr;
//     Point2D*  d_out       = nullptr;
//     uint32_t* d_codes     = nullptr;
//     int*      d_globalCnt = nullptr;

//     cudaMalloc(&d_in,  N * sizeof(Point2D));
//     cudaMalloc(&d_out, N * sizeof(Point2D));
//     cudaMalloc(&d_codes, N * sizeof(uint32_t));
//     cudaMalloc(&d_globalCnt, sizeof(int));
//     cudaMemset(d_globalCnt, 0, sizeof(int));

//     cudaMemcpy(d_in, input.data(),
//                N * sizeof(Point2D), cudaMemcpyHostToDevice);

//     /* Build Morton codes on host ------------------------------------------*/
//     std::vector<uint32_t> h_codes(N);
//     for (int i = 0; i < N; ++i)
//         h_codes[i] = morton2D_encode(static_cast<int>(input[i].x),
//                                      static_cast<int>(input[i].y));
//     cudaMemcpy(d_codes, h_codes.data(),
//                N * sizeof(uint32_t), cudaMemcpyHostToDevice);

//     /* Launch kernel --------------------------------------------------------*/
//     const int threads = 256;
//     const int blocks  = (N + threads - 1) / threads;
//     cudaEventRecord(k0);

//     compactBinAtomic<<<blocks, threads>>>(d_in, d_out, d_globalCnt,
//                                           d_codes, N, 0, threshold);

//     cudaEventRecord(k1);
//     cudaEventSynchronize(k1);

//     /* Copy result count & elements back ------------------------------------*/
//     int total = 0;
//     cudaMemcpy(&total, d_globalCnt,
//                sizeof(int), cudaMemcpyDeviceToHost);

//     output.resize(total);
//     if (total > 0)
//         cudaMemcpy(output.data(), d_out,
//                    total * sizeof(Point2D), cudaMemcpyDeviceToHost);

//     /* Timings --------------------------------------------------------------*/
//     cudaEventRecord(t1); cudaEventSynchronize(t1);
//     cudaEventElapsedTime(&t_kernel_ms, k0, k1);
//     cudaEventElapsedTime(&t_total_ms,  t0, t1);

//     /* Cleanup --------------------------------------------------------------*/
//     cudaEventDestroy(k0); cudaEventDestroy(k1);
//     cudaEventDestroy(t0); cudaEventDestroy(t1);
//     cudaFree(d_in); cudaFree(d_out);
//     cudaFree(d_codes); cudaFree(d_globalCnt);
// }


void testBinGPUCompaction_atomic(const std::vector<Point2D>& input,
                                 float                       threshold,
                                 int                         kBits,
                                 std::vector<Point2D>&       output,
                                 float&                      t_kernel_ms,
                                 float&                      t_total_ms)
{
    const int N       = static_cast<int>(input.size());
    const int numBins = 1 << kBits;
    const int mask    = numBins - 1;

    // --- CUDA events（统一口径）
    cudaEvent_t eStart, eStop, kStart, kStop;
    cudaEventCreate(&eStart); cudaEventCreate(&eStop);
    cudaEventCreate(&kStart); cudaEventCreate(&kStop);

    // --- 设备侧资源
    Point2D*  d_in  = nullptr;
    Point2D*  d_out = nullptr;
    uint32_t* d_codes = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,   N * sizeof(Point2D)));
    CUDA_CHECK(cudaMalloc(&d_out,  N * sizeof(Point2D)));
    CUDA_CHECK(cudaMalloc(&d_codes, N * sizeof(uint32_t)));

    // 设备端直方图/偏移/填充计数器
    thrust::device_vector<int> d_binValid (numBins, 0);   // 每 bin 有效数
    thrust::device_vector<int> d_binBase  (numBins+1, 0); // 每 bin 基址（exclusive）
    thrust::device_vector<int> d_binFill  (numBins, 0);   // Pass-2 每 bin 运行计数器

    const int threads = g_block_size;
    const int blocks  = (N + threads - 1) / threads;

    // --- E2E：从第一笔 H2D(input) 开始
    CUDA_CHECK(cudaEventRecord(eStart, 0));
    CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(),
                               N * sizeof(Point2D),
                               cudaMemcpyHostToDevice, 0));

    // --- Kernel-only：包含 生成codes + 计数 + 扫描 + 写回
    CUDA_CHECK(cudaEventRecord(kStart, 0));

    // Pass-0: 生成 Morton codes（GPU）
    genMortonCodesKernel<<<blocks, threads>>>(d_in, d_codes, N);

    // Pass-1: 统计每桶有效元素数
    countValidPerBin_fromCodes<<<blocks, threads>>>(
        d_in, d_codes,
        thrust::raw_pointer_cast(d_binValid.data()),
        N, mask, threshold);

    // Scan: 有效数做 exclusive_scan → bin 基址
    thrust::exclusive_scan(d_binValid.begin(), d_binValid.end(),
                           d_binBase.begin());

    // 计算总有效数（用于回拷大小）
    int totalValid = thrust::reduce(d_binValid.begin(), d_binValid.end(), 0, thrust::plus<int>());

    // 写入 sentinel（binBase[numBins] = totalValid）
    d_binBase[numBins] = totalValid;

    // Pass-2: 按桶写出（每桶原子），out[binBase[b] + offset]
    thrust::fill(d_binFill.begin(), d_binFill.end(), 0);
    writePerBinAtomic_fromCodes<<<blocks, threads>>>(
        d_in, d_codes, d_out,
        thrust::raw_pointer_cast(d_binBase.data()),
        thrust::raw_pointer_cast(d_binFill.data()),
        N, mask, threshold);

    CUDA_CHECK(cudaEventRecord(kStop, 0));
    CUDA_CHECK(cudaEventSynchronize(kStop));  // 保证 kernel-only 结束

    // --- 回拷结果（E2E 继续）
    output.resize(totalValid);
    if (totalValid > 0) {
        CUDA_CHECK(cudaMemcpyAsync(output.data(), d_out,
                                   totalValid * sizeof(Point2D),
                                   cudaMemcpyDeviceToHost, 0));
    }

    CUDA_CHECK(cudaEventRecord(eStop, 0));
    CUDA_CHECK(cudaEventSynchronize(eStop));

    // --- 读计时
    CUDA_CHECK(cudaEventElapsedTime(&t_kernel_ms, kStart, kStop));
    CUDA_CHECK(cudaEventElapsedTime(&t_total_ms,  eStart, eStop));

    // --- 清理
    cudaEventDestroy(eStart); cudaEventDestroy(eStop);
    cudaEventDestroy(kStart); cudaEventDestroy(kStop);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_codes);
}

void testPlanB_breakdown(const std::vector<Point2D>& input,
                         float                       threshold,
                         int                         kBits,
                         BreakdownPlanB&             out,
                         std::vector<Point2D>*       host_output /*opt, 可为nullptr*/)
{
    const int N       = (int)input.size();
    const int numBins = 1 << kBits;
    const int mask    = numBins - 1;

    // 事件
    cudaEvent_t e0,e1,k0,k1, ec0,ec1, ecount0,ecount1, escan1, ered1, ew0,ew1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    cudaEventCreate(&ec0); cudaEventCreate(&ec1);
    cudaEventCreate(&ecount0); cudaEventCreate(&ecount1);
    cudaEventCreate(&escan1); cudaEventCreate(&ered1);
    cudaEventCreate(&ew0); cudaEventCreate(&ew1);

    // 设备内存
    Point2D *d_in=nullptr, *d_out=nullptr; uint32_t* d_codes=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,    N * sizeof(Point2D)));
    CUDA_CHECK(cudaMalloc(&d_out,   N * sizeof(Point2D)));
    CUDA_CHECK(cudaMalloc(&d_codes, N * sizeof(uint32_t)));

    thrust::device_vector<int> d_binValid(numBins, 0);
    thrust::device_vector<int> d_binBase (numBins+1, 0);
    thrust::device_vector<int> d_binFill (numBins, 0);

    //const int threads=256, blocks=(N+threads-1)/threads;
    const int threads = g_block_size;
    const int blocks  = (N + threads - 1) / threads;

    // E2E 开始：H2D
    CUDA_CHECK(cudaEventRecord(e0, 0));
    CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(), N*sizeof(Point2D), cudaMemcpyHostToDevice, 0));

    // Kernel-only 从 codes 前开始
    CUDA_CHECK(cudaEventRecord(k0, 0));

    // Pass-0: codes (GPU)
    CUDA_CHECK(cudaEventRecord(ec0, 0));
    genMortonCodesKernel<<<blocks, threads>>>(d_in, d_codes, N);
    CUDA_CHECK(cudaEventRecord(ec1, 0));

    // Pass-1: count valid per bin
    CUDA_CHECK(cudaEventRecord(ecount0, 0));
    countValidPerBin_fromCodes<<<blocks, threads>>>(
        d_in, d_codes, thrust::raw_pointer_cast(d_binValid.data()),
        N, mask, threshold);
    CUDA_CHECK(cudaEventRecord(ecount1, 0));

    // Scan + reduce
    thrust::exclusive_scan(d_binValid.begin(), d_binValid.end(), d_binBase.begin());
    int totalValid = thrust::reduce(d_binValid.begin(), d_binValid.end(), 0, thrust::plus<int>());
    d_binBase[numBins] = totalValid; // sentinel
    CUDA_CHECK(cudaEventRecord(escan1, 0)); // scan+reduce 截止到此
    // 注：为单点计时清晰，我们把 scan 和 reduce 合并记入 scan_ms，或按需拆成 scan_ms / reduce_ms
    // 若要拆分，可在 exclusive_scan 前后、reduce 前后再打事件。

    // Pass-2: write per bin (atomic)
    thrust::fill(d_binFill.begin(), d_binFill.end(), 0);
    CUDA_CHECK(cudaEventRecord(ew0, 0));
    writePerBinAtomic_fromCodes<<<blocks, threads>>>(
        d_in, d_codes, d_out,
        thrust::raw_pointer_cast(d_binBase.data()),
        thrust::raw_pointer_cast(d_binFill.data()),
        N, mask, threshold);
    CUDA_CHECK(cudaEventRecord(ew1, 0));
    CUDA_CHECK(cudaEventRecord(k1, 0));
    CUDA_CHECK(cudaEventSynchronize(k1));

    // D2H （计入 E2E）
    if (host_output) {
        host_output->resize(totalValid);
        if (totalValid > 0) {
            CUDA_CHECK(cudaMemcpyAsync(host_output->data(), d_out,
                                       totalValid*sizeof(Point2D),
                                       cudaMemcpyDeviceToHost, 0));
        }
    }
    CUDA_CHECK(cudaEventRecord(e1, 0));
    CUDA_CHECK(cudaEventSynchronize(e1));

    // 读时长
    out.total_valid = totalValid;
    CUDA_CHECK(cudaEventElapsedTime(&out.kernel_ms, k0, k1));
    CUDA_CHECK(cudaEventElapsedTime(&out.e2e_ms,    e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&out.codes_ms,  ec0, ec1));
    CUDA_CHECK(cudaEventElapsedTime(&out.count_ms,  ecount0, ecount1));
    CUDA_CHECK(cudaEventElapsedTime(&out.write_ms,  ew0, ew1));
    // scan_ms（含 reduce）：用 ehist->escan1 区间；这里复用 ecount1→escan1
    CUDA_CHECK(cudaEventElapsedTime(&out.scan_ms,   ecount1, escan1));
    // 如需把 reduce 单独拆出，可在 reduce 前后再打两个事件得到 out.reduce_ms

    // 清理
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(k0); cudaEventDestroy(k1);
    cudaEventDestroy(ec0); cudaEventDestroy(ec1);
    cudaEventDestroy(ecount0); cudaEventDestroy(ecount1);
    cudaEventDestroy(escan1); cudaEventDestroy(ered1);
    cudaEventDestroy(ew0); cudaEventDestroy(ew1);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_codes);
}

// use pinned memory for async H2D/D2H 
// void testBinGPUCompaction_atomic(const std::vector<Point2D>& input,
//                                  float                       threshold,
//                                  int                         /*kBits*/,
//                                  std::vector<Point2D>&       output,
//                                  float&                      t_kernel_ms,
//                                  float&                      t_total_ms)
// {
//     const int    N       = static_cast<int>(input.size());
//     const size_t inBytes = N * sizeof(Point2D);

//     // ---- Device ----
//     Point2D *d_in=nullptr, *d_out=nullptr;
//     int*     d_globalCnt=nullptr;
//     CUDA_CHECK(cudaMalloc(&d_in,  inBytes));
//     CUDA_CHECK(cudaMalloc(&d_out, inBytes));
//     CUDA_CHECK(cudaMalloc(&d_globalCnt, sizeof(int)));
//     CUDA_CHECK(cudaMemset(d_globalCnt, 0, sizeof(int)));

//     // ---- Stream / events ----
//     cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
//     cudaEvent_t e0,e1,k0,k1;
//     CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
//     CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));

//     // 直接把 input 的现有内存注册为 pinned（无额外拷贝）
//     CUDA_CHECK(cudaHostRegister((void*)input.data(), inBytes, cudaHostRegisterDefault));

//     // E2E start: H2D
//     CUDA_CHECK(cudaEventRecord(e0, s));
//     CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(), inBytes, cudaMemcpyHostToDevice, s));

//     // kernel-only
//     const int threads = 256;
//     const int blocks  = (N + threads - 1) / threads;
//     CUDA_CHECK(cudaEventRecord(k0, s));
//     compactBinAtomic<<<blocks, threads, 0, s>>>(d_in, d_out, d_globalCnt,
//                                                 /*codes*/nullptr, N, /*mask*/0, threshold);
//     CUDA_CHECK(cudaEventRecord(k1, s));
//     CUDA_CHECK(cudaEventSynchronize(k1));

//     // 先取回计数
//     int total = 0;
//     CUDA_CHECK(cudaMemcpyAsync(&total, d_globalCnt, sizeof(int),
//                                cudaMemcpyDeviceToHost, s));
//     CUDA_CHECK(cudaStreamSynchronize(s)); // 确保 total 可用

//     // 准备 Host 输出：直接注册 output 的内存为 pinned，再做 D2H
//     output.resize(total);
//     if (total > 0) {
//         const size_t outBytes = total * sizeof(Point2D);
//         CUDA_CHECK(cudaHostRegister(output.data(), outBytes, cudaHostRegisterDefault));
//         CUDA_CHECK(cudaMemcpyAsync(output.data(), d_out, outBytes,
//                                    cudaMemcpyDeviceToHost, s));
//         CUDA_CHECK(cudaEventRecord(e1, s));
//         CUDA_CHECK(cudaEventSynchronize(e1));
//         CUDA_CHECK(cudaHostUnregister(output.data()));
//     } else {
//         CUDA_CHECK(cudaEventRecord(e1, s));
//         CUDA_CHECK(cudaEventSynchronize(e1));
//     }

//     // timings
//     CUDA_CHECK(cudaEventElapsedTime(&t_kernel_ms, k0, k1));
//     CUDA_CHECK(cudaEventElapsedTime(&t_total_ms,  e0, e1));

//     // cleanup
//     CUDA_CHECK(cudaHostUnregister((void*)input.data()));
//     cudaEventDestroy(e0); cudaEventDestroy(e1);
//     cudaEventDestroy(k0); cudaEventDestroy(k1);
//     cudaStreamDestroy(s);
//     cudaFree(d_in); cudaFree(d_out); cudaFree(d_globalCnt);
// }





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
//     Point2D*  d_in;   cudaMalloc(&d_in,  N * sizeof(Point2D));
//     Point2D*  d_tmp;  cudaMalloc(&d_tmp, N * sizeof(Point2D));   // 仍作为“分桶后”的缓冲
//     Point2D*  d_out;  cudaMalloc(&d_out, N * sizeof(Point2D));
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

//     /* Pass 2: scatter  (REPLACED) -----------------------------------------*/
//     // 原先是：thrust::device_vector<int> d_binCursor = d_binOffsets; // 作为起始写指针
//     // 现在我们改为：先建“dst->src”索引映射，然后做一次顺序写的聚集拷贝。

//     // 2.1 分配临时区：每bin计数器(清零) + 目标索引映射 [N]
//     thrust::device_vector<int> d_binCursor(numBins, 0);
//     thrust::device_vector<int> d_srcIndexForDest(N, -1);

//     cudaEventRecord(k0);

//     //2.2 为每个元素分配它在“分桶后数组”里的 dst 位置，并只写入其来源 src 下标
//     buildDestMapKernel<<<blocks, threads>>>(
//         d_codes,
//         thrust::raw_pointer_cast(d_binOffsets.data()),
//         thrust::raw_pointer_cast(d_binCursor.data()),
//         thrust::raw_pointer_cast(d_srcIndexForDest.data()),
//         N, mask);
//     // size_t smemBytes = static_cast<size_t>(numBins) * 3 * sizeof(int);
//     // buildDestMapKernel<<<blocks, threads, smemBytes>>>(
//     //     d_codes,
//     //     thrust::raw_pointer_cast(d_binOffsets.data()),
//     //     thrust::raw_pointer_cast(d_binCursor.data()),
//     //     thrust::raw_pointer_cast(d_srcIndexForDest.data()),
//     //     N, mask, numBins);

//     // 2.3 聚集拷贝：把 d_in 按 d_srcIndexForDest 的映射搬到 d_tmp（顺序写，写入合并）
//     gatherCopyKernel<<<blocks, threads>>>(
//         d_in, d_tmp,
//         thrust::raw_pointer_cast(d_srcIndexForDest.data()),
//         N);

//     cudaEventRecord(k1);
//     cudaEventSynchronize(k1);

//     ////////////////////////////////////////////////////////////////////////////////////////////////
//     // shared memory blocked version (ncu-safe)  – 目前不启用，因为 blocked 版本在 ncu 上会有 LaunchFailed
//     ////////////////////////////////////////////////////////////////////////////////////////////////
//     //  /* Pass 2: scatter (ncu-safe) -----------------------------------------*/

//     // // 2.1 临时区：每bin计数器(清零) + 目标索引映射 [N]
//     // thrust::device_vector<int> d_binCursor(numBins, 0);
//     // thrust::device_vector<int> d_srcIndexForDest(N, -1);

//     // // 2.2 blocked 版本需要的动态共享内存（三个 numBins 的 int 数组）
//     // const size_t smemBytes = static_cast<size_t>(numBins) * 3 * sizeof(int);

//     // // 2.3 设备默认共享内存上限（通常 48KB）
//     // int dev = 0;
//     // cudaGetDevice(&dev);
//     // int maxDefault = 0;
//     // cudaDeviceGetAttribute(&maxDefault, cudaDevAttrMaxSharedMemoryPerBlock, dev);

//     // // ⚠️ 为了让 ncu 一定稳定：只在 smemBytes ≤ 默认上限（~48KB）时使用 blocked 版本
//     // const bool useBlocked = (smemBytes <= static_cast<size_t>(maxDefault));

//     // // ⏱ 计时开始（包含 buildDestMap + gatherCopy）
//     // cudaEventRecord(k0);

//     // if (useBlocked) {
//     //     // 不走 opt-in，直接按默认上限跑 blocked
//     //     buildDestMapKernel<<<blocks, threads, smemBytes>>>(
//     //         d_codes,
//     //         thrust::raw_pointer_cast(d_binOffsets.data()),
//     //         thrust::raw_pointer_cast(d_binCursor.data()),
//     //         thrust::raw_pointer_cast(d_srcIndexForDest.data()),
//     //         N, mask, numBins);
//     // } else {
//     //     // 超过 48KB 就退回 naive，保证 ncu 不会 LaunchFailed
//     //     buildDestMapKernel_naive<<<blocks, threads>>>(
//     //         d_codes,
//     //         thrust::raw_pointer_cast(d_binOffsets.data()),
//     //         thrust::raw_pointer_cast(d_binCursor.data()),
//     //         thrust::raw_pointer_cast(d_srcIndexForDest.data()),
//     //         N, mask);
//     // }

//     // // 2.4 聚集拷贝（顺序写，合并良好）
//     // gatherCopyKernel<<<blocks, threads>>>(
//     //     d_in, d_tmp,
//     //     thrust::raw_pointer_cast(d_srcIndexForDest.data()),
//     //     N);

//     // // ⏱ 计时结束
//     // cudaEventRecord(k1);
//     // cudaEventSynchronize(k1);

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

//         Point2D* binIn  = d_tmp + off;        // 注意：仍然从 d_tmp 的分段读
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
//     cudaEventElapsedTime(&t_kernel_ms, k0, k1);   // 现在包括 buildDestMap + gatherCopy 两个 kernel
//     cudaEventElapsedTime(&t_total_ms,  t0, t1);

//     /* Cleanup --------------------------------------------------------------*/
//     cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_codes);
//     cudaEventDestroy(k0); cudaEventDestroy(k1);
//     cudaEventDestroy(t0); cudaEventDestroy(t1);
// }

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

//     // ---------- Events ----------
//     cudaEvent_t e0,e1,k0,k1, eh0,eh1, es1, esc0,esc1, ec0,ec1;
//     cudaEventCreate(&e0);  cudaEventCreate(&e1);
//     cudaEventCreate(&k0);  cudaEventCreate(&k1);
//     cudaEventCreate(&eh0); cudaEventCreate(&eh1);   // histogram window
//     cudaEventCreate(&es1);                          // scan done
//     cudaEventCreate(&esc0); cudaEventCreate(&esc1); // scatter window
//     cudaEventCreate(&ec0);  cudaEventCreate(&ec1);  // per-bin compaction

//     // ---------- Buffers ----------
//     Point2D*  d_in  = nullptr;
//     Point2D*  d_tmp = nullptr;    // scattered(binned) buffer
//     Point2D*  d_out = nullptr;
//     uint32_t* d_codes = nullptr;

//     cudaMalloc(&d_in,  N * sizeof(Point2D));
//     cudaMalloc(&d_tmp, N * sizeof(Point2D));
//     cudaMalloc(&d_out, N * sizeof(Point2D));
//     cudaMalloc(&d_codes, N * sizeof(uint32_t));

//     // Host → Device : input （E2E 计时包含）
//     cudaEventRecord(e0, 0);
//     cudaMemcpyAsync(d_in, input.data(),
//                     N * sizeof(Point2D), cudaMemcpyHostToDevice, 0);

//     // Morton codes 在主机端生成（不计入 E2E）；只把 H2D(codes) 计入 E2E
//     std::vector<uint32_t> h_codes(N);
//     for (int i = 0; i < N; ++i) {
//         h_codes[i] = morton2D_encode(static_cast<int>(input[i].x),
//                                      static_cast<int>(input[i].y));
//     }
//     cudaMemcpyAsync(d_codes, h_codes.data(),
//                     N * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);

//     // ---------- Device metadata ----------
//     thrust::device_vector<int> d_binSizes  (numBins,   0);
//     thrust::device_vector<int> d_binOffsets(numBins+1, 0);

//     const int threads = 256;
//     const int blocks  = (N + threads - 1) / threads;

//     // ===== Kernel-only 计时从 histogram 前开始 =====
//     cudaEventRecord(k0, 0);

//     // Pass-1: histogram
//     cudaEventRecord(eh0, 0);
//     histogramBins<<<blocks,threads>>>(d_codes,
//         thrust::raw_pointer_cast(d_binSizes.data()), N, mask);
//     cudaEventRecord(eh1, 0);

//     // scan → offsets
//     thrust::exclusive_scan(d_binSizes.begin(), d_binSizes.end(),
//                            d_binOffsets.begin());
//     d_binOffsets[numBins] = N;   // sentinel
//     cudaEventRecord(es1, 0);

//     // Pass-2: scatter (buildDestMap + gatherCopy)
//     thrust::device_vector<int> d_binCursor(numBins, 0);
//     thrust::device_vector<int> d_srcIndexForDest(N, -1);

//     cudaEventRecord(esc0, 0);
//     buildDestMapKernel<<<blocks, threads>>>(
//         d_codes,
//         thrust::raw_pointer_cast(d_binOffsets.data()),
//         thrust::raw_pointer_cast(d_binCursor.data()),
//         thrust::raw_pointer_cast(d_srcIndexForDest.data()),
//         N, mask);

//     gatherCopyKernel<<<blocks, threads>>>(
//         d_in, d_tmp,
//         thrust::raw_pointer_cast(d_srcIndexForDest.data()),
//         N);
//     cudaEventRecord(esc1, 0);

//     // 取回 offsets/sizes 到主机用于 per-bin 调度（很小的 D2H，影响可忽略）
//     std::vector<int> h_offsets(numBins+1);
//     std::vector<int> h_sizes  (numBins);
//     thrust::copy(d_binOffsets.begin(), d_binOffsets.end(), h_offsets.begin());
//     thrust::copy(d_binSizes.begin(),   d_binSizes.end(),   h_sizes.begin());

//     // Per-bin compaction（纳入 kernel-only）
//     int totalOut = 0;
//     cudaEventRecord(ec0, 0);
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
//     cudaDeviceSynchronize();   // 保证所有 bin 的 kernel 都结束
//     cudaEventRecord(ec1, 0);

//     // ===== Kernel-only 截止到此 =====
//     cudaEventRecord(k1, 0);
//     cudaEventSynchronize(k1);

//     // 最终输出回拷（E2E 包含）
//     output.resize(totalOut);
//     if (totalOut > 0) {
//         cudaMemcpyAsync(output.data(), d_out,
//                         totalOut * sizeof(Point2D), cudaMemcpyDeviceToHost, 0);
//     }

//     // E2E 结束
//     cudaEventRecord(e1, 0);
//     cudaEventSynchronize(e1);

//     // ---------- Timings ----------
//     cudaEventElapsedTime(&t_kernel_ms, k0, k1);
//     cudaEventElapsedTime(&t_total_ms,  e0, e1);

//     float t_hist=0.f, t_scan=0.f, t_scatter=0.f, t_comp=0.f;
//     cudaEventElapsedTime(&t_hist,    eh0,  eh1);
//     cudaEventElapsedTime(&t_scan,    eh1,  es1);
//     cudaEventElapsedTime(&t_scatter, esc0, esc1);
//     cudaEventElapsedTime(&t_comp,    ec0,  ec1);

//     std::cout << "   [Plan-A breakdown] "
//               << "hist "    << t_hist
//               << " ms, scan "    << t_scan
//               << " ms, scatter " << t_scatter
//               << " ms, compact " << t_comp << " ms\n";

//     // ---------- Cleanup ----------
//     cudaEventDestroy(e0);  cudaEventDestroy(e1);
//     cudaEventDestroy(k0);  cudaEventDestroy(k1);
//     cudaEventDestroy(eh0); cudaEventDestroy(eh1);
//     cudaEventDestroy(es1);
//     cudaEventDestroy(esc0); cudaEventDestroy(esc1);
//     cudaEventDestroy(ec0);  cudaEventDestroy(ec1);

//     cudaFree(d_in);
//     cudaFree(d_tmp);
//     cudaFree(d_out);
//     cudaFree(d_codes);
// }

// gpu generated morton codes, no longer copy from host
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

    // ---------- Events ----------
    cudaEvent_t e0,e1,k0,k1, ecodes0,ecodes1, eh0,eh1, es1, esc0,esc1, ec0,ec1;
    cudaEventCreate(&e0);     cudaEventCreate(&e1);
    cudaEventCreate(&k0);     cudaEventCreate(&k1);
    cudaEventCreate(&ecodes0);cudaEventCreate(&ecodes1); // codes (GPU)
    cudaEventCreate(&eh0);    cudaEventCreate(&eh1);     // histogram
    cudaEventCreate(&es1);                                // scan done
    cudaEventCreate(&esc0);   cudaEventCreate(&esc1);    // scatter
    cudaEventCreate(&ec0);    cudaEventCreate(&ec1);     // per-bin compaction

    // ---------- Buffers ----------
    Point2D*  d_in  = nullptr;
    Point2D*  d_tmp = nullptr;    // scattered(binned) buffer
    Point2D*  d_out = nullptr;
    uint32_t* d_codes = nullptr;

    cudaMalloc(&d_in,    N * sizeof(Point2D));
    cudaMalloc(&d_tmp,   N * sizeof(Point2D));
    cudaMalloc(&d_out,   N * sizeof(Point2D));
    cudaMalloc(&d_codes, N * sizeof(uint32_t));

    // Host → Device : input （E2E 计时包含）
    cudaEventRecord(e0, 0);
    cudaMemcpyAsync(d_in, input.data(),
                    N * sizeof(Point2D), cudaMemcpyHostToDevice, 0);

    // ---------- Device metadata ----------
    thrust::device_vector<int> d_binSizes  (numBins,   0);
    thrust::device_vector<int> d_binOffsets(numBins+1, 0);

    //const int threads = 256;
    const int threads = g_block_size;
    const int blocks  = (N + threads - 1) / threads;

    // ===== Kernel-only 计时从 codes 前开始 =====
    cudaEventRecord(k0, 0);

    // Pass-0: 在 GPU 上生成 Morton codes（不再在 CPU 侧构造/拷贝）
    cudaEventRecord(ecodes0, 0);
    genMortonCodesKernel<<<blocks, threads>>>(d_in, d_codes, N);
    cudaEventRecord(ecodes1, 0);

    // Pass-1: histogram
    cudaEventRecord(eh0, 0);
    histogramBins<<<blocks,threads>>>(d_codes,
        thrust::raw_pointer_cast(d_binSizes.data()), N, mask);
    cudaEventRecord(eh1, 0);

    // scan → offsets
    thrust::exclusive_scan(d_binSizes.begin(), d_binSizes.end(),
                           d_binOffsets.begin());
    d_binOffsets[numBins] = N;   // sentinel
    cudaEventRecord(es1, 0);

    // Pass-2: scatter (buildDestMap + gatherCopy)
    thrust::device_vector<int> d_binCursor(numBins, 0);
    thrust::device_vector<int> d_srcIndexForDest(N, -1);

    cudaEventRecord(esc0, 0);
    buildDestMapKernel<<<blocks, threads>>>(
        d_codes,
        thrust::raw_pointer_cast(d_binOffsets.data()),
        thrust::raw_pointer_cast(d_binCursor.data()),
        thrust::raw_pointer_cast(d_srcIndexForDest.data()),
        N, mask);

    gatherCopyKernel<<<blocks, threads>>>(
        d_in, d_tmp,
        thrust::raw_pointer_cast(d_srcIndexForDest.data()),
        N);
    cudaEventRecord(esc1, 0);

    // 取回 offsets/sizes 到主机用于 per-bin 调度
    std::vector<int> h_offsets(numBins+1);
    std::vector<int> h_sizes  (numBins);
    thrust::copy(d_binOffsets.begin(), d_binOffsets.end(), h_offsets.begin());
    thrust::copy(d_binSizes.begin(),   d_binSizes.end(),   h_sizes.begin());

    // Per-bin compaction（纳入 kernel-only）
    int totalOut = 0;
    cudaEventRecord(ec0, 0);
    for (int b = 0; b < numBins; ++b) {
        int off = h_offsets[b];
        int sz  = h_sizes[b];
        if (sz == 0) continue;

        Point2D* binIn  = d_tmp + off;
        Point2D* binOut = d_out + totalOut;
        int      h_cnt  = 0;

        compactOneBin(binIn, binOut, sz, threshold, h_cnt, kernelKind);
        totalOut += h_cnt;
    }
    cudaDeviceSynchronize();   // 保证所有 bin 的 kernel 都结束
    cudaEventRecord(ec1, 0);

    // ===== Kernel-only 截止到此 =====
    cudaEventRecord(k1, 0);
    cudaEventSynchronize(k1);

    // 最终输出回拷（E2E 包含）
    output.resize(totalOut);
    if (totalOut > 0) {
        cudaMemcpyAsync(output.data(), d_out,
                        totalOut * sizeof(Point2D), cudaMemcpyDeviceToHost, 0);
    }

    // E2E 结束
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);

    // ---------- Timings ----------
    cudaEventElapsedTime(&t_kernel_ms, k0, k1);
    cudaEventElapsedTime(&t_total_ms,  e0, e1);

    float t_codes=0.f, t_hist=0.f, t_scan=0.f, t_scatter=0.f, t_comp=0.f;
    cudaEventElapsedTime(&t_codes,   ecodes0, ecodes1);
    cudaEventElapsedTime(&t_hist,    eh0,     eh1);
    cudaEventElapsedTime(&t_scan,    eh1,     es1);
    cudaEventElapsedTime(&t_scatter, esc0,    esc1);
    cudaEventElapsedTime(&t_comp,    ec0,     ec1);

    std::cout << "   [Plan-A breakdown] "
              << "codes "   << t_codes
              << " ms, hist "    << t_hist
              << " ms, scan "    << t_scan
              << " ms, scatter " << t_scatter
              << " ms, compact " << t_comp << " ms\n";

    // ---------- Cleanup ----------
    cudaEventDestroy(e0);     cudaEventDestroy(e1);
    cudaEventDestroy(k0);     cudaEventDestroy(k1);
    cudaEventDestroy(ecodes0);cudaEventDestroy(ecodes1);
    cudaEventDestroy(eh0);    cudaEventDestroy(eh1);
    cudaEventDestroy(es1);
    cudaEventDestroy(esc0);   cudaEventDestroy(esc1);
    cudaEventDestroy(ec0);    cudaEventDestroy(ec1);

    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
    cudaFree(d_codes);
}

void testPlanA_breakdown(const std::vector<Point2D>& input,
                         float                       threshold,
                         int                         kBits,
                         BinKernel                   kernelKind,
                         BreakdownPlanA&             out,
                         std::vector<Point2D>*       host_output /*opt*/ )
{
    const int N       = (int)input.size();
    const int numBins = 1 << kBits;
    const int mask    = numBins - 1;

    // 事件
    cudaEvent_t e0,e1,k0,k1, ecodes0,ecodes1, eh0,eh1, es1, esc0,esc1, ec0,ec1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    cudaEventCreate(&ecodes0); cudaEventCreate(&ecodes1);
    cudaEventCreate(&eh0); cudaEventCreate(&eh1);
    cudaEventCreate(&es1);
    cudaEventCreate(&esc0); cudaEventCreate(&esc1);
    cudaEventCreate(&ec0); cudaEventCreate(&ec1);

    // 缓冲
    Point2D *d_in=nullptr, *d_tmp=nullptr, *d_out=nullptr; uint32_t* d_codes=nullptr;
    cudaMalloc(&d_in,  N*sizeof(Point2D));
    cudaMalloc(&d_tmp, N*sizeof(Point2D));
    cudaMalloc(&d_out, N*sizeof(Point2D));
    cudaMalloc(&d_codes, N*sizeof(uint32_t));

    // E2E: H2D
    cudaEventRecord(e0, 0);
    cudaMemcpyAsync(d_in, input.data(), N*sizeof(Point2D), cudaMemcpyHostToDevice, 0);

    thrust::device_vector<int> d_binSizes(numBins, 0), d_binOffsets(numBins+1, 0);

    //const int threads=256, blocks=(N+threads-1)/threads;
    const int threads = g_block_size;
    const int blocks  = (N + threads - 1) / threads;

    // Kernel-only from codes
    cudaEventRecord(k0, 0);

    // Pass-0: codes
    cudaEventRecord(ecodes0, 0);
    genMortonCodesKernel<<<blocks, threads>>>(d_in, d_codes, N);
    cudaEventRecord(ecodes1, 0);

    // Pass-1: histogram
    cudaEventRecord(eh0, 0);
    histogramBins<<<blocks,threads>>>(d_codes, thrust::raw_pointer_cast(d_binSizes.data()), N, mask);
    cudaEventRecord(eh1, 0);

    // scan → offsets
    thrust::exclusive_scan(d_binSizes.begin(), d_binSizes.end(), d_binOffsets.begin());
    d_binOffsets[numBins] = N;
    cudaEventRecord(es1, 0);

    // // Pass-2: scatter (map + gather)
    // thrust::device_vector<int> d_binCursor(numBins, 0);
    // thrust::device_vector<int> d_srcIndexForDest(N, -1);
    // cudaEventRecord(esc0, 0);
    // buildDestMapKernel<<<blocks, threads>>>(
    //     d_codes,
    //     thrust::raw_pointer_cast(d_binOffsets.data()),
    //     thrust::raw_pointer_cast(d_binCursor.data()),
    //     thrust::raw_pointer_cast(d_srcIndexForDest.data()),
    //     N, mask);
    // gatherCopyKernel<<<blocks, threads>>>(
    //     d_in, d_tmp,
    //     thrust::raw_pointer_cast(d_srcIndexForDest.data()),
    //     N);
    // cudaEventRecord(esc1, 0);

    // Pass-2: scatter / gather  — 由 g_planA_use_gather 控制
    cudaEventRecord(esc0, 0);

    if (g_planA_use_gather) {
        // ===== baseline: buildDestMap + gatherCopy（顺序写，合并好）=====
        thrust::device_vector<int> d_binCursor(numBins, 0);
        thrust::device_vector<int> d_srcIndexForDest(N, -1);

        buildDestMapKernel<<<blocks, threads>>>(
            d_codes,
            thrust::raw_pointer_cast(d_binOffsets.data()),
            thrust::raw_pointer_cast(d_binCursor.data()),
            thrust::raw_pointer_cast(d_srcIndexForDest.data()),
            N, mask);

        gatherCopyKernel<<<blocks, threads>>>(
            d_in, d_tmp,
            thrust::raw_pointer_cast(d_srcIndexForDest.data()),
            N);
    } else {
        // ===== no-gather ablation: 直接 scatter 到各 bin 的连续段 =====
        // 用 offsets 作为每个 bin 的起始写指针，拷到 d_tmp
        thrust::device_vector<int> d_binCursor = d_binOffsets;  // 起始 = exclusive offsets
        scatterToBins<<<blocks, threads>>>(
            d_in, d_tmp, d_codes,
            thrust::raw_pointer_cast(d_binCursor.data()),
            N, mask);
    }

    cudaEventRecord(esc1, 0);



    // 元数据回主机，用于 per-bin 调度
    std::vector<int> h_offsets(numBins+1), h_sizes(numBins);
    thrust::copy(d_binOffsets.begin(), d_binOffsets.end(), h_offsets.begin());
    thrust::copy(d_binSizes.begin(),   d_binSizes.end(),   h_sizes.begin());

    // Per-bin compaction（纳入 kernel-only）
    int totalOut = 0;
    cudaEventRecord(ec0, 0);
    for (int b=0; b<numBins; ++b) {
        int off=h_offsets[b], sz=h_sizes[b];
        if (sz==0) continue;
        int h_cnt=0;
        compactOneBin(d_tmp+off, d_out+totalOut, sz, threshold, h_cnt, kernelKind);
        totalOut += h_cnt;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(ec1, 0);

    // Kernel-only 截止
    cudaEventRecord(k1, 0);
    cudaEventSynchronize(k1);

    // D2H（计入 E2E）
    if (host_output) {
        host_output->resize(totalOut);
        if (totalOut>0) {
            cudaMemcpyAsync(host_output->data(), d_out, totalOut*sizeof(Point2D), cudaMemcpyDeviceToHost, 0);
        }
    }
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);

    // 回填
    out.total_out = totalOut;
    cudaEventElapsedTime(&out.kernel_ms, k0, k1);
    cudaEventElapsedTime(&out.e2e_ms,    e0, e1);
    cudaEventElapsedTime(&out.codes_ms,  ecodes0, ecodes1);
    cudaEventElapsedTime(&out.hist_ms,   eh0, eh1);
    cudaEventElapsedTime(&out.scan_ms,   eh1, es1);
    cudaEventElapsedTime(&out.scatter_ms,esc0, esc1);
    cudaEventElapsedTime(&out.compact_ms,ec0, ec1);

    // 清理
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(k0); cudaEventDestroy(k1);
    cudaEventDestroy(ecodes0); cudaEventDestroy(ecodes1);
    cudaEventDestroy(eh0); cudaEventDestroy(eh1);
    cudaEventDestroy(es1);
    cudaEventDestroy(esc0); cudaEventDestroy(esc1);
    cudaEventDestroy(ec0); cudaEventDestroy(ec1);
    cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_codes);
}

// use 自建流 & 事件，避免默认流的隐式同步
// void testBinGPUCompaction_partition(const std::vector<Point2D>& input,
//                                     float                       threshold,
//                                     int                         kBits,
//                                     std::vector<Point2D>&       output,
//                                     float&                      t_kernel_ms,
//                                     float&                      t_total_ms,
//                                     BinKernel                   kernelKind)
// {
//     const int    N       = static_cast<int>(input.size());
//     const int    numBins = 1 << kBits;
//     const int    mask    = numBins - 1;
//     const size_t inBytes = N * sizeof(Point2D);

//     // ---- Device ----
//     Point2D *d_in=nullptr, *d_tmp=nullptr, *d_out=nullptr;
//     uint32_t* d_codes=nullptr;
//     CUDA_CHECK(cudaMalloc(&d_in,    inBytes));
//     CUDA_CHECK(cudaMalloc(&d_tmp,   inBytes));
//     CUDA_CHECK(cudaMalloc(&d_out,   inBytes));
//     CUDA_CHECK(cudaMalloc(&d_codes, N * sizeof(uint32_t)));

//     thrust::device_vector<int> d_binSizes  (numBins,   0);
//     thrust::device_vector<int> d_binOffsets(numBins+1, 0);

//     // ---- Stream / events ----
//     cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
//     cudaEvent_t e0,e1,k0,k1, ec0,ec1, eh0,eh1, es1, esc0,esc1;
//     CUDA_CHECK(cudaEventCreate(&e0));   CUDA_CHECK(cudaEventCreate(&e1));
//     CUDA_CHECK(cudaEventCreate(&k0));   CUDA_CHECK(cudaEventCreate(&k1));
//     CUDA_CHECK(cudaEventCreate(&ec0));  CUDA_CHECK(cudaEventCreate(&ec1));
//     CUDA_CHECK(cudaEventCreate(&eh0));  CUDA_CHECK(cudaEventCreate(&eh1));
//     CUDA_CHECK(cudaEventCreate(&es1));
//     CUDA_CHECK(cudaEventCreate(&esc0)); CUDA_CHECK(cudaEventCreate(&esc1));

//     // 注册 input 为 pinned，并 H2D
//     CUDA_CHECK(cudaHostRegister((void*)input.data(), inBytes, cudaHostRegisterDefault));
//     CUDA_CHECK(cudaEventRecord(e0, s));
//     CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(), inBytes, cudaMemcpyHostToDevice, s));

//     const int threads = 256;
//     const int blocks  = (N + threads - 1) / threads;

//     // ===== kernel-only from codes =====
//     CUDA_CHECK(cudaEventRecord(k0, s));

//     CUDA_CHECK(cudaEventRecord(ec0, s));
//     genMortonCodesKernel<<<blocks, threads, 0, s>>>(d_in, d_codes, N);
//     CUDA_CHECK(cudaEventRecord(ec1, s));

//     CUDA_CHECK(cudaEventRecord(eh0, s));
//     histogramBins<<<blocks, threads, 0, s>>>(d_codes,
//         thrust::raw_pointer_cast(d_binSizes.data()), N, mask);
//     CUDA_CHECK(cudaEventRecord(eh1, s));

//     thrust::exclusive_scan(d_binSizes.begin(), d_binSizes.end(),
//                            d_binOffsets.begin());
//     {   // sentinel d_binOffsets[numBins] = N
//         int N_host = N;
//         CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_binOffsets.data())+numBins,
//                               &N_host, sizeof(int), cudaMemcpyHostToDevice));
//     }
//     CUDA_CHECK(cudaEventRecord(es1, s));

//     // 回到 scatterToBins（你这块卡更快）
//     thrust::device_vector<int> d_binCursor = d_binOffsets;
//     CUDA_CHECK(cudaEventRecord(esc0, s));
//     scatterToBins<<<blocks, threads, 0, s>>>(d_in, d_tmp, d_codes,
//         thrust::raw_pointer_cast(d_binCursor.data()), N, mask);
//     CUDA_CHECK(cudaEventRecord(esc1, s));

//     // 元数据回主机
//     std::vector<int> h_offsets(numBins+1);
//     std::vector<int> h_sizes  (numBins);
//     thrust::copy(d_binOffsets.begin(), d_binOffsets.end(), h_offsets.begin());
//     thrust::copy(d_binSizes.begin(),   d_binSizes.end(),   h_sizes.begin());

//     // per-bin compaction
//     int totalOut = 0;
//     for (int b = 0; b < numBins; ++b) {
//         int off = h_offsets[b], sz = h_sizes[b];
//         if (sz == 0) continue;
//         Point2D* binIn  = d_tmp + off;
//         Point2D* binOut = d_out + totalOut;
//         int      h_cnt  = 0;
//         compactOneBin(binIn, binOut, sz, threshold, h_cnt, kernelKind);
//         totalOut += h_cnt;
//     }
//     CUDA_CHECK(cudaDeviceSynchronize());
//     CUDA_CHECK(cudaEventRecord(k1, s));
//     CUDA_CHECK(cudaEventSynchronize(k1));

//     // D2H（把 output 自己注册为 pinned）
//     output.resize(totalOut);
//     if (totalOut > 0) {
//         const size_t outBytes = totalOut * sizeof(Point2D);
//         CUDA_CHECK(cudaHostRegister(output.data(), outBytes, cudaHostRegisterDefault));
//         CUDA_CHECK(cudaMemcpyAsync(output.data(), d_out, outBytes,
//                                    cudaMemcpyDeviceToHost, s));
//         CUDA_CHECK(cudaEventRecord(e1, s));
//         CUDA_CHECK(cudaEventSynchronize(e1));
//         CUDA_CHECK(cudaHostUnregister(output.data()));
//     } else {
//         CUDA_CHECK(cudaEventRecord(e1, s));
//         CUDA_CHECK(cudaEventSynchronize(e1));
//     }

//     // timings
//     CUDA_CHECK(cudaEventElapsedTime(&t_kernel_ms, k0, k1));
//     CUDA_CHECK(cudaEventElapsedTime(&t_total_ms,  e0, e1));

//     float t_codes=0.f,t_hist=0.f,t_scan=0.f,t_scatter=0.f;
//     CUDA_CHECK(cudaEventElapsedTime(&t_codes,   ec0,  ec1));
//     CUDA_CHECK(cudaEventElapsedTime(&t_hist,    eh0,  eh1));
//     CUDA_CHECK(cudaEventElapsedTime(&t_scan,    eh1,  es1));
//     CUDA_CHECK(cudaEventElapsedTime(&t_scatter, esc0, esc1));
//     std::cout << "   [Plan-A breakdown] "
//               << "codes "   << t_codes
//               << " ms, hist "    << t_hist
//               << " ms, scan "    << t_scan
//               << " ms, scatter " << t_scatter << " ms\n";

//     // cleanup
//     CUDA_CHECK(cudaHostUnregister((void*)input.data()));
//     cudaEventDestroy(e0);  cudaEventDestroy(e1);
//     cudaEventDestroy(k0);  cudaEventDestroy(k1);
//     cudaEventDestroy(ec0); cudaEventDestroy(ec1);
//     cudaEventDestroy(eh0); cudaEventDestroy(eh1);
//     cudaEventDestroy(es1);
//     cudaEventDestroy(esc0); cudaEventDestroy(esc1);
//     cudaStreamDestroy(s);
//     cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out); cudaFree(d_codes);
// }





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


// ────────────────────────────────────────────────────────────────
// Device Morton helpers (32-bit, interleave x/y lower 16 bits)
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__ uint32_t part1by1(uint32_t x) {
    x &= 0x0000ffffu;
    x = (x | (x << 8))  & 0x00ff00ffu;
    x = (x | (x << 4))  & 0x0f0f0f0fu;
    x = (x | (x << 2))  & 0x33333333u;
    x = (x | (x << 1))  & 0x55555555u;
    return x;
}
__device__ __forceinline__ uint32_t morton2D_encode_dev(uint32_t x, uint32_t y) {
    return part1by1(x) | (part1by1(y) << 1);
}

// ────────────────────────────────────────────────────────────────
// Pass-0: 由点直接生成 Morton codes（避免 CPU 侧 codes+H2D）
// ────────────────────────────────────────────────────────────────
__global__ void genMortonCodesKernel(const Point2D* __restrict__ in,
                                     uint32_t*      __restrict__ codes,
                                     int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // 注：按你现有数据，x/y 是 float，这里与原 host 代码一致做 (int) 截断
    int xi = static_cast<int>(in[i].x);
    int yi = static_cast<int>(in[i].y);
    codes[i] = morton2D_encode_dev((uint32_t)xi, (uint32_t)yi);
}

// ────────────────────────────────────────────────────────────────
// Pass-1: 统计“每个 bin 的有效元素数”（带阈值判定）
// ────────────────────────────────────────────────────────────────
__global__ void countValidPerBin_fromCodes(const Point2D*  __restrict__ in,
                                           const uint32_t* __restrict__ codes,
                                           int*            __restrict__ binValid,
                                           int N, int mask, float thr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (in[i].temp > thr) {
        int b = static_cast<int>(codes[i] & (uint32_t)mask);
        atomicAdd(&binValid[b], 1);
    }
}

// ────────────────────────────────────────────────────────────────
// Pass-2: 按桶写出（per-bin atomic），保证每桶连续
// ────────────────────────────────────────────────────────────────
__global__ void writePerBinAtomic_fromCodes(const Point2D*  __restrict__ in,
                                            const uint32_t* __restrict__ codes,
                                            Point2D*        __restrict__ out,
                                            const int*      __restrict__ binBase,
                                            int*            __restrict__ binFill,
                                            int N, int mask, float thr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const Point2D p = in[i];
    if (p.temp > thr) {
        int b   = static_cast<int>(codes[i] & (uint32_t)mask);
        int off = atomicAdd(&binFill[b], 1);
        out[binBase[b] + off] = p;
    }
}