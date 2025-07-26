/**
 * @file stream_compaction_bin.h
 * @brief Implementation of bin-based stream compaction extension.
 *        This version introduces Morton code binning (based on k lower bits),
 *        allowing per-bin GPU compaction to improve memory locality.
 * 
 *        Functions include bin offset computation and wrapper functions to run
 *        bin-aware benchmarks under various configurations.
 * 
 *        This header is designed to accompany the bin-optimized implementation
 *        in stream_compaction_bin.cu and works alongside the naive version.
 * 
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-07-14
 */

#pragma once

#include <cstdint>
#include <string>
#include "common.h"  // for Point2D
#include "bin_kernel.h"

/**
 * @brief Compute bin offsets and sizes based on k lower bits of Morton codes.
 * 
 * @param mortonCodes Input Morton code array (device pointer)
 * @param numPoints Total number of points
 * @param k Number of lower bits used to determine bin ID
 * @param binOffsets Output array of bin starting positions (device pointer)
 * @param binSizes Output array of bin sizes (device pointer)
 */
void computeBinOffsets(const uint32_t* mortonCodes, int numPoints, int k,
                       int* binOffsets, int* binSizes);

/**
 * @brief Perform compaction per bin using shared memory.
 * 
 * @param d_in Input array of Point2D (device pointer)
 * @param d_out Output array (device pointer)
 * @param mortonCodes Morton codes corresponding to input points (device pointer)
 * @param numPoints Total number of points
 * @param k Number of lower bits used to compute bin ID
 * @param d_outCount Output: total number of valid points after compaction (device pointer)
 */
void compactWithBinsGPU(const Point2D* d_in, Point2D* d_out,
                        const uint32_t* mortonCodes,
                        int numPoints, int kBits,
                        int* d_outCount);

/**
 * @brief Benchmark wrapper to run bin-based compaction and record timing and error.
 *        Used in benchmark_runner.cu.
 * 
 * @param size Input size
 * @param blockSize CUDA thread block size
 * @param precision "float" or "double"
 * @param time_ms Output: measured GPU execution time
 * @param error Output: error compared to reference (optional)
 */
void runBitmaskBenchmarkWithBins(int size, int blockSize,
                                 const std::string& precision,
                                 float& time_ms, float& error);


void testBinGPUCompaction(const std::vector<Point2D>& input,
                          float                       threshold,
                          int                         kBits,
                          std::vector<Point2D>&       output);


/* === Plan-B: single-pass atomic version ================================= */
// void testBinGPUCompaction_atomic(const std::vector<Point2D>& input,
//                                  float                       threshold,
//                                  int                         kBits,
//                                  std::vector<Point2D>&       output,
//                                  float&                      t_kernel_ms,
//                                  float&                      t_total_ms);
void testBinGPUCompaction_atomic(const std::vector<Point2D>&, float, int,
                                 std::vector<Point2D>&, float&, float&);
/**
 * @brief Single-pass bin compaction using atomics.
 *        Measures both “kernel-only” time (t_kernel_ms)
 *        and “end-to-end” time including copies (t_total_ms). *
 * @param input       Host-side input points
 * @param threshold   Temperature threshold
 * @param kBits       Number of low bits used for bin ID
 * @param output      Host-side compacted output points (filled by this func)
 * @param t_kernel_ms Return: pure-kernel time   (ms)
 * @param t_total_ms  Return: total GPU time incl. H2D / D2H (ms)
 */
// __global__ void compactBinAtomic(const Point2D* __restrict__ in,
//                                  Point2D*       __restrict__ out,
//                                  const uint32_t*__restrict__ codes,
//                                  int*           binCursor,
//                                  int            N,
//                                  int            mask,
//                                  float          threshold);

// __global__ void compactBinAtomic(const Point2D* in,
//                                  Point2D*       out,
//                                  int*           binCursor,
//                                  const int*     binOffsets,  // 新增
//                                  const uint32_t* codes,
//                                  int            N,
//                                  int            mask,
//                                  float          thr);

__global__ void compactBinAtomic(const Point2D*, Point2D*, int*, const uint32_t*,
                                 int, int, float);

                                 

/**
 * @brief Build per-bin histogram (sizes) from Morton codes.
 *
 * @param codes      Device array of Morton codes
 * @param binSizes   Device array (len = numBins) — histogram output, init to 0
 * @param N          Number of points
 * @param mask       (1<<kBits)-1 bit-mask to extract binID
 */
__global__ void histogramBins(const uint32_t* codes,
                              int*           binSizes,
                              int            N,
                              int            mask);

/**
 * @brief Scatter points so that each bin occupies a contiguous slice.
 *
 * @param in         Input points  (device)
 * @param tmp        Scatter buffer (device, len = N)
 * @param codes      Morton codes   (device)
 * @param binCursor  Device array, init with binOffsets, atomically incremented
 * @param N          Number of points
 * @param mask       (1<<kBits)-1 bit-mask to extract binID
 */
__global__ void scatterToBins(const Point2D* in,
                              Point2D*       tmp,
                              const uint32_t* codes,
                              int*           binCursor,
                              int            N,
                              int            mask);



/**
 * @brief Plan-A 版本：直方图 → scan → scatter → per-bin compaction（共享内存或其他）。
 *
 * @param input        Host-side input vector of points
 * @param threshold    Compaction threshold (e.g. temperature)
 * @param kBits        Number of low Morton bits used for bin ID (numBins = 1<<kBits)
 * @param output       Host-side vector to receive compacted points
 * @param t_kernel_ms  Return — kernel-only elapsed time (ms)
 * @param t_total_ms   Return — total GPU time incl. H2D / D2H (ms)
 */
// void testBinGPUCompaction_partition(const std::vector<Point2D>& input,
//                                     float                       threshold,
//                                     int                         kBits,
//                                     std::vector<Point2D>&       output,
//                                     float&                      t_kernel_ms,
//                                     float&                      t_total_ms
//                                     BinKernel kernelKind);

void testBinGPUCompaction_partition(const std::vector<Point2D>& input,
                                    float  threshold,
                                    int    kBits,
                                    std::vector<Point2D>& output,
                                    float& t_kernel_ms,
                                    float& t_total_ms,
                                    BinKernel kernelKind);        // ← 第 7 个参数




void compactWarpGPU(const Point2D* d_in,
                    Point2D*       d_out,
                    int            N,
                    float          threshold,
                    int&           h_outCount);


void compactOneBin(Point2D* d_in,      // bin input (contiguous)
                   Point2D* d_out,     // bin output base
                   int      N,         // elements in this bin
                   float    threshold, // predicate value
                   int&     h_outCnt,  // host-side result count
                   BinKernel kind);     // strategy