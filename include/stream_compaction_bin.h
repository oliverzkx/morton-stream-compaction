/**
 * @file stream_compaction_bin.h
 * @brief Interfaces for Morton-code, bin-aware stream-compaction.
 *
 * The routines declared here implement a “partition-then-compact” pipeline:
 *   1.  Build a per-bin histogram from the lowest @p kBits of each Morton code.
 *   2.  Prefix-scan the histogram to obtain per-bin offsets.
 *   3.  Scatter points so each bin forms a contiguous slice in memory.
 *   4.  Compact every bin with a chosen kernel variant
 *       (shared memory, warp intrinsics, bitmask, or atomic fallback).
 *
 * Host-side helpers are provided for benchmarking, autotuning, and
 * measuring end-to-end GPU time.  Device-side kernels are declared so that
 * downstream translation units may launch them directly.
 *
 * @author  Kaixiang Zou
 * @version 1.3
 * @date    2025-07-26
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"      // Point2D definition
#include "bin_kernel.h"  // BinKernel enum

// ────────────────────────────────────────────────────────────────
// Host-side helpers (declarations)
// ────────────────────────────────────────────────────────────────

/**
 * @brief Compute per-bin offsets and sizes from Morton codes.
 *
 * @param mortonCodes Device array of Morton codes (length = @p numPoints).
 * @param numPoints   Total number of points.
 * @param kBits       Number of low bits that define the bin ID (bins = 2^kBits).
 * @param binOffsets  Device array (length = 2^kBits) that receives start indices.
 * @param binSizes    Device array (length = 2^kBits) that receives element counts.
 */
void computeBinOffsets(const uint32_t* mortonCodes,
                       int             numPoints,
                       int             kBits,
                       int*            binOffsets,
                       int*            binSizes);

/**
 * @brief End-to-end bin pipeline (histogram → scan → scatter → per-bin compaction).
 *
 * @param d_in        Device pointer to input points.
 * @param d_out       Device pointer to output buffer.
 * @param mortonCodes Device array of Morton codes corresponding to @p d_in.
 * @param numPoints   Total number of input points.
 * @param kBits       Number of low Morton bits used for the bin ID.
 * @param d_outCount  Device pointer receiving total number of valid points.
 */
void compactWithBinsGPU(const Point2D*  d_in,
                        Point2D*        d_out,
                        const uint32_t* mortonCodes,
                        int             numPoints,
                        int             kBits,
                        int*            d_outCount);

/**
 * @brief Run a bin-aware *bitmask* compaction benchmark and capture timing / error.
 *
 * @param size      Number of input elements.
 * @param blockSize CUDA thread-block size.
 * @param precision "float" or "double".
 * @param time_ms   Returns measured GPU time (ms).
 * @param error     Returns RMS error against the CPU reference.
 */
void runBitmaskBenchmarkWithBins(int               size,
                                 int               blockSize,
                                 const std::string& precision,
                                 float&            time_ms,
                                 float&            error);

/**
 * @brief Convenience wrapper: complete bin pipeline, returns compacted output.
 *
 * @param input     Host-side input points.
 * @param threshold Predicate threshold for isHot.
 * @param kBits     Number of low Morton bits used for the bin ID.
 * @param output    Host-side vector receiving compacted points.
 */
void testBinGPUCompaction(const std::vector<Point2D>& input,
                          float                       threshold,
                          int                         kBits,
                          std::vector<Point2D>&       output);

/**
 * @brief Single-pass atomic version for quick prototyping / fallback.
 *
 * @param input        Host-side input points.
 * @param threshold    Predicate threshold for isHot.
 * @param kBits        Number of low Morton bits (bins = 2^kBits).
 * @param output       Host-side vector receiving compacted points.
 * @param t_kernel_ms  Returns kernel-only time (ms).
 * @param t_total_ms   Returns total GPU time incl. H2D / D2H (ms).
 */
void testBinGPUCompaction_atomic(const std::vector<Point2D>& input,
                                 float                       threshold,
                                 int                         kBits,
                                 std::vector<Point2D>&       output,
                                 float&                      t_kernel_ms,
                                 float&                      t_total_ms);

/**
 * @brief Plan-A pipeline wrapper that lets the caller choose a ::BinKernel variant.
 *
 * @param input        Host-side input points.
 * @param threshold    Predicate threshold for isHot.
 * @param kBits        Number of low Morton bits (bins = 2^kBits).
 * @param output       Host-side vector receiving compacted points.
 * @param t_kernel_ms  Returns kernel-only time (ms).
 * @param t_total_ms   Returns total GPU time incl. H2D / D2H (ms).
 * @param kernelKind   Kernel strategy to use (shared / warp / bitmask / auto).
 */
void testBinGPUCompaction_partition(const std::vector<Point2D>& input,
                                    float                       threshold,
                                    int                         kBits,
                                    std::vector<Point2D>&       output,
                                    float&                      t_kernel_ms,
                                    float&                      t_total_ms,
                                    BinKernel                   kernelKind);

// ────────────────────────────────────────────────────────────────
// Stand-alone helpers and micro-benchmarks
// ────────────────────────────────────────────────────────────────

/**
 * @brief Warp-level compaction of a single contiguous bin (micro-benchmark).
 *
 * @param d_in       Device pointer to bin input (contiguous).
 * @param d_out      Device pointer to output buffer.
 * @param N          Elements in the bin.
 * @param threshold  Predicate threshold for isHot.
 * @param h_outCount Returns number of valid points (host-side).
 */
void compactWarpGPU(const Point2D* d_in,
                    Point2D*       d_out,
                    int            N,
                    float          threshold,
                    int&           h_outCount);

/**
 * @brief Dispatch a chosen ::BinKernel variant to compact one bin.
 *
 * @param d_in     Device pointer to contiguous bin input.
 * @param d_out    Device pointer to output buffer base.
 * @param N        Elements in the bin.
 * @param threshold Predicate threshold for isHot.
 * @param h_outCnt  Returns number of valid points (host-side).
 * @param kind      Kernel strategy (shared / warp / bitmask / auto).
 */
void compactOneBin(Point2D*  d_in,
                   Point2D*  d_out,
                   int       N,
                   float     threshold,
                   int&      h_outCnt,
                   BinKernel kind);

// ────────────────────────────────────────────────────────────────
// Device-side kernels
// ────────────────────────────────────────────────────────────────

/**
 * @brief Build a per-bin histogram (bin sizes) from Morton codes.
 *
 * @param codes     Device array of Morton codes.
 * @param binSizes  Device array (length = numBins) initialised to zero.
 * @param N         Total number of points.
 * @param mask      (1 << kBits) − 1 bit-mask to extract the bin ID.
 */
__global__ void histogramBins(const uint32_t* codes,
                              int*            binSizes,
                              int             N,
                              int             mask);

/**
 * @brief Scatter points so that each bin occupies a contiguous slice.
 *
 * @param in         Input points  (device).
 * @param tmp        Scratch buffer (device, length = N).
 * @param codes      Morton codes  (device).
 * @param binCursor  Device array initialised with binOffsets; atomically
 *                   incremented by each thread when it places its point.
 * @param N          Total number of points.
 * @param mask       (1 << kBits) − 1 bit-mask to extract the bin ID.
 */
__global__ void scatterToBins(const Point2D*  in,
                              Point2D*        tmp,
                              const uint32_t* codes,
                              int*            binCursor,
                              int             N,
                              int             mask);

/**
 * @brief Atomic fallback kernel that compacts one bin in a single pass.
 *
 * @param in         Input points (device).
 * @param out        Output buffer (device).
 * @param binCursor  Device counter pointing to the next free slot.
 * @param codes      Morton codes (device).
 * @param N          Elements in the bin.
 * @param mask       (1 << kBits) − 1 bit-mask to extract the bin ID.
 * @param threshold  Predicate threshold for isHot.
 */
__global__ void compactBinAtomic(const Point2D*  in,
                                 Point2D*        out,
                                 int*            binCursor,
                                 const uint32_t* codes,
                                 int             N,
                                 int             mask,
                                 float           threshold);
