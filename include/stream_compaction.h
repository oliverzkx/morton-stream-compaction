/**
 * @file stream_compaction.h
 * @brief Naïve, shared-memory, warp-shuffle, and bitmask variants of
 *        stream compaction for a single Morton-curve point array.
 *
 * This header groups together:
 *   • A device-side predicate for “hot” points (single & double precision).  
 *   • Four GPU kernel families (naïve / shared / warp / bitmask).  
 *   • Host-side wrappers that launch those kernels and return counts.  
 *   • Stand-alone test helpers that measure correctness and timing.  
 *   • Float and double specialisations of the bitmask pipeline.  
 *
 * Kernels are implemented in the accompanying <code>.cu</code> files.  
 * Host wrappers allocate device buffers, launch kernels, and copy results
 * back for verification / benchmarking.
 *
 * Author:  Kaixiang Zou  
 * Version: 1.2  
 * Date:    2025-07-26
 */
#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>          // cudaSurfaceObject_t
#include "common.h"                // Point2D / Point2D_double
#include "bin_kernel.h"            // BinKernel enum (shared across modules)

// ────────────────────────────────────────────────────────────────
// Global configuration
// ────────────────────────────────────────────────────────────────

/** @brief Device-resident threshold used by all isHot predicates. */
extern float d_threshold;

// ────────────────────────────────────────────────────────────────
// Device-side predicates (inlined for both precisions)
// ────────────────────────────────────────────────────────────────

/**
 * @brief Return <tt>true</tt> if the point’s temperature exceeds ::d_threshold.
 *
 * @param p Point sample (single precision).
 */
__device__ inline bool isHotPredicateDevice(const Point2D& p);

/**
 * @brief Double-precision overload of ::isHotPredicateDevice.
 *
 * @param p Point sample (double precision).
 */
__device__ inline bool isHotPredicateDevice(const Point2D_double& p);

// ────────────────────────────────────────────────────────────────
// Naïve implementation (global atomic counter)
// ────────────────────────────────────────────────────────────────

/**
 * @brief Naïve compaction kernel; each valid thread atomically appends to @p d_counter.
 *
 * @param in        Device pointer to input points.
 * @param out       Device pointer to output buffer.
 * @param N         Number of input elements.
 * @param d_counter Device counter for the write position.
 */
__global__ void streamCompactNaive(const Point2D* in,
                                   Point2D*       out,
                                   int            N,
                                   int*           d_counter);

/**
 * @brief Host wrapper for ::streamCompactNaive.
 *
 * @param d_in       Device pointer to input points.
 * @param d_out      Device pointer to output buffer.
 * @param N          Number of input elements.
 * @param h_outCount Returns number of valid points (host-side).
 */
void compactNaiveGPU(const Point2D* d_in,
                     Point2D*       d_out,
                     int            N,
                     int&           h_outCount);

// ────────────────────────────────────────────────────────────────
// Shared-memory per-block counting
// ────────────────────────────────────────────────────────────────

/**
 * @brief Shared-memory kernel that counts valid points per block, then
 *        performs a block-level exclusive scan before global write-out.
 *
 * @param in           Device pointer to input points.
 * @param out          Device pointer to output buffer.
 * @param N            Number of input elements.
 * @param threshold    Predicate threshold (copied to shared mem for speed).
 * @param block_counts Device array (one per block) storing local counts.
 */
__global__ void streamCompactShared(const Point2D* in,
                                    Point2D*       out,
                                    int            N,
                                    float          threshold,
                                    int*           block_counts);

/**
 * @brief Host wrapper for ::streamCompactShared.
 */
void compactSharedGPU(const Point2D* d_in,
                      Point2D*       d_out,
                      int            N,
                      float          threshold,
                      int&           h_outCount);

// ────────────────────────────────────────────────────────────────
// Warp-shuffle implementation
// ────────────────────────────────────────────────────────────────

/**
 * @brief Warp-level compaction using shuffle and ballot intrinsics.
 *
 * @param d_input  Device pointer to input points.
 * @param d_output Device pointer to output buffer.
 * @param d_count  Device counter updated atomically by lane 0 of each warp.
 * @param num_points Total number of input elements.
 */
__global__ void compactPointsWarpShuffle(Point2D* d_input,
                                         Point2D* d_output,
                                         int*     d_count,
                                         int      num_points);

/** @brief Host wrapper for ::compactPointsWarpShuffle. */
void compact_points_warp(Point2D* d_input,
                         Point2D* d_output,
                         int*     d_count,
                         int      num_points);

// ────────────────────────────────────────────────────────────────
// Bitmask implementation (ballot + popc)
// ────────────────────────────────────────────────────────────────

/**
 * @brief Bitmask kernel that uses <tt>__ballot_sync</tt> & <tt>__popc</tt>.
 */
__global__ void compactPointsBitmask(const Point2D* d_input,
                                     Point2D*       d_output,
                                     int*           d_count,
                                     int            num_points);

/**
 * @brief Surface-memory variant that writes output via a 2-D CUDA surface.
 *
 * @param surface_width Width of the destination surface (pitch / sizeof(Point2D)).
 */
__global__ void compactPointsBitmaskSurface(const Point2D*        d_input,
                                            cudaSurfaceObject_t   surfaceOutput,
                                            int*                  d_count,
                                            int                   num_points,
                                            int                   surface_width);

/** @brief Host wrapper for ::compactPointsBitmask. */
void compact_points_bitmask(const Point2D* d_input,
                            Point2D*       d_output,
                            int*           d_count,
                            int            num_points);

/** @brief Host wrapper for ::compactPointsBitmaskSurface. */
void compact_points_bitmask_surface(const Point2D*       d_input,
                                    cudaSurfaceObject_t  surfaceOutput,
                                    int*                 d_count,
                                    int                  num_points,
                                    int                  surface_width);

// ────────────────────────────────────────────────────────────────
// Host-side correctness / performance tests
// ────────────────────────────────────────────────────────────────

void testNaiveGPUCompaction(const std::vector<Point2D>& input,
                            float                       threshold,
                            std::vector<Point2D>&       output);

void testSharedGPUCompaction(const std::vector<Point2D>& input,
                             float                       threshold,
                             std::vector<Point2D>&       output);

void testWarpGPUCompaction(const std::vector<Point2D>& input,
                           float                       threshold,
                           std::vector<Point2D>&       output);

void testBitmaskGPUCompaction(const std::vector<Point2D>& input,
                              float                       threshold,
                              std::vector<Point2D>&       output);

void testBitmaskSurfaceGPUCompaction(const std::vector<Point2D>& input,
                                     float                       threshold,
                                     int                         surface_width);

// ────────────────────────────────────────────────────────────────
// Bitmask pipeline (float / double specialisations)
// ────────────────────────────────────────────────────────────────

/**
 * @brief Complete float-precision bitmask pipeline; returns elapsed time.
 *
 * @param input     Host-side input points.
 * @param threshold Predicate threshold.
 * @param blockSize CUDA thread-block size.
 * @param time_ms   Returns GPU time (ms).
 * @param output    Host-side vector receiving compacted points.
 */
void bitmask_stream_compaction_gpu_float(const std::vector<Point2D>& input,
                                         float                       threshold,
                                         int                         blockSize,
                                         float&                      time_ms,
                                         std::vector<Point2D>&       output);

/**
 * @brief Complete double-precision bitmask pipeline; returns elapsed time.
 */
void bitmask_stream_compaction_gpu_double(const std::vector<Point2D_double>& input,
                                          double                             threshold,
                                          int                                blockSize,
                                          float&                             time_ms,
                                          std::vector<Point2D_double>&       output);

/**
 * @brief Double-precision bitmask kernel.
 */
__global__ void compact_points_bitmask_double(const Point2D_double* d_input,
                                              Point2D_double*       d_output,
                                              int*                  d_count,
                                              int                   num_points);

// (Optional) Future inline overload
// __device__ bool isHotPredicateDevice(Point2D_double pt);
