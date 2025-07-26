/**
 * @file benchmark_utils.h
 * @brief GPU benchmark utility interfaces.
 *
 * This header provides helper functions to facilitate performance
 * benchmarking and accuracy checking of different stream-compaction kernels.
 *
 * It centralizes commonly-used routines such as result comparison,
 * automated batch execution, and Morton-code-based ordering utilities.
 *
 * @author Kaixiang Zou
 * @version 1.1
 * @date 2025-07-26
 */
#pragma once

#include "common.h"

/**
 * @brief Print a side-by-side comparison between GPU output and CPU baseline.
 *
 * @param gpu_output Vector of points computed on the GPU.
 * @param cpu_baseline Vector of reference points computed on the CPU.
 */
void printPointsComparison(const std::vector<Point2D>& gpu_output,
                           const std::vector<Point2D>& cpu_baseline);

/**
 * @brief Launch a bitmask stream-compaction benchmark.
 *
 * This function allocates device buffers, runs the specified kernel,
 * measures runtime via CUDA events, and returns error metrics against the CPU baseline.
 *
 * @param size      Number of input elements to process.
 * @param blockSize CUDA thread-block size.
 * @param precision Precision mode string: "float" or "double".
 * @param time_ms   Output: measured kernel time in milliseconds.
 * @param error     Output: RMS error against CPU baseline.
 */
void runBitmaskBenchmark(int size,
                         int blockSize,
                         const std::string& precision,
                         float& time_ms,
                         float& error);

/**
 * @brief Comparator for sorting points by Morton code.
 *
 * @param a First point to compare.
 * @param b Second point to compare.
 * @return true if the Morton code of `a` precedes that of `b`.
 */
bool compareMorton(const Point2D& a, const Point2D& b);
