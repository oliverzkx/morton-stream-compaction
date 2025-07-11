/**
 * @file benchmark_utils.h
 * @brief Interface for GPU benchmark utility functions.
 *        Provides an abstraction to run stream compaction benchmarks with specified
 *        input size, block size, and precision mode, and records runtime and error metrics.
 * 
 *        This file is intended to support batch testing and data collection for
 *        performance and accuracy evaluation.
 * 
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-07-11
 */
#pragma once

#include "common.h"

void printPointsComparison(const std::vector<Point2D>& gpu_output,
                           const std::vector<Point2D>& cpu_baseline);

void runBitmaskBenchmark(int size, int blockSize, const std::string& precision, float& time_ms, float& error);

bool compareMorton(const Point2D& a, const Point2D& b);