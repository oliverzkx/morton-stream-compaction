#pragma once
#include "common.h" 

// Device-side predicate used in CUDA kernel
__device__ inline bool isHotPredicateDevice(const Point2D& p);

// Naïve GPU kernel for stream compaction using a global atomic counter
__global__ void streamCompactNaive(const Point2D* in, Point2D* out, int N, int* d_counter);

// Host wrapper for launching the naïve GPU compaction kernel
void compactNaiveGPU(const Point2D* d_in, Point2D* d_out, int N, int& h_outCount);

__global__ void streamCompactShared(const Point2D* in, Point2D* out, int N,float threshold, int* block_counts);

void compactSharedGPU(const Point2D* d_in, Point2D* d_out, int N, float threshold, int& h_outCount);

// stream_compaction.cuh
void testNaiveGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output);
void testSharedGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output);
