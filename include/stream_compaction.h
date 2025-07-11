#pragma once

#include "common.h"

/// Device-side predicate used in GPU kernels
__device__ inline bool isHotPredicateDevice(const Point2D& p);
__device__ inline bool isHotPredicateDevice(const Point2D_double& p);


/// Naive GPU kernel using global atomic counter
__global__ void streamCompactNaive(const Point2D* in, Point2D* out, int N, int* d_counter);

/// Host wrapper for launching naive compaction kernel
void compactNaiveGPU(const Point2D* d_in, Point2D* d_out, int N, int& h_outCount);

/// Shared memory GPU kernel using per-block counting
__global__ void streamCompactShared(const Point2D* in, Point2D* out, int N, float threshold, int* block_counts);

/// Host wrapper for launching shared memory compaction kernel
void compactSharedGPU(const Point2D* d_in, Point2D* d_out, int N, float threshold, int& h_outCount);

/// Warp shuffle kernel for stream compaction
__global__ void compactPointsWarpShuffle(Point2D* d_input, Point2D* d_output, int* d_count, int num_points);

/// Host wrapper for launching warp shuffle compaction kernel
void compact_points_warp(Point2D* d_input, Point2D* d_output, int* d_count, int num_points);

/// Bitmask-based compaction kernel using __ballot_sync and __popc
__global__ void compactPointsBitmask(const Point2D* d_input, Point2D* d_output, int* d_count, int num_points);

__global__ void compactPointsBitmaskSurface(
    const Point2D* d_input,
    cudaSurfaceObject_t surfaceOutput,
    int* d_count,
    int num_points,
    int surface_width
);

/// Host wrapper to launch Bitmask compaction kernel
void compact_points_bitmask(const Point2D* d_input, Point2D* d_output, int* d_count, int num_points);

void compact_points_bitmask_surface(
    const Point2D* d_input,
    cudaSurfaceObject_t surfaceOutput,
    int* d_count,
    int num_points,
    int surface_width
);

/// Test: Naive GPU compaction
void testNaiveGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output);

/// Test: Shared memory GPU compaction
void testSharedGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output);

/// Test: Warp shuffle GPU compaction (with input/output interface)
void testWarpGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output);

/// Test: Bitmask GPU compaction
void testBitmaskGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output);

void testBitmaskSurfaceGPUCompaction(const std::vector<Point2D>& input, float threshold, int surface_width);



void bitmask_stream_compaction_gpu_float(const std::vector<Point2D>& input,
                                         float threshold,
                                         int blockSize,
                                         float& time_ms,
                                         std::vector<Point2D>& output);
void bitmask_stream_compaction_gpu_double(const std::vector<Point2D_double>& input,
                                          double threshold,
                                          int blockSize,
                                          float& time_ms,
                                          std::vector<Point2D_double>& output);
__global__ void compact_points_bitmask_double(
    const Point2D_double* d_input,
    Point2D_double* d_output,
    int* d_count,
    int num_points
);

//__device__ bool isHotPredicateDevice(Point2D_double pt);

