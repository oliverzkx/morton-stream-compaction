/**
 * @file benchmark_utils.cu
 * @brief Benchmark utilities for Morton-curve stream-compaction kernels.
 *
 * The helpers here:
 *   • Generate and sort synthetic point sets.
 *   • Launch GPU compaction kernels (bitmask variant, float and double).
 *   • Time kernel execution and compute size-based error vs. a CPU baseline.
 *   • Provide lightweight visual comparison of the first few output elements.
 *
 * Only float precision is fully supported by every kernel; the double-precision
 * path is included for completeness and future extensions.
 *
 * @author  Kaixiang Zou
 * @version 1.1
 * @date    2025-07-26
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#include "benchmark_utils.h"
#include "common.h"
#include "utils.h"
#include "stream_compaction.h"

// ────────────────────────────────────────────────────────────────
// Utility helpers
// ────────────────────────────────────────────────────────────────

/**
 * @brief Compare the first few elements of GPU and CPU results.
 *
 * The two vectors are sorted by Morton code so that identical points
 * appear in the same order before printing.
 *
 * @param gpu Vector containing GPU-compacted points.
 * @param cpu Vector containing CPU-compacted points.
 */
void printPointsComparison(const std::vector<Point2D>& gpu,
                           const std::vector<Point2D>& cpu)
{
    std::cout << "\n=== Comparing first 5 points ===\n";

    // Make copies so we can sort without touching the originals
    std::vector<Point2D> sorted_gpu = gpu;
    std::vector<Point2D> sorted_cpu = cpu;
    std::sort(sorted_gpu.begin(), sorted_gpu.end(), compareMorton);
    std::sort(sorted_cpu.begin(), sorted_cpu.end(), compareMorton);

    const int count = std::min(5,
                     std::min(static_cast<int>(sorted_gpu.size()),
                              static_cast<int>(sorted_cpu.size())));

    for (int i = 0; i < count; ++i)
    {
        // Print x, y, and temperature fields side-by-side
        const Point2D& g = sorted_gpu[i];
        const Point2D& c = sorted_cpu[i];
        std::cout << "GPU[" << i << "]: (x=" << g.x << ", y=" << g.y
                  << ", temp=" << g.temp << ")  |  "
                  << "CPU[" << i << "]: (x=" << c.x << ", y=" << c.y
                  << ", temp=" << c.temp << ")\n";
    }

    std::cout << "GPU output size = " << gpu.size()
              << ", CPU baseline size = " << cpu.size() << '\n';
}

/**
 * @brief Ordering predicate used to sort points by their Morton code.
 */
bool compareMorton(const Point2D& a, const Point2D& b)
{
    return morton2D_encode(static_cast<unsigned int>(a.x),
                           static_cast<unsigned int>(a.y))
         < morton2D_encode(static_cast<unsigned int>(b.x),
                           static_cast<unsigned int>(b.y));
}

// ────────────────────────────────────────────────────────────────
// Benchmark driver
// ────────────────────────────────────────────────────────────────

/**
 * @brief Run the bitmask compaction kernel and report timing & error.
 *
 * @param size       Total number of points to generate.
 * @param blockSize  CUDA thread-block dimension (x).
 * @param precision  `"float"` or `"double"`.
 * @param time_ms    [out] Measured GPU execution time in milliseconds.
 * @param error      [out] Relative size error compared with CPU baseline.
 */
void runBitmaskBenchmark(int               size,
                         int               blockSize,
                         const std::string& precision,
                         float&            time_ms,
                         float&            error)
{
    /* ----------------------------------------------------------------
     * 1. Generate a roughly square grid of randomised points.
     * ---------------------------------------------------------------- */
    const int rows = static_cast<int>(std::sqrt(size));
    const int cols = (size + rows - 1) / rows;

    std::vector<Point2D> input = generatePoints(rows, cols, 1.0f, 42);
    if (static_cast<int>(input.size()) > size)
        input.resize(size);                 // Trim to exact count

    // Sort by Morton code for better locality (matches GPU access)
    std::vector<Point2D> input_sorted = input;
    std::sort(input_sorted.begin(), input_sorted.end(), compareMorton);

    /* ----------------------------------------------------------------
     * 2. Prepare thresholds and result containers.
     * ---------------------------------------------------------------- */
    constexpr float  threshold_f = 25.0f;
    constexpr double threshold_d = 25.0;

    std::vector<Point2D>        cpu_baseline;  // Reference
    std::vector<Point2D>        output_f;      // GPU float result
    std::vector<Point2D_double> output_d_raw;  // GPU double (raw)
    std::vector<Point2D>        output_d;      // GPU double → float

    /* ----------------------------------------------------------------
     * 3-A. Float-precision path
     * ---------------------------------------------------------------- */
    if (precision == "float")
    {
        // Launch GPU bitmask pipeline (float)
        bitmask_stream_compaction_gpu_float(input_sorted,
                                            threshold_f,
                                            blockSize,
                                            time_ms,
                                            output_f);

        // CPU baseline for comparison
        cpu_baseline = compact_stream_cpu(input_sorted, threshold_f);

        // Compute relative size error
        const int diff = std::abs(static_cast<int>(output_f.size()) -
                                  static_cast<int>(cpu_baseline.size()));
        error = cpu_baseline.empty() ? 0.0f
                                     : static_cast<float>(diff) /
                                       static_cast<float>(cpu_baseline.size());

        printPointsComparison(output_f, cpu_baseline);
        return;
    }

    /* ----------------------------------------------------------------
     * 3-B. Double-precision path
     * ---------------------------------------------------------------- */

    // Convert Point2D → Point2D_double
    std::vector<Point2D_double> input_double;
    input_double.reserve(input_sorted.size());
    for (const auto& pt : input_sorted)
    {
        input_double.push_back(Point2D_double{
            static_cast<double>(pt.x),
            static_cast<double>(pt.y),
            static_cast<double>(pt.vx),
            static_cast<double>(pt.vy),
            static_cast<double>(pt.temp)
        });
    }

    // Launch GPU bitmask pipeline (double)
    bitmask_stream_compaction_gpu_double(input_double,
                                         threshold_d,
                                         blockSize,
                                         time_ms,
                                         output_d_raw);

    // Convert results back to float for an apples-to-apples size check
    output_d.reserve(output_d_raw.size());
    for (const auto& pt : output_d_raw)
    {
        output_d.push_back(Point2D{
            static_cast<float>(pt.x),
            static_cast<float>(pt.y),
            static_cast<float>(pt.vx),
            static_cast<float>(pt.vy),
            static_cast<float>(pt.temp)
        });
    }

    // CPU baseline
    cpu_baseline = compact_stream_cpu(input_sorted, threshold_d);

    // Size error
    const int diff = std::abs(static_cast<int>(output_d.size()) -
                              static_cast<int>(cpu_baseline.size()));
    error = cpu_baseline.empty() ? 0.0f
                                 : static_cast<float>(diff) /
                                   static_cast<float>(cpu_baseline.size());

    printPointsComparison(output_d, cpu_baseline);
}