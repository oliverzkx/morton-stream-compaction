/**
 * @file benchmark_utils.cu
 * @brief Implementation of benchmark utility functions for stream compaction tests.
 *        This includes running GPU compaction kernels (currently bitmask version only)
 *        under different input sizes, thread block configurations, and precision modes.
 * 
 *        The benchmark outputs include execution time and (optionally) numerical error.
 *        Results are used for CSV output and further visualization.
 * 
 *        Note: Currently only float-precision kernels are supported. Double-precision
 *        versions will be implemented in future extensions.
 * 
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-07-11
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

/**
 * @brief Print the first few points from both GPU and CPU outputs for comparison.
 *
 * This function compares the first few elements of the GPU result and the CPU baseline.
 * It helps visually verify correctness when debugging numerical mismatches or unexpected results.
 *
 * @param gpu_output Vector containing GPU-compacted points (converted to float)
 * @param cpu_baseline Vector containing CPU-compacted points
 */
void printPointsComparison(const std::vector<Point2D>& gpu, const std::vector<Point2D>& cpu) {
    std::cout << "\n=== 🔵 Comparing first 5 points ===\n";

    // Sort both GPU and CPU output by Morton code to ensure consistent order
    std::vector<Point2D> sorted_gpu = gpu;
    std::vector<Point2D> sorted_cpu = cpu;
    std::sort(sorted_gpu.begin(), sorted_gpu.end(), compareMorton);
    std::sort(sorted_cpu.begin(), sorted_cpu.end(), compareMorton);

    int count = std::min(5, std::min((int)sorted_gpu.size(), (int)sorted_cpu.size()));
    for (int i = 0; i < count; ++i) {
        const Point2D& g = sorted_gpu[i];
        const Point2D& c = sorted_cpu[i];
        std::cout << "GPU[" << i << "]: (x=" << g.x << ", y=" << g.y << ", temp=" << g.temp << ")  |  ";
        std::cout << "CPU[" << i << "]: (x=" << c.x << ", y=" << c.y << ", temp=" << c.temp << ")\n";
    }

    std::cout << "GPU output size = " << gpu.size()
              << ", CPU baseline size = " << cpu.size() << "\n";
}


bool compareMorton(const Point2D& a, const Point2D& b) {
    return morton2D_encode((int)a.x, (int)a.y) < morton2D_encode((int)b.x, (int)b.y);
}

// -----------------------------------------------------------------------------
//  runBitmaskBenchmark
//  • size       : total number of points to generate            | 总点数
//  • blockSize  : CUDA blockDim.x                               | CUDA 线程块大小
//  • precision  : "float" or "double"                           | 精度选择
//  • time_ms    : (out) measured GPU time in ms                 | 输出：GPU 计时
//  • error      : (out) relative size diff to CPU baseline      | 输出：结果大小误差
// -----------------------------------------------------------------------------
void runBitmaskBenchmark(int size,
                         int blockSize,
                         const std::string& precision,
                         float& time_ms,
                         float& error)
{
    /* ------------------------------------------------------------ *
     * 1. Generate & sort input points                              *
     * ------------------------------------------------------------ */
    int rows = std::sqrt(size);                         // English: derive grid dims
    int cols = (size + rows - 1) / rows;                // 中文：根据 size 生成近似方阵

    std::vector<Point2D> input =
        generatePoints(rows, cols, 1.0f, 42);           // English: fixed seed
                                                        // 中文：固定随机种子
    if ((int)input.size() > size) input.resize(size);   // Trim to exact size | 截断到精确大小

    std::vector<Point2D> input_sorted = input;          // Clone for Morton sort | 克隆一份再排序
    std::sort(input_sorted.begin(), input_sorted.end(), compareMorton);

    /* ------------------------------------------------------------ *
     * 2. Set threshold                                             *
     * ------------------------------------------------------------ */
    constexpr float  threshold_f = 25.0f;
    constexpr double threshold_d = 25;

    std::vector<Point2D>        cpu_baseline;           // CPU reference | CPU 参考结果
    std::vector<Point2D>        output_f;               // GPU float out | GPU float 结果
    std::vector<Point2D_double> output_d_raw;           // GPU double raw| GPU double 原始结果
    std::vector<Point2D>        output_d;               // converted back | 转回 float 便于对比

    /* ------------------------------------------------------------ *
     * 3-A. FLOAT path                                              *
     * ------------------------------------------------------------ */
    if (precision == "float")
    {
        bitmask_stream_compaction_gpu_float(
            input_sorted, threshold_f, blockSize,
            time_ms, output_f);                         // GPU float compaction | GPU float 压缩

        cpu_baseline = compact_stream_cpu(
            input_sorted, threshold_f);                 // CPU baseline | CPU 基准

        int diff = std::abs((int)output_f.size() -
                            (int)cpu_baseline.size());
        error = cpu_baseline.empty() ? 0.0f
                                     : static_cast<float>(diff) /
                                       cpu_baseline.size();

        printPointsComparison(output_f, cpu_baseline);  // Compare top-5 | 对比前 5 个
        return;
    }

    /* ------------------------------------------------------------ *
     * 3-B. DOUBLE path                                             *
     * ------------------------------------------------------------ */

    // 3-B-1.  Point2D → Point2D_double  (fill ALL fields)
    //        英文：convert & copy all 5 members
    //        中文：转换并完整填充 5 个成员
    std::vector<Point2D_double> input_double;
    input_double.reserve(input_sorted.size());

    for (const auto& pt : input_sorted)
    {
        input_double.push_back(Point2D_double{
            .x    = static_cast<double>(pt.x),
            .y    = static_cast<double>(pt.y),
            .vx   = static_cast<double>(pt.vx),
            .vy   = static_cast<double>(pt.vy),
            .temp = static_cast<double>(pt.temp)
        });
    }

    // 3-B-2. GPU double compaction
    bitmask_stream_compaction_gpu_double(
        input_double, threshold_d, blockSize,
        time_ms, output_d_raw);

    // 3-B-3. Convert GPU result back to Point2D for easy diff
    output_d.reserve(output_d_raw.size());
    for (const auto& pt : output_d_raw)
    {
        output_d.push_back(Point2D{
            .x    = static_cast<float>(pt.x),
            .y    = static_cast<float>(pt.y),
            .vx   = static_cast<float>(pt.vx),
            .vy   = static_cast<float>(pt.vy),
            .temp = static_cast<float>(pt.temp)
        });
    }

    // 3-B-4. CPU baseline (still float threshold is OK)
    cpu_baseline = compact_stream_cpu(
        input_sorted, threshold_d);

    int diff = std::abs((int)output_d.size() -
                        (int)cpu_baseline.size());
    error = cpu_baseline.empty() ? 0.0f
                                 : static_cast<float>(diff) /
                                   cpu_baseline.size();

    printPointsComparison(output_d, cpu_baseline);      // Top-5 diff | 打印前 5 个
}



