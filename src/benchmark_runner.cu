/**
 * @file benchmark_runner.cu
 * @brief Entry point for stream compaction benchmarking program.
 *        This program runs GPU stream compaction under different configurations
 *        and writes performance/error results to a CSV file.
 * 
 *        Currently supports the bitmask-based compaction kernel in both float
 *        and double precision.
 * 
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-07-11
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "benchmark_utils.h"

int main() {
    // 📁 Open CSV file for writing results
    std::ofstream outFile("results/benchmark_results.csv");
    outFile << "size,block_size,precision,time_ms,error\n";

    // 📏 Define input sizes: from 2^10 (1K) to 2^23 (~8M)
    std::vector<int> sizes;
    for (int exp = 10; exp <= 23; ++exp)
        sizes.push_back(1 << exp);

    // 🧱 Define thread block sizes (common CUDA choices)
    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};

    // ⚙️ Define precision modes: float and double
    std::vector<std::string> precisions = {"float", "double"};

    // 🧪 Run all combinations
    for (const int& size : sizes) {
        for (const int& blockSize : blockSizes) {
            for (const std::string& precision : precisions) {
                float time_ms = 0.0f;
                float error = 0.0f;

                // Run the benchmark
                runBitmaskBenchmark(size, blockSize, precision, time_ms, error);

                // Write result to CSV
                outFile << size << "," << blockSize << "," << precision << ","
                        << time_ms << "," << error << "\n";

                // 🟢 Print progress to console
                std::cout << "[✓] size=" << size
                          << " block=" << blockSize
                          << " precision=" << precision
                          << " time=" << time_ms << " ms"
                          << " error=" << error << "\n";
            }
        }
    }

    outFile.close();
    std::cout << "✅ Benchmark completed! Results saved to: results/benchmark_results.csv\n";
    return 0;
}
