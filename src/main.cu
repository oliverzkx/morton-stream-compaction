/**
 * @file main.cu
 * @brief Entry point for the Morton Stream Compaction project. Generates 2D points, prints Morton codes, 
 * performs sorting, and applies various CPU/GPU stream compaction methods.
 * 
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-06-23
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include "common.h"
#include "utils.h"
#include "stream_compaction.h"

// Global runtime configuration
int numElements = 100;
bool useRandomSeed = false;
bool showTiming = false;
bool runCPU = true;
bool runGPU = true;
int maxPrint = 10;  // Maximum number of points to print

/// Enum to select sorting method
enum SortMethod {
    CPU_STL,
    THRUST_CPU,
    THRUST_GPU
};

void printUsage();
void parseArgs(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    parseArgs(argc, argv);

    std::cout << "✅ Morton encoding test started!" << std::endl;

    // Show CLI configuration
    std::cout << "🧭 Settings: "
              << "[Elements: " << numElements << "] "
              << "[Seed: " << (useRandomSeed ? "Random" : "Fixed") << "] "
              << "[Timing: " << (showTiming ? "On" : "Off") << "] "
              << "[RunCPU: " << (runCPU ? "Yes" : "No") << "] "
              << "[RunGPU: " << (runGPU ? "Yes" : "No") << "]\n";

    // Step 0: Select CUDA device
    int numDevices = chooseCudaCard(true);
    if (numDevices == 0) return -1;

    // Step 1: Generate 2D points based on numElements
    int rows = std::sqrt(numElements);
    int cols = (numElements + rows - 1) / rows;
    auto points = generatePoints(rows, cols, useRandomSeed ? -1.0f : 1.0f);
    if (points.size() > numElements) {
        points.resize(numElements);  // Trim extra points
    }

    // Step 2: Print generated points
    printPointList(points, "🔵 Generated Points", maxPrint);

    // Step 3: Print Morton codes before sorting
    std::cout << "\n📌 Morton codes before sorting:\n";
    int limit = std::min(maxPrint, static_cast<int>(points.size()));
    for (int i = 0; i < limit; ++i) {
        int x = static_cast<int>(points[i].x);
        int y = static_cast<int>(points[i].y);
        unsigned int code = morton2D_encode(x, y);
        std::cout << "(x=" << x << ", y=" << y << ") → Morton = " << code << "\n";
    }
    if ((int)points.size() > maxPrint) {
        std::cout << "... [Showing first " << maxPrint 
                << " of total " << points.size() << " points]\n";
    }

    // Step 4: Sorting (default: CPU STL)
    SortMethod method = CPU_STL;

    if (runCPU) {
        if (method == CPU_STL) {
            std::cout << "\n🧠 Sorting with std::sort (CPU)...\n";
            sort_by_morton(points);
        } else if (method == THRUST_CPU) {
            std::cout << "\n🧠 Sorting with Thrust (CPU backend)...\n";
            sort_by_morton_thrust(points, false);
        }
    }

    if (runGPU && method == THRUST_GPU) {
        std::cout << "\n🚀 Sorting with Thrust (GPU backend)...\n";
        sort_by_morton_thrust(points, true);
    }

    // Step 5: Print sorted points
    printPointList(points, "🌀 Points After Morton Sorting", maxPrint);

    // Step 6: Apply stream compaction
    float threshold = 30.0f;

    if (runCPU) {
        std::vector<Point2D> compacted_cpu = compact_stream_cpu(points, threshold);
        std::vector<Point2D> compacted_thrust_cpu = compact_points_thrust(points, threshold, false);

        std::cout << "👉 CPU manual compaction:        " << compacted_cpu.size() << " points\n";
        std::cout << "👉 Thrust CPU compaction:        " << compacted_thrust_cpu.size() << " points\n";

        printPointList(compacted_cpu, "✅ CPU Compacted Points", maxPrint);
        printPointList(compacted_thrust_cpu, "✅ Thrust CPU Compacted Points", maxPrint);
    }

    if (runGPU) {
        std::vector<Point2D> compacted_thrust_gpu = compact_points_thrust(points, threshold, true);
        std::vector<Point2D> compacted_naive_gpu;
        std::vector<Point2D> compacted_shared_gpu;
        std::vector<Point2D> compacted_warp_gpu;
        std::vector<Point2D> compacted_bitmask_gpu;

        testNaiveGPUCompaction(points, threshold, compacted_naive_gpu);
        testSharedGPUCompaction(points, threshold, compacted_shared_gpu);
        testWarpGPUCompaction(points, threshold, compacted_warp_gpu);
        testBitmaskGPUCompaction(points, threshold, compacted_bitmask_gpu);

        std::cout << "👉 Thrust GPU compaction:        " << compacted_thrust_gpu.size() << " points\n";
        std::cout << "👉 Naive GPU compaction (CUDA):  " << compacted_naive_gpu.size() << " points\n";
        std::cout << "👉 Shared memory GPU compaction: " << compacted_shared_gpu.size() << " points\n";
        std::cout << "👉 Warp shuffle GPU compaction:  " << compacted_warp_gpu.size() << " points\n";
        std::cout << "👉 Bitmask GPU compaction:       " << compacted_bitmask_gpu.size() << " points\n";

        printPointList(compacted_thrust_gpu,  "✅ Thrust GPU Compacted Points", maxPrint);
        printPointList(compacted_naive_gpu,   "✅ Naive GPU Compacted Points", maxPrint);
        printPointList(compacted_shared_gpu,  "✅ Shared Memory GPU Compacted Points", maxPrint);
        printPointList(compacted_warp_gpu,    "✅ Warp Shuffle GPU Compacted Points", maxPrint);
        printPointList(compacted_bitmask_gpu, "✅ Bitmask GPU Compacted Points", maxPrint);
    }

    return 0;
}


/// Parses command-line arguments and updates global settings
void parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            numElements = std::stoi(argv[++i]);
        } else if (arg == "-r") {
            useRandomSeed = true;
        } else if (arg == "-t") {
            showTiming = true;
        } else if (arg == "-c") {
            runCPU = true;
            runGPU = false;
        } else if (arg == "-g") {
            runCPU = false;
            runGPU = true;
        } else if (arg == "-h") {
            printUsage();
            exit(0);
        }else {
            std::cerr << "[❌] Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }
}

void printUsage() {
    std::cout << "\n===========================================================\n";
    std::cout << "🧾 Program:   Stream Compaction with Morton Encoding\n";
    std::cout << "👨‍💻 Author:    Kaixiang Zou <zouk@tcd.ie>\n";
    std::cout << "🔢 Version:   1.0\n";
    std::cout << "📅 Date:      2025-07-03\n";
    std::cout << "📝 Description:\n";
    std::cout << "   This program generates 2D points, encodes them using Morton codes,\n";
    std::cout << "   sorts them, and performs stream compaction using CPU and GPU methods.\n";
    std::cout << "\n📌 Usage: ./stream_compaction [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -n <int>    Set number of points (default: 10000)\n";
    std::cout << "  -r          Use random seed for point generation\n";
    std::cout << "  -t          Show kernel execution timing\n";
    std::cout << "  -c          Run only CPU implementations\n";
    std::cout << "  -g          Run only GPU implementations\n";
    std::cout << "  -h          Show this help message and exit\n";
    std::cout << "===========================================================\n\n";
}