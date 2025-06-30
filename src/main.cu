/**
 * @file main.cu
 * @brief Entry point for Morton Stream Compaction project. Generates points, prints Morton codes, and applies sorting.
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-06-23
 */


#include <iostream>
#include "common.h"
#include "utils.h"
#include "stream_compaction.h"


/// Enum to select sorting method
enum SortMethod {
    CPU_STL,
    THRUST_CPU,
    THRUST_GPU
};

int main() {
    std::cout << "✅ Morton encoding test started!" << std::endl;

    // Step 1: generate grid points (with optional fixed seed)
    auto points = generatePoints(4, 2, 1.0f);  // You can fix seed for repeatability

    // Step 2: print generated points
    printPointList(points, "🔵 Generated Points");

    // Step 3: print Morton codes before sorting
    std::cout << "\n📌 Morton codes before sorting:" << std::endl;
    for (const auto& p : points) {
        int x = static_cast<int>(p.x);
        int y = static_cast<int>(p.y);
        unsigned int code = morton2D_encode(x, y);
        std::cout << "(x=" << x << ", y=" << y << ") → Morton = " << code << std::endl;
    }

    // Step 4: choose sorting method
    SortMethod method = CPU_STL;  // Change to CPU_STL / THRUST_CPU / THRUST_GPU

    if (method == CPU_STL) {
        std::cout << "\n🧠 Sorting with std::sort (CPU)...\n";
        sort_by_morton(points);
    }
    else if (method == THRUST_CPU) {
        std::cout << "\n🧠 Sorting with Thrust (CPU backend)...\n";
        sort_by_morton_thrust(points, false);  // useGPU = false
    }
    else if (method == THRUST_GPU) {
        std::cout << "\n🚀 Sorting with Thrust (GPU backend)...\n";
        sort_by_morton_thrust(points, true);   // useGPU = true
    }

    // Step 5: print points after sorting
    printPointList(points, "🌀 Points After Morton Sorting");

    // Step 6: apply stream compaction with 3 methods
    float threshold = 30.0f;

    std::vector<Point2D> compacted_cpu = compact_stream_cpu(points, threshold);
    std::vector<Point2D> compacted_thrust_cpu = compact_points_thrust(points, threshold, false);
    std::vector<Point2D> compacted_thrust_gpu = compact_points_thrust(points, threshold, true);

   // ✅ Step 6.1: test CUDA compactions
    std::vector<Point2D> compacted_naive_gpu;
    testNaiveGPUCompaction(points, threshold, compacted_naive_gpu);

    std::vector<Point2D> compacted_shared_gpu;
    testSharedGPUCompaction(points, threshold, compacted_shared_gpu);

    // Step 7: print results
    std::cout << "\n🔥 Stream Compaction Results (threshold = " << threshold << ")\n";
    std::cout << "Original points count: " << points.size() << std::endl;
    std::cout << "👉 CPU manual compaction:        " << compacted_cpu.size() << " points\n";
    std::cout << "👉 Thrust CPU compaction:        " << compacted_thrust_cpu.size() << " points\n";
    std::cout << "👉 Thrust GPU compaction:        " << compacted_thrust_gpu.size() << " points\n";
    std::cout << "👉 Naive GPU compaction (CUDA):  " << compacted_naive_gpu.size() << " points\n";
    std::cout << "👉 Shared memory GPU compaction: " << compacted_shared_gpu.size() << " points\n";

    printPointList(compacted_cpu,           "✅ CPU Compacted Points");
    printPointList(compacted_thrust_cpu,    "✅ Thrust CPU Compacted Points");
    printPointList(compacted_thrust_gpu,    "✅ Thrust GPU Compacted Points");
    printPointList(compacted_naive_gpu,     "✅ Naive GPU Compacted Points");
    printPointList(compacted_shared_gpu,    "✅ Shared Memory GPU Compacted Points");

    return 0;
}