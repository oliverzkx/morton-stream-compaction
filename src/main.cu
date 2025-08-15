/**
 * @file main.cu
 * @brief Morton stream-compaction demo showcasing several GPU variants.
 *
 * Supported variants
 *   • naive         – whole-array GPU compaction
 *   • bin-atomic    – Plan-B single-pass atomic method
 *   • bin-partition – Plan-A histogram → scan → scatter pipeline
 *
 * Command-line arguments let the user select mode, variant, per-bin kernel,
 * and various run-time parameters (point count, k-bits, etc.).
 *
 * Author:  Kaixiang Zou  
 * Version: 1.1  
 * Date:    2025-07-26
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include "common.h"
#include "utils.h"
#include "stream_compaction.h"
#include "stream_compaction_bin.h"
#include "bin_kernel.h"

// ────────────────────────────────────────────────────────────────
// Runtime-configuration structure
// ────────────────────────────────────────────────────────────────

/**
 * @struct CmdCfg
 * @brief Aggregates all command-line options parsed at start-up.
 */
struct CmdCfg
{
    int  numPoints  = 100;   ///< Total points to generate
    bool randomSeed = false; ///< Use std::random_device if true
    bool showTiming = false; ///< Print kernel / total timings
    bool runCPU     = true;  ///< Execute CPU reference
    bool runGPU     = true;  ///< Execute GPU variants
    int  maxPrint   = 10;    ///< Max points to print for inspection
    int  kBits      = 8;     ///< Low-bit width for bin IDs

    /// Top-level execution mode
    enum class Mode { Naive, Bin } mode = Mode::Naive;
    /// Bin implementation family (Plan-B atomic vs. Plan-A partition)
    enum class Variant { Atomic, Partition } variant = Variant::Atomic;
    /// Per-bin kernel strategy when using Plan-A
    BinKernel kernelKind = BinKernel::Shared;
};

// Forward declarations
void                printUsage();
std::optional<CmdCfg> parseArgs(int, char**);

// ────────────────────────────────────────────────────────────────
// main
// ────────────────────────────────────────────────────────────────

/**
 * @brief Entry point: parse options, generate data, run selected variants,
 *        and print results.
 */
int main(int argc, char* argv[])
{
    auto cfgOpt = parseArgs(argc, argv);
    if (!cfgOpt) return 0;
    CmdCfg cfg = *cfgOpt;

    /* ------------------- settings summary ------------------- */
    std::cout << "✅ Morton stream-compaction demo\n";
    std::cout << "🧭 Settings → N=" << cfg.numPoints
              << ", seed="   << (cfg.randomSeed ? "random" : "fixed")
              << ", mode="   << (cfg.mode == CmdCfg::Mode::Naive ? "naive" : "bin")
              << ", variant="<< (cfg.variant == CmdCfg::Variant::Atomic ?
                                 "atomic" : "partition")
              << ", kernel=" << ([&]{
                     switch (cfg.kernelKind) {
                         case BinKernel::Shared:  return "shared";
                         case BinKernel::Warp:    return "warp";
                         case BinKernel::Bitmask: return "bitmask";
                         case BinKernel::Auto:    return "auto";
                     } return "??";
                 })()
              << ", k="      << cfg.kBits
              << ", CPU="    << (cfg.runCPU ? "on" : "off")
              << ", GPU="    << (cfg.runGPU ? "on" : "off") << '\n';

    //if (cfg.runGPU && chooseCudaCard(true) == 0) return -1;
    if (cfg.runGPU) {
        int devCount = 0;
        auto st = cudaGetDeviceCount(&devCount);
        if (st != cudaSuccess || devCount <= 0) {
            std::cerr << "No CUDA device found\n";
            return -1;
        }
        // 静默选择 0 号设备；如果以后需要可加命令行参数来指定
        cudaSetDevice(0);
    }

    /* ------------------- data generation -------------------- */
    const int rows = static_cast<int>(std::sqrt(cfg.numPoints));
    const int cols = (cfg.numPoints + rows - 1) / rows;
    auto points = generatePoints(rows, cols, cfg.randomSeed ? -1.f : 1.f);
    if (points.size() > cfg.numPoints) points.resize(cfg.numPoints);

    printPointList(points, "🔵 Generated Points", cfg.maxPrint);
    sort_by_morton(points);
    printPointList(points, "🌀 Points After Morton Sorting", cfg.maxPrint);

    constexpr float THRESHOLD = 30.f;

    /* ------------------- CPU reference ---------------------- */
    if (cfg.runCPU) {
        auto cpu1 = compact_stream_cpu(points, THRESHOLD);
        auto cpu2 = compact_points_thrust(points, THRESHOLD, false);
        std::cout << "👉 CPU manual  : " << cpu1.size() << " pts\n";
        std::cout << "👉 Thrust CPU  : " << cpu2.size() << " pts\n";
    }

    /* ------------------- GPU variants ----------------------- */
    if (cfg.runGPU) {
        auto gpu_th = compact_points_thrust(points, THRESHOLD, true);
        std::cout << "👉 Thrust GPU  : " << gpu_th.size() << " pts\n";

        std::vector<Point2D> gpu_out;
        float tKer = 0.f, tTot = 0.f;

        if (cfg.mode == CmdCfg::Mode::Naive) {
        testNaiveGPUCompaction(points, THRESHOLD, gpu_out, &tKer, &tTot);
        if (cfg.showTiming) {
            std::cout << "👉 Unbinned baseline : "
                    << gpu_out.size() << " pts, kernel "
                    << tKer << " ms, total " << tTot << " ms\n";
        } else {
            std::cout << "👉 Unbinned baseline : "
                    << gpu_out.size() << " pts\n";
        }
    } else { /* Bin mode */
        if (cfg.variant == CmdCfg::Variant::Atomic) {
            testBinGPUCompaction_atomic(points, THRESHOLD, cfg.kBits,
                                        gpu_out, tKer, tTot);
            if (cfg.showTiming) {
                std::cout << "👉 Bin GPU(k=" << cfg.kBits << ") [atomic] : "
                        << gpu_out.size() << " pts, kernel "
                        << tKer << " ms, total " << tTot << " ms\n";
            } else {
                std::cout << "👉 Bin GPU(k=" << cfg.kBits << ") [atomic] : "
                        << gpu_out.size() << " pts\n";
            }
        } else { /* Partition (Plan-A) */
            testBinGPUCompaction_partition(points, THRESHOLD, cfg.kBits,
                                        gpu_out, tKer, tTot, cfg.kernelKind);
            if (cfg.showTiming) {
                std::cout << "👉 Bin GPU(k=" << cfg.kBits << ") [partition/"
                        << ([&]{
                                switch (cfg.kernelKind) {
                                    case BinKernel::Shared:  return "shared";
                                    case BinKernel::Warp:    return "warp";
                                    case BinKernel::Bitmask: return "bitmask";
                                    case BinKernel::Auto:    return "auto";
                                } return "??";
                            })()
                        << "] : " << gpu_out.size() << " pts, kernel "
                        << tKer << " ms, total " << tTot << " ms\n";
            } else {
                std::cout << "👉 Bin GPU(k=" << cfg.kBits << ") [partition/"
                        << ([&]{
                                switch (cfg.kernelKind) {
                                    case BinKernel::Shared:  return "shared";
                                    case BinKernel::Warp:    return "warp";
                                    case BinKernel::Bitmask: return "bitmask";
                                    case BinKernel::Auto:    return "auto";
                                } return "??";
                            })()
                        << "] : " << gpu_out.size() << " pts\n";
            }
        }
    }
        printPointList(gpu_out, "✅ GPU Compacted Output", cfg.maxPrint);
    }

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}

// ────────────────────────────────────────────────────────────────
// Helper functions
// ────────────────────────────────────────────────────────────────

/**
 * @brief Parse command-line arguments into a ::CmdCfg structure.
 *
 * @return A populated configuration on success, <tt>nullopt</tt> if the user
 *         requested <tt>-h/--help</tt>.
 */
std::optional<CmdCfg> parseArgs(int argc, char* argv[])
{
    CmdCfg cfg;

    auto nextInt = [&](int& out, int& i, char* argv[]) {
        if (i + 1 >= argc) {
            std::cerr << "Missing value after " << argv[i] << '\n';
            std::exit(1);
        }
        out = std::stoi(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        std::string_view arg{argv[i]};

        if (arg == "-n")          nextInt(cfg.numPoints, i, argv);
        else if (arg == "-r")     cfg.randomSeed = true;
        else if (arg == "-t")     cfg.showTiming = true;
        else if (arg == "-c")     { cfg.runCPU = true;  cfg.runGPU = false; }
        else if (arg == "-g")     { cfg.runCPU = false; cfg.runGPU = true;  }
        else if (arg == "-k")     nextInt(cfg.kBits, i, argv);

        else if (arg == "--mode") {
            if (++i >= argc) { std::cerr << "Missing value after --mode\n"; exit(1); }
            std::string_view m{argv[i]};
            if      (m == "naive") cfg.mode = CmdCfg::Mode::Naive;
            else if (m == "bin")   cfg.mode = CmdCfg::Mode::Bin;
            else { std::cerr << "Unknown mode " << m << '\n'; exit(1); }
        }
        else if (arg == "--variant") {
            if (++i >= argc) { std::cerr << "Missing value after --variant\n"; exit(1); }
            std::string_view v{argv[i]};
            if      (v == "atomic")     cfg.variant = CmdCfg::Variant::Atomic;
            else if (v == "partition")  cfg.variant = CmdCfg::Variant::Partition;
            else { std::cerr << "Unknown variant " << v << '\n'; exit(1); }
        }
        else if (arg == "--kernel") {
            if (++i >= argc) { std::cerr << "Missing value after --kernel\n"; exit(1); }
            std::string_view k{argv[i]};
            if      (k == "shared")  cfg.kernelKind = BinKernel::Shared;
            else if (k == "warp")    cfg.kernelKind = BinKernel::Warp;
            else if (k == "bitmask") cfg.kernelKind = BinKernel::Bitmask;
            else if (k == "auto")    cfg.kernelKind = BinKernel::Auto;
            else { std::cerr << "Unknown kernel " << k << '\n'; exit(1); }
        }
        else if (arg == "-h" || arg == "--help") {
            printUsage();
            return std::nullopt;
        }
        else {
            std::cerr << "Unknown argument: " << arg << '\n';
            printUsage();
            std::exit(1);
        }
    }
    return cfg;
}

/**
 * @brief Print command-line usage information.
 */
void printUsage()
{
    std::cout << R"(
============================================================
Stream-Compaction with Morton Encoding
Author : Kaixiang Zou
Contact: gracefulblack2001@gmail.com

Options:
  -n <int>        Number of points              (default 100)
  -r              Use random seed
  -t              Show timing (kernel / total)
  -c              CPU only
  -g              GPU only
  --mode <str>    naive | bin                   (default naive)
  --variant <str> atomic | partition            (bin mode only)
  --kernel <str>  shared | warp | bitmask | auto  (partition only)
  -k <int>        k bits for bin ID             (default 8)
  -h, --help      Show this help
============================================================
)" << std::endl;
}