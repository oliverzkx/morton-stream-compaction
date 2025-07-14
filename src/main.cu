/**
 * @file main.cu
 * @brief Morton stream-compaction demo with multiple GPU variants.
 *        Supports:
 *          • naive  : whole-array GPU compaction
 *          • bin-atomic     (Plan-B)
 *          • bin-partition  (Plan-A, histogram→scan→scatter)
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

/* ============================================================
   Runtime configuration
   ============================================================*/
struct CmdCfg {
    int  numPoints  = 100;
    bool randomSeed = false;
    bool showTiming = false;
    bool runCPU     = true;
    bool runGPU     = true;
    int  maxPrint   = 10;
    int  kBits      = 8;

    enum class Mode    { Naive, Bin }      mode    = Mode::Naive;
    enum class Variant { Atomic, Partition } variant = Variant::Atomic; // NEW
};

/* ============================================================*/
void printUsage();
std::optional<CmdCfg> parseArgs(int, char**);

/* ============================================================*/
int main(int argc, char* argv[])
{
    auto cfgOpt = parseArgs(argc, argv);
    if (!cfgOpt) return 0;
    CmdCfg cfg = *cfgOpt;

    std::cout << "✅ Morton stream-compaction demo\n";
    std::cout << "🧭 Settings → N=" << cfg.numPoints
              << ", seed=" << (cfg.randomSeed ? "random" : "fixed")
              << ", mode=" << (cfg.mode == CmdCfg::Mode::Naive ? "naive" : "bin")
              << ", variant=" << (cfg.variant == CmdCfg::Variant::Atomic ?
                                  "atomic" : "partition")
              << ", k="   << cfg.kBits
              << ", CPU=" << (cfg.runCPU ? "on" : "off")
              << ", GPU=" << (cfg.runGPU ? "on" : "off") << '\n';

    if (cfg.runGPU && chooseCudaCard(true) == 0) return -1;

    /* ---------- generate & sort -------- */
    int rows = static_cast<int>(std::sqrt(cfg.numPoints));
    int cols = (cfg.numPoints + rows - 1) / rows;
    auto points = generatePoints(rows, cols, cfg.randomSeed ? -1.f : 1.f);
    if (points.size() > cfg.numPoints) points.resize(cfg.numPoints);

    printPointList(points, "🔵 Generated Points", cfg.maxPrint);
    sort_by_morton(points);
    printPointList(points, "🌀 Points After Morton Sorting", cfg.maxPrint);

    constexpr float THRESHOLD = 30.f;

    /* ---------- CPU reference ---------- */
    if (cfg.runCPU) {
        auto cpu1 = compact_stream_cpu(points, THRESHOLD);
        auto cpu2 = compact_points_thrust(points, THRESHOLD, false);
        std::cout << "👉 CPU manual  : " << cpu1.size() << " pts\n";
        std::cout << "👉 Thrust CPU  : " << cpu2.size() << " pts\n";
    }

    /* ---------- GPU variants ---------- */
    if (cfg.runGPU) {
        auto gpu_th = compact_points_thrust(points, THRESHOLD, true);
        std::cout << "👉 Thrust GPU  : " << gpu_th.size() << " pts\n";

        std::vector<Point2D> gpu_out;
        float tKer = 0.f, tTot = 0.f;

        if (cfg.mode == CmdCfg::Mode::Naive) {
            testNaiveGPUCompaction(points, THRESHOLD, gpu_out);
            std::cout << "👉 Naive GPU   : " << gpu_out.size() << " pts\n";

        } else {            /* Bin mode */
            if (cfg.variant == CmdCfg::Variant::Atomic) {
                testBinGPUCompaction_atomic(points, THRESHOLD,
                                            cfg.kBits, gpu_out,
                                            tKer, tTot);
                std::cout << "👉 Bin GPU(k=" << cfg.kBits << ") [atomic]    : "
                          << gpu_out.size() << " pts, kernel "
                          << tKer << " ms, total " << tTot << " ms\n";
            } else {        /* Partition (Plan-A) */
                testBinGPUCompaction_partition(points, THRESHOLD,
                                               cfg.kBits, gpu_out,
                                               tKer, tTot);
                std::cout << "👉 Bin GPU(k=" << cfg.kBits << ") [partition] : "
                          << gpu_out.size() << " pts, kernel "
                          << tKer << " ms, total " << tTot << " ms\n";
            }
        }
        printPointList(gpu_out, "✅ Bin GPU Compacted", cfg.maxPrint);
    }
    return 0;
}

/* ============================================================
   Helpers
   ============================================================*/
std::optional<CmdCfg> parseArgs(int argc, char* argv[])
{
    CmdCfg cfg;
    auto nextInt = [&](int& out, int& i, char* argv[]) {
        if (i + 1 >= argc) { std::cerr << "Missing value after " << argv[i] << '\n'; exit(1); }
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
        else if (arg == "-h" || arg == "--help") { printUsage(); return std::nullopt; }
        else { std::cerr << "[❌] Unknown argument: " << arg << '\n'; printUsage(); exit(1); }
    }
    return cfg;
}

void printUsage()
{
    std::cout << R"(
    ============================================================
    Stream-Compaction with Morton Encoding
    Options:
    -n <int>        Number of points           (default 100)
    -r              Use random seed
    -t              Show timing (kernel/total)
    -c              CPU only
    -g              GPU only
    --mode <str>    naive | bin               (default naive)
    --variant <str> atomic | partition        (default atomic, bin mode only)
    -k   <int>      k bits for bin ID         (default 8)
    -h, --help      Show this help
    ============================================================
    )" << std::endl;
}
