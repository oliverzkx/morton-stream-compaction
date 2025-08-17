// q6_ablation.cu
//
// Q6 Ablation Driver (Plan A/B + kernel forcing + optional bin bypass via k=0)
// 使用 common.h，调用 stream_compaction_bin.h 中的：
//   - choose_threshold_for_rate(...)
//   - testPlanA_breakdown(...)
//   - testPlanB_breakdown(...)
// 从 BreakdownPlanA/B 读取阶段时间与总时长。
// 说明：如果要让 --no-gather 真正生效，需要在 stream_compaction_bin.cu 内切换到 scatterToBins 路径。
//
// Build:
//   nvcc -O3 -std=c++17 -Iinclude src/q6_ablation.cu -o build/q6_ablation
//
// Example:
//   ./build/q6_ablation --N 20000000 --k 8 --hit 0.50 --dist uniform \
//                       --plan A --force-kernel auto --repeats 5 --csv q6.csv
//
// CSV Columns:
//   plan,ablation,variant,dist,kBits,N,hit_rate,
//   codes_ms,hist_ms,scan_ms,scatter_ms,compact_ms,count_ms,reduce_ms,write_ms,
//   kernel_ms,e2e_ms,total
//
// 注：Plan A 会填 hist/scan/scatter/compact；Plan B 会填 count/scan(+reduce)/write。

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>

#include "common.h"                 // Point2D, set_block_size(...)
#include "bin_kernel.h"             // BinKernel
#include "benchmark_utils.h"        // BreakdownPlanA/B
#include "stream_compaction_bin.h"  // choose_threshold_for_rate, testPlanA/B_breakdown

// -------------------- dataset generators (local) --------------------
static std::vector<Point2D> make_uniform_dataset(std::size_t N, uint64_t seed=1234) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U01(0.f,1.f), V(-4.f,4.f), T(15.f,40.f);
    std::vector<Point2D> v(N);
    for (std::size_t i=0;i<N;++i){ Point2D p{}; p.x=U01(rng); p.y=U01(rng); p.vx=V(rng); p.vy=V(rng); p.temp=T(rng); v[i]=p; }
    return v;
}

static std::vector<Point2D> make_clustered_dataset(std::size_t N, int K=8, float sigma=0.03f, uint64_t seed=5678) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U01(0.f,1.f), V(-4.f,4.f), T(15.f,40.f);
    std::normal_distribution<float> N01(0.f,1.f);
    std::vector<std::pair<float,float>> centers; centers.reserve(K);
    for(int i=0;i<K;++i) centers.emplace_back(U01(rng),U01(rng));
    std::uniform_int_distribution<int> pick(0, K-1);

    std::vector<Point2D> v(N);
    for (std::size_t i=0;i<N;++i){
        auto [cx,cy] = centers[pick(rng)];
        float dx = sigma * N01(rng), dy = sigma * N01(rng);
        Point2D p{}; p.x = std::clamp(cx+dx, 0.f, 1.f); p.y = std::clamp(cy+dy, 0.f, 1.f);
        p.vx = V(rng); p.vy = V(rng); p.temp = T(rng); v[i]=p;
    }
    return v;
}

static std::vector<Point2D> make_skewed_dataset(std::size_t N, double heavy_frac=0.8, int heavy_cells=8, uint64_t seed=9012) {
    // heavy_frac of points go to a few random “heavy” cells; others uniform.
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U01(0.f,1.f), V(-4.f,4.f), T(15.f,40.f);
    std::vector<std::pair<float,float>> cells; cells.reserve(heavy_cells);
    for (int i=0;i<heavy_cells;++i) cells.emplace_back(U01(rng),U01(rng));
    std::uniform_int_distribution<int> pick(0, heavy_cells-1);

    std::vector<Point2D> v(N);
    const std::size_t heavyN = static_cast<std::size_t>(N * heavy_frac);
    for (std::size_t i=0;i<heavyN;++i){
        auto [cx,cy] = cells[pick(rng)];
        float jitter = 1e-3f;
        Point2D p{}; p.x = std::clamp(cx + jitter*U01(rng), 0.f, 1.f);
                      p.y = std::clamp(cy + jitter*U01(rng), 0.f, 1.f);
        p.vx = V(rng); p.vy = V(rng); p.temp = T(rng); v[i]=p;
    }
    for (std::size_t i=heavyN;i<N;++i){
        Point2D p{}; p.x=U01(rng); p.y=U01(rng); p.vx=V(rng); p.vy=V(rng); p.temp=T(rng); v[i]=p;
    }
    return v;
}

static std::vector<Point2D> make_dataset(std::size_t N, const std::string& dist, uint64_t seed){
    if (dist=="uniform")   return make_uniform_dataset(N, seed);
    if (dist=="clustered") return make_clustered_dataset(N, 8, 0.03f, seed);
    if (dist=="skewed")    return make_skewed_dataset(N, 0.8, 8, seed);
    std::cerr << "[Q6] Unknown --dist='"<<dist<<"', fallback to uniform.\n";
    return make_uniform_dataset(N, seed);
}

// -------------------- CLI --------------------
struct Args {
    long long N = 20'000'000;
    int       kBits = 8;
    double    hit_rate = 0.50;
    std::string dist = "uniform"; // uniform|clustered|skewed
    std::string plan = "A";       // A|B
    std::string force_kernel = "auto"; // auto|shared|warp|bitmask
    bool      no_gather = false;      // NOTE: requires internal switch in stream_compaction_bin.cu
    bool      no_binning = false;     // implemented as k=0 (single bin)
    int       repeats = 5;
    std::string csv_path = "";        // empty -> stdout
    int       blockSize = 256;        // optional: override g_block_size
    uint64_t  seed = 1234;
};

static void print_help() {
    std::cout <<
    "Q6 Ablation Driver\n"
    "Options:\n"
    "  --N <int>                (default 20000000)\n"
    "  --k <int>                kBits, default 8 (use 0 to emulate no-binning)\n"
    "  --hit <float>            selectivity, default 0.50\n"
    "  --dist <uniform|clustered|skewed>  default uniform\n"
    "  --plan <A|B>             which plan to run, default A\n"
    "  --force-kernel <auto|shared|warp|bitmask>  default auto\n"
    "  --no-gather              disable gatherCopy (requires code toggle inside Plan A)\n"
    "  --no-binning             bypass Morton/binning via k=0\n"
    "  --repeats <int>          average over runs, default 5\n"
    "  --block <int>            CUDA block size override (default 256)\n"
    "  --seed <uint64>          RNG seed for dataset, default 1234\n"
    "  --csv <path>             write CSV to file; default stdout\n"
    "  -h/--help                show this message\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i=1; i<argc; ++i) {
        const char* s = argv[i];
        if (!std::strcmp(s, "-h") || !std::strcmp(s, "--help")) { print_help(); return false; }
        else if (!std::strcmp(s, "--N") && i+1<argc) { a.N = std::atoll(argv[++i]); }
        else if (!std::strcmp(s, "--k") && i+1<argc) { a.kBits = std::atoi(argv[++i]); }
        else if (!std::strcmp(s, "--hit") && i+1<argc) { a.hit_rate = std::atof(argv[++i]); }
        else if (!std::strcmp(s, "--dist") && i+1<argc) { a.dist = argv[++i]; }
        else if (!std::strcmp(s, "--plan") && i+1<argc) { a.plan = argv[++i]; }
        else if (!std::strcmp(s, "--force-kernel") && i+1<argc) { a.force_kernel = argv[++i]; }
        else if (!std::strcmp(s, "--no-gather")) { a.no_gather = true; }
        else if (!std::strcmp(s, "--no-binning")) { a.no_binning = true; }
        else if (!std::strcmp(s, "--repeats") && i+1<argc) { a.repeats = std::atoi(argv[++i]); }
        else if (!std::strcmp(s, "--block") && i+1<argc) { a.blockSize = std::atoi(argv[++i]); }
        else if (!std::strcmp(s, "--seed") && i+1<argc) { a.seed = static_cast<uint64_t>(std::stoull(argv[++i])); }
        else if (!std::strcmp(s, "--csv") && i+1<argc) { a.csv_path = argv[++i]; }
        else {
            std::cerr << "Unknown option: " << s << "\n";
            print_help();
            return false;
        }
    }
    if (a.repeats < 1) a.repeats = 1;
    return true;
}

static BinKernel parse_kernel(const std::string& s) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    if (t == "shared")  return BinKernel::Shared;
    if (t == "warp")    return BinKernel::Warp;
    if (t == "bitmask") return BinKernel::Bitmask;
    return BinKernel::Auto;
}

static std::string kernel_to_str(BinKernel k) {
    switch (k) {
        case BinKernel::Shared:  return "shared";
        case BinKernel::Warp:    return "warp";
        case BinKernel::Bitmask: return "bitmask";
        default:                 return "auto";
    }
}

// -------------------- CSV --------------------
static void csv_header(std::ostream& os){
    os << "plan,ablation,variant,dist,kBits,N,hit_rate,"
          "codes_ms,hist_ms,scan_ms,scatter_ms,compact_ms,"
          "count_ms,reduce_ms,write_ms,"
          "kernel_ms,e2e_ms,total\n";
}

static void csv_rowA(std::ostream& os, const std::string& ablation, BinKernel k, const Args& a,
                     const BreakdownPlanA& b_avg, long long total_avg){
    os << "A," << ablation << "," << kernel_to_str(k) << ","
       << a.dist << "," << a.kBits << "," << a.N << ","
       << std::fixed << std::setprecision(2) << a.hit_rate << ","
       << std::setprecision(4)
       << b_avg.codes_ms   << ","
       << b_avg.hist_ms    << ","
       << b_avg.scan_ms    << ","
       << b_avg.scatter_ms << ","
       << b_avg.compact_ms << ","
       << 0.0              << ","   // count_ms (N/A)
       << 0.0              << ","   // reduce_ms (N/A)
       << 0.0              << ","   // write_ms (N/A)
       << b_avg.kernel_ms  << ","
       << b_avg.e2e_ms     << ","
       << total_avg        << "\n";
}

static void csv_rowB(std::ostream& os, const std::string& ablation, const Args& a,
                     const BreakdownPlanB& b_avg, long long total_avg){
    os << "B," << ablation << ",atomic,"
       << a.dist << "," << a.kBits << "," << a.N << ","
       << std::fixed << std::setprecision(2) << a.hit_rate << ","
       << std::setprecision(4)
       << b_avg.codes_ms   << ","
       << 0.0              << ","   // hist_ms (N/A)
       << b_avg.scan_ms    << ","   // scan(+reduce)
       << 0.0              << ","   // scatter_ms (N/A)
       << 0.0              << ","   // compact_ms (N/A)
       << b_avg.count_ms   << ","
       << b_avg.reduce_ms  << ","
       << b_avg.write_ms   << ","
       << b_avg.kernel_ms  << ","
       << b_avg.e2e_ms     << ","
       << total_avg        << "\n";
}

// -------------------- main --------------------
int main(int argc, char** argv){
    Args args;
    if (!parse_args(argc, argv, args)) return 0;

    // Allow overriding CUDA block size used by internal kernels
    if (args.blockSize > 0) set_block_size(args.blockSize);

    // Effective k (for --no-binning)
    const int k_effective = args.no_binning ? 0 : args.kBits;

    // Compose ablation tag (fixed '+')
    std::string ablation = "baseline";
    auto add_tag = [&](const std::string& t){
        if (ablation=="baseline") ablation = t; else ablation += "+" + t;
    };
    if (args.no_binning) add_tag("no-binning");
    if (args.no_gather)  add_tag("no-gather");
    BinKernel forced = parse_kernel(args.force_kernel);
    if (forced != BinKernel::Auto) add_tag("force-" + kernel_to_str(forced));

    // Output stream
    std::ofstream ofs;
    std::ostream* osp = &std::cout;
    if (!args.csv_path.empty()){
        ofs.open(args.csv_path);
        if (!ofs){ std::cerr << "Failed to open CSV path: " << args.csv_path << "\n"; return 1; }
        osp = &ofs;
    }
    csv_header(*osp);

    // Generate dataset
    auto h_input = make_dataset((std::size_t)args.N, args.dist, args.seed);

    // Choose threshold for target hit rate
    const float thr = choose_threshold_for_rate(h_input, args.hit_rate);

    if (args.plan=="A" || args.plan=="a"){
        BinKernel which = forced; // Auto or forced
        BreakdownPlanA avg{};      // store averaged times
        long long total_sum = 0;   // sum across repeats for averaging

        for (int r=0;r<args.repeats;++r){
            BreakdownPlanA b{};
            testPlanA_breakdown(h_input, thr, k_effective, which, b, /*host_output*/nullptr);
            // accumulate
            avg.codes_ms   += b.codes_ms;
            avg.hist_ms    += b.hist_ms;
            avg.scan_ms    += b.scan_ms;
            avg.scatter_ms += b.scatter_ms;
            avg.compact_ms += b.compact_ms;
            avg.kernel_ms  += b.kernel_ms;
            avg.e2e_ms     += b.e2e_ms;
            total_sum      += static_cast<long long>(b.total_out);
        }
        // average
        avg.codes_ms   /= args.repeats;
        avg.hist_ms    /= args.repeats;
        avg.scan_ms    /= args.repeats;
        avg.scatter_ms /= args.repeats;
        avg.compact_ms /= args.repeats;
        avg.kernel_ms  /= args.repeats;
        avg.e2e_ms     /= args.repeats;
        long long total_avg = static_cast<long long>( (double)total_sum / args.repeats + 0.5 );
        csv_rowA(*osp, ablation, which, args, avg, total_avg);
    }
    else if (args.plan=="B" || args.plan=="b"){
        BreakdownPlanB avg{};
        long long total_sum = 0;

        for (int r=0;r<args.repeats;++r){
            BreakdownPlanB b{};
            testPlanB_breakdown(h_input, thr, k_effective, b, /*host_output*/nullptr);
            avg.codes_ms   += b.codes_ms;
            avg.count_ms   += b.count_ms;
            avg.scan_ms    += b.scan_ms;
            avg.reduce_ms  += b.reduce_ms;
            avg.write_ms   += b.write_ms;
            avg.kernel_ms  += b.kernel_ms;
            avg.e2e_ms     += b.e2e_ms;
            total_sum      += static_cast<long long>(b.total_valid);
        }
        avg.codes_ms   /= args.repeats;
        avg.count_ms   /= args.repeats;
        avg.scan_ms    /= args.repeats;
        avg.reduce_ms  /= args.repeats;
        avg.write_ms   /= args.repeats;
        avg.kernel_ms  /= args.repeats;
        avg.e2e_ms     /= args.repeats;
        long long total_avg = static_cast<long long>( (double)total_sum / args.repeats + 0.5 );
        csv_rowB(*osp, ablation, args, avg, total_avg);
    } else {
        std::cerr << "[Q6] Unknown --plan '"<<args.plan<<"' (use A|B)\n";
        return 2;
    }

    return 0;
}
