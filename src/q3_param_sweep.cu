// src/q3_param_sweep.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <random>
#include <utility>
#include <cmath>

#include "common.h"               // g_block_size, set_block_size, Point2D
#include "stream_compaction_bin.h"// testPlanA_breakdown / testPlanB_breakdown
#include "bin_kernel.h"           // BinKernel
#include "benchmark_utils.h"      // choose_threshold_for_rate(...)

struct Meas { float kernel_ms, e2e_ms; int out; };

static void write_header(std::ostream& os) {
    os << "plan,variant,kernel,kBits,blockSize,hit_rate,N,"
          "ms_kernel,ms_e2e,thrpt_kernel_Meps,thrpt_e2e_Meps,out_count\n";
}

static Meas run_planB(const std::vector<Point2D>& input, float thr, int kBits) {
    BreakdownPlanB b{}; testPlanB_breakdown(input, thr, kBits, b, nullptr);
    return {b.kernel_ms, b.e2e_ms, b.total_valid};
}
static Meas run_planA(const std::vector<Point2D>& input, float thr, int kBits, BinKernel kind) {
    BreakdownPlanA b{}; testPlanA_breakdown(input, thr, kBits, kind, b, nullptr);
    return {b.kernel_ms, b.e2e_ms, b.total_out};
}

static void dump(std::ostream& os, const char* plan, const char* variant, const char* kernel,
                 int kBits, int blockSize, double hit, size_t N, const Meas& m) {
    const double tK = std::max(1e-6, m.kernel_ms/1000.0);
    const double tE = std::max(1e-6, m.e2e_ms/1000.0);
    const double tk = double(N)/tK/1e6;   // M elems/s
    const double te = double(N)/tE/1e6;
    os << plan << ',' << variant << ',' << kernel << ','
       << kBits << ',' << blockSize << ','
       << std::fixed << std::setprecision(2) << hit << ','
       << N << ','
       << std::setprecision(3) << m.kernel_ms << ',' << m.e2e_ms << ','
       << tk << ',' << te << ',' << m.out << "\n";
}

// ----------------- Local dataset helpers (same as before) -------------------
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Uniform
static std::vector<Point2D> make_uniform_dataset(std::size_t N, uint32_t seed=1234) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> XY(0, 65535);
  std::uniform_real_distribution<float> F01(0.f, 1.f);
  std::vector<Point2D> v(N);
  for (std::size_t i = 0; i < N; ++i) {
    Point2D p{};
    p.x = static_cast<float>(XY(rng));
    p.y = static_cast<float>(XY(rng));
    p.vx = 0.f; p.vy = 0.f;
    p.temp = F01(rng);
    v[i] = p;
  }
  return v;
}
// Clustered
static std::vector<Point2D> make_clustered_dataset(std::size_t N, int K = 32, float sigma = 300.f, uint32_t seed=5678) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> XYc(0, 65535);
  std::normal_distribution<float> N01(0.f, 1.f);
  std::uniform_real_distribution<float> F01(0.f, 1.f);
  std::vector<std::pair<int,int>> centers; centers.reserve(K);
  for (int i = 0; i < K; ++i) centers.emplace_back(XYc(rng), XYc(rng));
  std::uniform_int_distribution<int> pick(0, K-1);
  std::vector<Point2D> v(N);
  for (std::size_t i = 0; i < N; ++i) {
    auto c = centers[pick(rng)];
    float xf = static_cast<float>(c.first)  + sigma * N01(rng);
    float yf = static_cast<float>(c.second) + sigma * N01(rng);
    Point2D p{}; p.x = float(clampi((int)std::lround(xf), 0, 65535)); p.y = float(clampi((int)std::lround(yf), 0, 65535));
    p.vx = 0.f; p.vy = 0.f; p.temp = F01(rng); v[i] = p;
  }
  return v;
}
// Skewed
static std::vector<Point2D> make_skewed_dataset(std::size_t N, double heavy_frac = 0.9, int window = 1024, uint32_t seed=9012) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> XY(0, 65535);
  std::uniform_real_distribution<float> F01(0.f, 1.f);
  std::bernoulli_distribution heavy(heavy_frac);
  int cx = XY(rng), cy = XY(rng);
  std::uniform_int_distribution<int> Xw(std::max(0, cx-window), std::min(65535, cx+window));
  std::uniform_int_distribution<int> Yw(std::max(0, cy-window), std::min(65535, cy+window));
  std::vector<Point2D> v(N);
  for (std::size_t i = 0; i < N; ++i) {
    bool h = heavy(rng);
    int xi = h ? Xw(rng) : XY(rng);
    int yi = h ? Yw(rng) : XY(rng);
    Point2D p{}; p.x = float(xi); p.y = float(yi); p.vx = 0.f; p.vy = 0.f; p.temp = F01(rng); v[i] = p;
  }
  return v;
}
static std::vector<Point2D> make_dataset(std::size_t N, const std::string& dist) {
  if (dist == "uniform")   return make_uniform_dataset(N);
  if (dist == "clustered") return make_clustered_dataset(N);
  if (dist == "skewed")    return make_skewed_dataset(N);
  std::cerr << "Unknown distribution '" << dist << "', using uniform.\n";
  return make_uniform_dataset(N);
}
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // -------- CLI: --sweep block|kbits|hit (default=block), --repeats N --------
    std::string sweep = "block";
    int repeats = 3;
    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a == "--sweep" && i+1<argc) { sweep = argv[++i]; continue; }
        if (a == "--repeats" && i+1<argc) { repeats = std::max(1, atoi(argv[++i])); continue; }
    }

    // ---- baseline ----
    const size_t N = 20000000;             // 2e7
    const std::string dist = "uniform";

    // Lists for the three sweeps
    const std::vector<int>    blockList = {64,128,256,384,512,768,1024};
    const std::vector<int>    kList     = {6,7,8,9,10};
    const std::vector<double> hitList   = {0.05,0.50,0.95};

    // Shared input & (for kbits/block sweeps) shared threshold
    auto input = make_dataset(N, dist);

    // ---------------------- blockSize sweep ----------------------
    if (sweep == "block") {
        const int kBits = 8;
        const double hit = 0.50;
        float thr = choose_threshold_for_rate(input, hit);

        std::ofstream os("csv/q3_block.csv"); write_header(os);
        for (int bs : blockList) {
            set_block_size(bs);

            // Plan B
            Meas mb{0,0,0};
            for (int r=0;r<repeats;r++){ auto m=run_planB(input,thr,kBits); mb.kernel_ms+=m.kernel_ms; mb.e2e_ms+=m.e2e_ms; mb.out=m.out; }
            mb.kernel_ms/=repeats; mb.e2e_ms/=repeats; dump(os,"PlanB","atomic","-",kBits,bs,hit,N,mb);

            // Plan A
            for (auto kv : { std::pair{BinKernel::Shared,"shared"},
                             std::pair{BinKernel::Warp,"warp"},
                             std::pair{BinKernel::Bitmask,"bitmask"} }) {
                Meas ma{0,0,0};
                for (int r=0;r<repeats;r++){ auto m=run_planA(input,thr,kBits,kv.first); ma.kernel_ms+=m.kernel_ms; ma.e2e_ms+=m.e2e_ms; ma.out=m.out; }
                ma.kernel_ms/=repeats; ma.e2e_ms/=repeats; dump(os,"PlanA","partition",kv.second,kBits,bs,hit,N,ma);
            }
            std::cerr << "[Q3][block] bs=" << bs << " done\n";
        }
        std::cerr << "[Q3] sweep blockSize done. -> csv/q3_block.csv\n";
        return 0;
    }

    // ---------------------- kBits sweep ----------------------
    if (sweep == "kbits") {
        const int blockSize = 256; set_block_size(blockSize);
        const double hit = 0.50;  float thr = choose_threshold_for_rate(input, hit);

        std::ofstream os("csv/q3_kbits.csv"); write_header(os);
        for (int kBits : kList) {
            // Plan B
            Meas mb{0,0,0};
            for (int r=0;r<repeats;r++){ auto m=run_planB(input,thr,kBits); mb.kernel_ms+=m.kernel_ms; mb.e2e_ms+=m.e2e_ms; mb.out=m.out; }
            mb.kernel_ms/=repeats; mb.e2e_ms/=repeats; dump(os,"PlanB","atomic","-",kBits,blockSize,hit,N,mb);

            // Plan A
            for (auto kv : { std::pair{BinKernel::Shared,"shared"},
                             std::pair{BinKernel::Warp,"warp"},
                             std::pair{BinKernel::Bitmask,"bitmask"} }) {
                Meas ma{0,0,0};
                for (int r=0;r<repeats;r++){ auto m=run_planA(input,thr,kBits,kv.first); ma.kernel_ms+=m.kernel_ms; ma.e2e_ms+=m.e2e_ms; ma.out=m.out; }
                ma.kernel_ms/=repeats; ma.e2e_ms/=repeats; dump(os,"PlanA","partition",kv.second,kBits,blockSize,hit,N,ma);
            }
            std::cerr << "[Q3][kbits] k=" << kBits << " done\n";
        }
        std::cerr << "[Q3] sweep kBits done. -> csv/q3_kbits.csv\n";
        return 0;
    }

    // ---------------------- hit-rate sweep ----------------------
    if (sweep == "hit") {
        const int kBits = 8;
        const int blockSize = 256; set_block_size(blockSize);

        std::ofstream os("csv/q3_hit.csv"); write_header(os);
        for (double hit : hitList) {
            float thr = choose_threshold_for_rate(input, hit);

            // Plan B
            Meas mb{0,0,0};
            for (int r=0;r<repeats;r++){ auto m=run_planB(input,thr,kBits); mb.kernel_ms+=m.kernel_ms; mb.e2e_ms+=m.e2e_ms; mb.out=m.out; }
            mb.kernel_ms/=repeats; mb.e2e_ms/=repeats; dump(os,"PlanB","atomic","-",kBits,blockSize,hit,N,mb);

            // Plan A
            for (auto kv : { std::pair{BinKernel::Shared,"shared"},
                             std::pair{BinKernel::Warp,"warp"},
                             std::pair{BinKernel::Bitmask,"bitmask"} }) {
                Meas ma{0,0,0};
                for (int r=0;r<repeats;r++){ auto m=run_planA(input,thr,kBits,kv.first); ma.kernel_ms+=m.kernel_ms; ma.e2e_ms+=m.e2e_ms; ma.out=m.out; }
                ma.kernel_ms/=repeats; ma.e2e_ms/=repeats; dump(os,"PlanA","partition",kv.second,kBits,blockSize,hit,N,ma);
            }
            std::cerr << "[Q3][hit] hit=" << hit << " done\n";
        }
        std::cerr << "[Q3] sweep hit-rate done. -> csv/q3_hit.csv\n";
        return 0;
    }

    // Bad arg fallback
    std::cerr << "Unknown --sweep mode: " << sweep << " (use block|kbits|hit)\n";
    return 1;
}
