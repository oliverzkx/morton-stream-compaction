// src/q5_scalability.cu
// Q5 — Scalability (dataset size sweep) for Morton-binned GPU stream compaction
//
// This benchmark mirrors the style of q3/q4 and integrates directly with your
// existing wrappers:
//   - choose_threshold_for_rate(...)
//   - testPlanA_breakdown(...)
//   - testPlanB_breakdown(...)
// from stream_compaction_bin.{h,cu} and uses common.h (not types.h).
//
// Build target suggested in Makefile as: build/q5_scalability
// Example run:
//   ./build/q5_scalability \
//       --Ns 1M,5M,10M,20M,50M \
//       --k 8 --hit 0.50 --dist uniform \
//       --variants planB,planA_shared,planA_warp,planA_bitmask \
//       --block 256 --repeat 5 --seed 1234 \
//       --csv csv/q5_scaling.csv --roofline proxy_v1
//
// CSV columns:
//   plan,variant,kBits,blockSize,N,hit_rate,repeat,seed,dist,
//   kernel_ms,e2e_ms,kernel_Melps,e2e_Melps,bytes_est,achieved_BW_GBps,roofline

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cstdint>
#include <cmath>

#include "common.h"                 // Point2D, set_block_size(...)
#include "bin_kernel.h"             // BinKernel::Shared/Warp/Bitmask
#include "benchmark_utils.h"        // BreakdownPlanA/B
#include "stream_compaction_bin.h"  // choose_threshold_for_rate, testPlanA/B_breakdown

// -------------------- dataset generators (local, avoid TU collisions) --------------------
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
        float x = cx + sigma*N01(rng);
        float y = cy + sigma*N01(rng);
        x = std::min(1.f,std::max(0.f,x));
        y = std::min(1.f,std::max(0.f,y));
        Point2D p{}; p.x=x; p.y=y; p.vx=V(rng); p.vy=V(rng); p.temp=T(rng); v[i]=p;
    }
    return v;
}

static std::vector<Point2D> make_skewed_dataset(std::size_t N, double heavy_frac=0.8, int window=8, uint64_t seed=91011) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U01(0.f,1.f), V(-4.f,4.f), T(15.f,40.f);
    std::vector<Point2D> v(N);
    // push heavy_frac of points into a narrow x-window
    std::uniform_real_distribution<float> Xh(0.f, 1.f/float(window));
    std::uniform_int_distribution<int> pick(0, window-1);
    for (std::size_t i=0;i<N;++i){
        bool heavy = (double)i/N < heavy_frac;
        float x = heavy ? (pick(rng) + Xh(rng)) / float(window) : U01(rng);
        float y = U01(rng);
        Point2D p{}; p.x=x; p.y=y; p.vx=V(rng); p.vy=V(rng); p.temp=T(rng); v[i]=p;
    }
    return v;
}

static std::vector<Point2D> make_dataset(std::size_t N, const std::string& dist, uint64_t seed){
    if (dist=="uniform")   return make_uniform_dataset(N, seed);
    if (dist=="clustered") return make_clustered_dataset(N, 8, 0.03f, seed);
    if (dist=="skewed")    return make_skewed_dataset(N, 0.8, 8, seed);
    std::cerr << "[Q5] Unknown --dist='"<<dist<<"', fallback to uniform.\n";
    return make_uniform_dataset(N, seed);
}

// -------------------- CLI parsing --------------------
struct Q5Args{
    std::vector<long long> Ns{1'000'000LL, 5'000'000LL, 10'000'000LL, 20'000'000LL, 50'000'000LL};
    int kBits=8; double hit_rate=0.50; std::string dist="uniform"; uint64_t seed=1234;
    std::vector<std::string> variants{"planB","planA_shared","planA_warp","planA_bitmask"};
    int blockSize=256; int repeat=5; std::string csv="csv/q5_scaling.csv"; std::string roof="proxy_v1";
};

static std::vector<long long> parse_Ns(const std::string& s){
    std::vector<long long> out; std::stringstream ss(s); std::string tok;
    while(std::getline(ss,tok,',')){
        if(tok.empty()) continue; bool hasM = (tok.back()=='M' || tok.back()=='m');
        long long v = std::stoll(hasM? tok.substr(0,tok.size()-1) : tok);
        if(hasM) v *= 1'000'000LL; out.push_back(v);
    }
    return out;
}

static Q5Args parse_cli(int argc, char** argv){
    Q5Args a; for(int i=1;i<argc;++i){ std::string k=argv[i]; auto need=[&](int n){ if(i+n>=argc){ std::cerr<<"Missing value for "<<k<<"\n"; std::exit(2);} }; 
        if(k=="--Ns"){ need(1); a.Ns = parse_Ns(argv[++i]); }
        else if(k=="--k"){ need(1); a.kBits = std::stoi(argv[++i]); }
        else if(k=="--hit"){ need(1); a.hit_rate = std::stod(argv[++i]); }
        else if(k=="--dist"){ need(1); a.dist = argv[++i]; }
        else if(k=="--seed"){ need(1); a.seed = (uint64_t)std::stoull(argv[++i]); }
        else if(k=="--variants"){ need(1); a.variants.clear(); std::stringstream ss(argv[++i]); std::string t; while(std::getline(ss,t,',')) if(!t.empty()) a.variants.push_back(t); }
        else if(k=="--block"){ need(1); a.blockSize = std::stoi(argv[++i]); }
        else if(k=="--repeat"){ need(1); a.repeat = std::stoi(argv[++i]); }
        else if(k=="--csv"){ need(1); a.csv = argv[++i]; }
        else if(k=="--roofline"){ need(1); a.roof = argv[++i]; }
        else { std::cerr<<"Unknown arg: "<<k<<"\n"; std::exit(2);} }
    return a;
}

// -------------------- small helpers --------------------
struct Meas{ float kernel_ms=0.f, e2e_ms=0.f; int out_count=0; };

static inline double melps(long long N, double ms){ return (ms>0)? double(N)/(ms*1e3) : 0.0; }

static inline double estimate_bytes_proxy_v1(const char* plan,
                                             const char* variant,
                                             long long N, double hit_rate){
    const double P = double(sizeof(Point2D));
    const double kept = double(N) * hit_rate;
    if (std::string(plan)=="PlanB") {
        // One pass: read N*P, write kept*P
        return double(N)*P + kept*P;
    } else {
        // Multi-pass (codes + hist/scan + scatter + compact)
        // Conservatively: 3 full reads of N*P (pre-scatter, scatter out, compact in)
        // + kept*P writes for compact output + small metadata overhead
        const double meta = double(N) * 8.0; // bytes; codes/scan ints etc.
        return 3.0*double(N)*P + kept*P + meta;
    }
}

static Meas run_variant_once(const std::string& variant,
                             const std::vector<Point2D>& h_input,
                             int kBits, double hit_rate){
    Meas m{}; const float thr = choose_threshold_for_rate(h_input, hit_rate);
    if (variant=="planB"){
        BreakdownPlanB pb{}; testPlanB_breakdown(h_input, thr, kBits, pb, /*host_out*/nullptr);
        m.kernel_ms = pb.kernel_ms; m.e2e_ms = pb.e2e_ms; m.out_count = pb.total_valid; return m;
    }
    if (variant=="planA_shared"){
        BreakdownPlanA pa{}; testPlanA_breakdown(h_input, thr, kBits, BinKernel::Shared, pa, nullptr);
        m.kernel_ms = pa.kernel_ms; m.e2e_ms = pa.e2e_ms; m.out_count = pa.total_out; return m;
    }
    if (variant=="planA_warp"){
        BreakdownPlanA pa{}; testPlanA_breakdown(h_input, thr, kBits, BinKernel::Warp, pa, nullptr);
        m.kernel_ms = pa.kernel_ms; m.e2e_ms = pa.e2e_ms; m.out_count = pa.total_out; return m;
    }
    if (variant=="planA_bitmask"){
        BreakdownPlanA pa{}; testPlanA_breakdown(h_input, thr, kBits, BinKernel::Bitmask, pa, nullptr);
        m.kernel_ms = pa.kernel_ms; m.e2e_ms = pa.e2e_ms; m.out_count = pa.total_out; return m;
    }
    std::cerr << "[Q5] Unknown variant '"<<variant<<"'\n"; return m;
}

// -------------------- main --------------------
int main(int argc, char** argv){
    Q5Args args = parse_cli(argc, argv);
    // honour user block size (used by your kernels via g_block_size)
    set_block_size(args.blockSize);

    // open CSV
    std::ofstream os(args.csv);
    if(!os){ std::cerr << "[Q5] Cannot open CSV path: "<<args.csv<<"\n"; return 1; }
    os << "plan,variant,kBits,blockSize,N,hit_rate,repeat,seed,dist,"
          "kernel_ms,e2e_ms,kernel_Melps,e2e_Melps,bytes_est,achieved_BW_GBps,roofline\n";

    for(const auto& variant : args.variants){
        const char* plan = (variant=="planB") ? "PlanB" : "PlanA";
        for(long long N : args.Ns){
            // Build dataset once per N (per Q5, dist fixed; seed varied with N to avoid correlation)
            auto h_input = make_dataset((std::size_t)N, args.dist, args.seed + (uint64_t)N);

            // Warm-up once
            (void)run_variant_once(variant, h_input, args.kBits, args.hit_rate);

            // Repeat & average
            double k_ms=0, e_ms=0; int out_cnt=0;
            for(int r=0;r<args.repeat;++r){
                auto m = run_variant_once(variant, h_input, args.kBits, args.hit_rate);
                k_ms += m.kernel_ms; e_ms += m.e2e_ms; out_cnt = m.out_count; // last count
            }
            k_ms /= std::max(1, args.repeat);
            e_ms /= std::max(1, args.repeat);

            const double k_Melps = melps(N, k_ms);
            const double e_Melps = melps(N, e_ms);

            const double bytes   = (args.roof=="proxy_v1")
                                  ? estimate_bytes_proxy_v1(plan, variant.c_str(), N, args.hit_rate)
                                  : 0.0;
            const double bw_GBps = (k_ms>0) ? (bytes / (k_ms/1e3)) / 1e9 : 0.0;

            os << plan << ',' << variant << ','
               << args.kBits << ',' << args.blockSize << ',' << N << ','
               << std::fixed << std::setprecision(2) << args.hit_rate << ','
               << args.repeat << ',' << args.seed << ',' << args.dist << ','
               << std::setprecision(4) << k_ms << ',' << e_ms << ','
               << std::setprecision(6) << k_Melps << ',' << e_Melps << ','
               << std::setprecision(0) << bytes << ','
               << std::setprecision(3) << bw_GBps << ',' << args.roof
               << "\n";
            os.flush();

            std::cout << "[Q5] variant="<<variant
                      << " N="<<N
                      << " kernel_ms="<<std::setprecision(4)<<k_ms
                      << " e2e_ms="<<e_ms
                      << " | kernel Mel/s="<<std::setprecision(3)<<k_Melps
                      << " BW(GiB/s est)="<<std::setprecision(3)<<bw_GBps << "\n";

            (void)out_cnt; // reserved for optional sanity checks
        }
    }

    std::cout << "[Q5] CSV written: " << args.csv << "\n";
    return 0;
}