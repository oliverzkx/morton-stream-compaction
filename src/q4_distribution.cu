// q4_distribution.cu
// Q4: Impact of spatial distribution (uniform / clustered / skewed)
// 复用项目内的：genMortonCodesKernel + histogramBins（低 k 位分桶, 可切高位）
// 以及 testPlanA_breakdown / testPlanB_breakdown（你的 PlanA/PlanB 实现）

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "common.h"                 // Point2D, morton2D_encode(...)
#include "bin_kernel.h"             // BinKernel::Shared/Warp/Bitmask
#include "benchmark_utils.h"        // BreakdownPlanA/B
#include "stream_compaction_bin.h"  // choose_threshold_for_rate, testPlanA/B_breakdown,
                                    // genMortonCodesKernel, histogramBins

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define CUDA_CHECK(x) do { \
  cudaError_t err__ = (x); \
  if (err__ != cudaSuccess) { \
    std::cerr << "[CUDA] " << cudaGetErrorName(err__) << ": " \
              << cudaGetErrorString(err__) \
              << " @ " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::abort(); \
  } \
} while(0)

// ---------- Distributions ----------
enum class DistType { Uniform, Clustered, Skewed };
static const char* dist2s(DistType d){
    switch(d){ case DistType::Uniform: return "uniform";
               case DistType::Clustered: return "clustered";
               case DistType::Skewed:    return "skewed"; }
    return "unknown";
}

static inline float clamp01(float x){ return x<0.f?0.f:(x>1.f?1.f:x); }

static void gen_uniform(std::vector<Point2D>& h, uint64_t seed){
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U01(0.f,1.f), V(-4.f,4.f), T(15.f,40.f);
    for(auto& p: h){ p.x=U01(rng); p.y=U01(rng); p.vx=V(rng); p.vy=V(rng); p.temp=T(rng); }
}
static void gen_clustered(std::vector<Point2D>& h, uint64_t seed, int K=8){
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> Uc(0.15f, 0.85f);
    std::vector<std::pair<float,float>> centers; centers.reserve(K);
    for(int i=0;i<K;++i) centers.emplace_back(Uc(rng), Uc(rng));
    const float sigma = std::max(0.01f, 0.12f/std::sqrt((float)K));
    std::normal_distribution<float> G(0.f, sigma);
    std::uniform_real_distribution<float> V(-4.f,4.f), T(15.f,40.f);
    for(size_t i=0;i<h.size();++i){
        auto& p=h[i]; auto& c=centers[i%K];
        p.x=clamp01(c.first  + G(rng));
        p.y=clamp01(c.second + G(rng));
        p.vx=V(rng); p.vy=V(rng); p.temp=T(rng);
    }
}
static void gen_skewed(std::vector<Point2D>& h, uint64_t seed){
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U01(0.f,1.f), V(-4.f,4.f), T(15.f,40.f);
    for(auto& p: h){
        float u=U01(rng);
        p.x=clamp01(std::pow(u,2.0f));
        p.y=U01(rng);
        p.vx=V(rng); p.vy=V(rng); p.temp=T(rng);
    }
}
static void gen_points(std::vector<Point2D>& h, DistType d, uint64_t seed){
    if(d==DistType::Uniform) gen_uniform(h,seed);
    else if(d==DistType::Clustered) gen_clustered(h,seed);
    else gen_skewed(h,seed);
}

// 关键：把 [0,1] 浮点坐标量化到整数网格 0..(2^k-1)，存回 x/y（float里放整数）
static void quantize_xy_for_bins(std::vector<Point2D>& h, int kBits){
    const int G = 1<<kBits;
    for(auto& p: h){
        int xi = std::min(G-1, std::max(0, (int)std::floor(p.x * G)));
        int yi = std::min(G-1, std::max(0, (int)std::floor(p.y * G)));
        p.x = (float)xi;
        p.y = (float)yi;
    }
}

// ---------- 可选“高位分桶”直方图（只用于 Q4 统计，不影响算法内核） ----------
__global__ void histogram_highbits(const uint32_t* codes, int N, int kBits, int* hist){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    for (int i = t; i < N; i += s) {
        unsigned bin = codes[i] >> kBits;   // 取 Morton 码的“高 k 位”
        atomicAdd(&hist[bin], 1);
    }
}

enum class BinBits { Low, High };

// ---------- Reuse your binning to build histogram (low/high k-bits) ----------
static void build_histogram_via_project(const std::vector<Point2D>& h_in,
                                        int kBits,
                                        BinBits whichBits,
                                        std::vector<int>& h_counts)
{
    const int N = (int)h_in.size();
    const int numBins = 1<<kBits;
    const int mask = numBins - 1;

    Point2D* d_in=nullptr; uint32_t* d_codes=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,   N*sizeof(Point2D)));
    CUDA_CHECK(cudaMalloc(&d_codes,N*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N*sizeof(Point2D), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    // Pass-0: codes（用你的 kernel）
    genMortonCodesKernel<<<blocks, threads>>>(d_in, d_codes, N);
    CUDA_CHECK(cudaGetLastError());

    // Pass-1: histogram（低位 or 高位）
    thrust::device_vector<int> d_binSizes(numBins, 0);
    if (whichBits == BinBits::Low) {
        histogramBins<<<blocks, threads>>>(d_codes,
            thrust::raw_pointer_cast(d_binSizes.data()), N, mask); // 低 k 位（你现有的）
    } else {
        histogram_highbits<<<blocks, threads>>>(d_codes, N, kBits,
            thrust::raw_pointer_cast(d_binSizes.data()));          // 高 k 位（新增）
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // D2H
    h_counts.resize(numBins);
    CUDA_CHECK(cudaMemcpy(h_counts.data(),
                          thrust::raw_pointer_cast(d_binSizes.data()),
                          numBins*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_codes));
}

// ---------- Load-imbalance stats ----------
struct LoadStats { int active_bins; float active_bins_ratio; float max_over_mean; float std_over_mean; };
static LoadStats compute_load_stats(const std::vector<int>& c){
    const int B=(int)c.size(); int active=0, mx=0; long long sum=0;
    for(int v:c){ if(v>0) ++active; if(v>mx) mx=v; sum+=v; }
    const float mean = B? float(sum)/B : 0.f; double var=0.0;
    if(B){ for(int v:c){ double d=v-mean; var+=d*d; } var/=B; }
    const float stdv = std::sqrt((float)var);
    return { active, B? float(active)/B : 0.f, (mean>0)? mx/mean : 0.f, (mean>0)? stdv/mean : 0.f };
}

// ---------- Use your PlanA/PlanB wrappers ----------
struct RunResult{ double kernel_ms{0}, e2e_ms{0}; };

static RunResult run_variant(const std::string& variant,
                             const std::vector<Point2D>& h_input,
                             int kBits, double hit_rate)
{
    RunResult r{};
    const float thr = choose_threshold_for_rate(h_input, hit_rate);

    if (variant == "planB") {
        BreakdownPlanB pb{};
        testPlanB_breakdown(h_input, thr, kBits, pb, /*host_output*/nullptr);
        r.kernel_ms = pb.kernel_ms; r.e2e_ms = pb.e2e_ms; return r;
    }
    if (variant == "planA_shared") {
        BreakdownPlanA pa{};
        testPlanA_breakdown(h_input, thr, kBits, BinKernel::Shared, pa, nullptr);
        r.kernel_ms = pa.kernel_ms; r.e2e_ms = pa.e2e_ms; return r;
    }
    if (variant == "planA_warp") {
        BreakdownPlanA pa{};
        testPlanA_breakdown(h_input, thr, kBits, BinKernel::Warp, pa, nullptr);
        r.kernel_ms = pa.kernel_ms; r.e2e_ms = pa.e2e_ms; return r;
    }
    if (variant == "planA_bitmask") {
        BreakdownPlanA pa{};
        testPlanA_breakdown(h_input, thr, kBits, BinKernel::Bitmask, pa, nullptr);
        r.kernel_ms = pa.kernel_ms; r.e2e_ms = pa.e2e_ms; return r;
    }
    // thrust 基线（可选）：没有就不跑
    return r;
}

// ---------- CSV ----------
static void write_csv_header(std::ofstream& os){
    os<<"plan,variant,dist,kBits,N,hit_rate,"
      <<"active_bins,active_bins_ratio,max_over_mean,std_over_mean,"
      <<"kernel_ms,e2e_ms\n";
}
static void write_csv_row(std::ofstream& os,
    const std::string& plan,const std::string& variant,DistType dist,
    int kBits,long long N,double hit_rate,const LoadStats& s,const RunResult& r){
    os<<plan<<","<<variant<<","<<dist2s(dist)<<","<<kBits<<","<<N<<","<<hit_rate<<","
      <<s.active_bins<<","<<s.active_bins_ratio<<","<<s.max_over_mean<<","<<s.std_over_mean<<","
      <<r.kernel_ms<<","<<r.e2e_ms<<"\n";
}

// ---------- CLI（避免与其它 TUs 的 Args 冲突） ----------
struct Q4Args{
    long long N=10'000'000; int kBits=8; double hit_rate=0.50;
    DistType dist=DistType::Uniform; std::string csv="csv/q4_uniform.csv";
    std::string plan="PlanX"; uint64_t seed=1234;
    std::string binbits="low"; // 新增：统计口径（low|high），默认 low（与你现有一致）
    std::vector<std::string> variants={"planB","planA_shared","planA_warp","planA_bitmask"};
};
static DistType parseDist(const std::string& s){
    if(s=="uniform") return DistType::Uniform;
    if(s=="clustered")return DistType::Clustered;
    if(s=="skewed")  return DistType::Skewed;
    throw std::runtime_error("bad --dist (uniform|clustered|skewed)");
}
static Q4Args parse_cli_q4(int argc,char** argv){
    Q4Args a;
    for(int i=1;i<argc;++i){
        std::string arg=argv[i];
        if(arg=="--N" && i+1<argc) a.N=std::stoll(argv[++i]);
        else if(arg=="--k" && i+1<argc) a.kBits=std::stoi(argv[++i]);
        else if(arg=="--rates" && i+1<argc) a.hit_rate=std::stod(argv[++i]);
        else if(arg=="--dist" && i+1<argc) a.dist=parseDist(argv[++i]);
        else if(arg=="--csv" && i+1<argc) a.csv=argv[++i];
        else if(arg=="--seed" && i+1<argc) a.seed=std::stoull(argv[++i]);
        else if(arg=="--binbits" && i+1<argc) a.binbits=argv[++i]; // 新增
        else if(arg=="--variants" && i+1<argc){
            a.variants.clear(); std::string s=argv[++i];
            size_t b=0; while(true){ size_t e=s.find(',',b);
                a.variants.emplace_back(s.substr(b, e==std::string::npos? s.size()-b : e-b));
                if(e==std::string::npos) break; b=e+1;
            }
        }
    }
    return a;
}

// ---------- Main ----------
int main(int argc,char** argv){
    Q4Args args = parse_cli_q4(argc,argv);
    std::ofstream os(args.csv, std::ios::out);
    write_csv_header(os);

    // 1) 生成点（[0,1] 空间）→ 2) 量化到整数网格（与你的代码一致的输入格式）
    std::vector<Point2D> h_in(args.N);
    gen_points(h_in, args.dist, args.seed);
    quantize_xy_for_bins(h_in, args.kBits);

    // 3) 用你自己的分桶阶段计算直方图（低/高 k 位，按 CLI 选择）
    std::vector<int> counts;
    const BinBits which = (args.binbits=="high") ? BinBits::High : BinBits::Low;
    build_histogram_via_project(h_in, args.kBits, which, counts);

    // 4) 负载不均衡统计
    LoadStats stats = compute_load_stats(counts);
    long long sum=0; for(int c:counts) sum+=c;
    if (sum != args.N) {
        std::cerr << "[Q4] WARNING: histogram sum=" << sum << " but N=" << args.N << "\n";
    }

    // 5) 跑你的 PlanA/PlanB 变体
    for(const auto& v : args.variants){
        RunResult r = run_variant(v, h_in, args.kBits, args.hit_rate);
        write_csv_row(os, args.plan, v, args.dist, args.kBits, args.N, args.hit_rate, stats, r);
        std::cout<<"[Q4] dist="<<dist2s(args.dist)<<" variant="<<v
                 <<" kernel_ms="<<r.kernel_ms<<" e2e_ms="<<r.e2e_ms<<"\n";
    }
    return 0;
}
