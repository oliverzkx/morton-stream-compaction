// src/q7_proxy_bins_agg.cu
// Chapter 7 — Case Study: Per-Bin Aggregation Proxy
// Demonstrate atomic baseline vs. slice-based single-pass aggregation
// after Morton-binned partitioning.
//
// Build (add target in Makefile similar to q4/q5):
//   $(NVCC) $(CXXFLAGS) $(INCLUDES) -o build/q7_proxy src/q7_proxy_bins_agg.cu \
//       src/stream_compaction_bin.cu src/utils.cu src/morton.cu
//
// Example run:
//   ./build/q7_proxy --N 20000000 --k 8 --hit 0.50 --dist uniform \
//                    --block 256 --repeat 5 --csv csv/q7_proxy.csv
//
// CSV columns:
//   impl,dist,kBits,N,hit_rate,codes_ms,offsets_ms,scatter_ms,agg_ms,e2e_ms,count_sum,avg_temp,max_temp

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#include "common.h"
#include "stream_compaction_bin.h"  // genMortonCodesKernel, computeBinOffsets, scatterToBins, choose_threshold_for_rate

// ---------------- CLI args ----------------
struct Args {
    long long N = 20000000;
    int kBits = 8;
    double hit = 0.50;
    std::string dist = "uniform";   // uniform|clustered|skewed
    int block = 256;
    int repeat = 5;
    std::string csv = "csv/q7_proxy.csv";
    uint64_t seed = 1234;
};

static Args parse_args(int argc, char** argv){
    Args a;
    auto need=[&](int i){ if(i>=argc){ std::cerr<<"Missing value\n"; std::exit(2);} };
    for(int i=1;i<argc;++i){
        std::string k=argv[i];
        if(k=="--N"){ need(i+1); std::string s=argv[++i];
            if(!s.empty() && (s.back()=='M'||s.back()=='m')){ s.pop_back(); a.N = (long long)(std::stoll(s)*1000000LL); }
            else if(!s.empty() && (s.back()=='K'||s.back()=='k')){ s.pop_back(); a.N = (long long)(std::stoll(s)*1000LL); }
            else { a.N = std::stoll(s); } }
        else if(k=="--k"){ need(i+1); a.kBits = std::stoi(argv[++i]); }
        else if(k=="--hit"){ need(i+1); a.hit = std::stod(argv[++i]); }
        else if(k=="--dist"){ need(i+1); a.dist = argv[++i]; }
        else if(k=="--block"){ need(i+1); a.block = std::stoi(argv[++i]); }
        else if(k=="--repeat"){ need(i+1); a.repeat = std::stoi(argv[++i]); }
        else if(k=="--csv"){ need(i+1); a.csv = argv[++i]; }
        else if(k=="--seed"){ need(i+1); a.seed = (uint64_t)std::stoull(argv[++i]); }
        else { std::cerr<<"Unknown arg: "<<k<<"\n"; std::exit(2); }
    }
    return a;
}

// ------------- data generation (reuse Q4 style) -------------
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
static void quantize_xy_for_bins(std::vector<Point2D>& h, int kBits){
    const int G = 1<<kBits;
    for(auto& p: h){
        int xi = std::min(G-1, std::max(0, (int)std::floor(p.x * G)));
        int yi = std::min(G-1, std::max(0, (int)std::floor(p.y * G)));
        p.x = (float)xi; p.y = (float)yi;
    }
}
static std::vector<Point2D> make_dataset(long long N, const std::string& dist, uint64_t seed, int kBits){
    std::vector<Point2D> h; h.resize((size_t)N);
    if(dist=="uniform") gen_uniform(h, seed);
    else if(dist=="clustered") gen_clustered(h, seed);
    else gen_skewed(h, seed);
    quantize_xy_for_bins(h, kBits);
    return h;
}

// ------------------ device helpers & kernels ------------------
__device__ inline void atomicMaxFloat(float* addr, float val){
    int* ai = reinterpret_cast<int*>(addr);
    int old = *ai, assumed;
    while(true){
        assumed = old;
        float cur = __int_as_float(assumed);
        if (cur >= val) break;
        int next = __float_as_int(val);
        old = atomicCAS(ai, assumed, next);
        if (old == assumed) break;
    }
}

// Atomic baseline: each element updates its bin's count/sum/max
__global__ void binsAgg_atomic(const Point2D* __restrict__ in,
                               const uint32_t* __restrict__ codes,
                               int N, int mask, float thr,
                               int* __restrict__ binCount,
                               float* __restrict__ binSum,
                               float* __restrict__ binMax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int b = codes[i] & mask;
    float t = in[i].temp;
    if (t > thr){
        atomicAdd(&binCount[b], 1);
        atomicAdd(&binSum[b], t);
        atomicMaxFloat(&binMax[b], t);
    }
}

// Slice-based pass: one block per bin, iterate over its contiguous slice
__global__ void binsAgg_slices(const Point2D* __restrict__ binsBuf,
                               const int* __restrict__ offsets, // len=numBins+1
                               int numBins, float thr,
                               int* __restrict__ count,
                               float* __restrict__ sum,
                               float* __restrict__ vmax)
{
    int b = blockIdx.x;
    if(b >= numBins) return;
    int start = offsets[b];
    int end   = offsets[b+1];
    if(start >= end){
        if(threadIdx.x==0){ count[b]=0; sum[b]=0.f; vmax[b]=-INFINITY; }
        return;
    }

    // Per-thread partials
    float lsum = 0.f;
    float lmax = -INFINITY;
    int   lcnt = 0;

    for(int i = start + threadIdx.x; i < end; i += blockDim.x){
        float t = binsBuf[i].temp;
        if (t > thr){ lsum += t; lmax = fmaxf(lmax, t); lcnt += 1; }
    }

    extern __shared__ unsigned char smem[];
    float* ssum = reinterpret_cast<float*>(smem);
    float* smax = ssum + blockDim.x;
    int*   scnt = reinterpret_cast<int*>(smax + blockDim.x);

    ssum[threadIdx.x] = lsum;
    smax[threadIdx.x] = lmax;
    scnt[threadIdx.x] = lcnt;
    __syncthreads();

    // Block-wide reduction
    for(int s = blockDim.x >> 1; s > 0; s >>= 1){
        if(threadIdx.x < s){
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
            scnt[threadIdx.x] += scnt[threadIdx.x + s];
            smax[threadIdx.x]  = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        count[b] = scnt[0];
        sum[b]   = ssum[0];
        vmax[b]  = smax[0];
    }
}

// ------------------ run one round ------------------
struct Meas {
    float codes_ms=0.f, offsets_ms=0.f, scatter_ms=0.f, agg_ms=0.f, e2e_ms=0.f;
    long long count_sum=0;
    double avg_temp=0.0;
    float max_temp=-INFINITY;
};

static Meas run_atomic_round(const std::vector<Point2D>& h, int kBits, double hit, int block){
    // Use project-provided choose_threshold_for_rate (declared in stream_compaction_bin.h)
    float thr = choose_threshold_for_rate(h, hit);

    Meas m{};
    const int N = (int)h.size();
    const int numBins = 1<<kBits;
    const int mask = numBins - 1;

    // Host -> device
    Point2D* d_in=nullptr; uint32_t* d_codes=nullptr;
    cudaMalloc(&d_in,    N*sizeof(Point2D));
    cudaMalloc(&d_codes, N*sizeof(uint32_t));
    cudaMemcpy(d_in, h.data(), N*sizeof(Point2D), cudaMemcpyHostToDevice);

    // Generate morton codes
    dim3 t1(256), b1((N+255)/256);
    cudaEvent_t e0,e1,e2,e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventRecord(e0);
    genMortonCodesKernel<<<b1,t1>>>(d_in, d_codes, N);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&m.codes_ms, e0, e1);

    // Allocate result arrays and init
    int* d_count=nullptr; float* d_sum=nullptr; float* d_max=nullptr;
    cudaMalloc(&d_count, numBins*sizeof(int));
    cudaMalloc(&d_sum,   numBins*sizeof(float));
    cudaMalloc(&d_max,   numBins*sizeof(float));
    cudaMemset(d_count, 0, numBins*sizeof(int));
    cudaMemset(d_sum,   0, numBins*sizeof(float));
    // set d_max to -INF
    std::vector<float> h_min(numBins, -INFINITY);
    cudaMemcpy(d_max, h_min.data(), numBins*sizeof(float), cudaMemcpyHostToDevice);

    // Launch aggregation (kernel-only)
    dim3 t(block), b((N + block - 1)/block);
    cudaEventRecord(e2);
    binsAgg_atomic<<<b,t>>>(d_in, d_codes, N, mask, thr, d_count, d_sum, d_max);
    cudaEventRecord(e3); cudaEventSynchronize(e3);
    cudaEventElapsedTime(&m.agg_ms, e2, e3);

    // e2e = codes + agg (no offsets/scatter needed)
    m.e2e_ms = m.codes_ms + m.agg_ms;

    // Copy back and compute sanity stats
    std::vector<int>    h_count(numBins);
    std::vector<float>  h_sum  (numBins);
    std::vector<float>  h_max  (numBins);
    cudaMemcpy(h_count.data(), d_count, numBins*sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum.data(),   d_sum,   numBins*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max.data(),   d_max,   numBins*sizeof(float), cudaMemcpyDeviceToHost);

    long long cnt=0; double s=0.0; float mx=-INFINITY;
    for(int b2=0;b2<numBins;++b2){ cnt += h_count[b2]; s += (double)h_sum[b2]; mx = fmaxf(mx, h_max[b2]); }
    m.count_sum = cnt;
    m.avg_temp = (cnt>0)? (s / (double)cnt) : 0.0;
    m.max_temp = mx;

    // Cleanup
    cudaFree(d_in); cudaFree(d_codes); cudaFree(d_count); cudaFree(d_sum); cudaFree(d_max);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    return m;
}

static Meas run_slices_round(const std::vector<Point2D>& h, int kBits, double hit, int block){
    // Use project-provided choose_threshold_for_rate
    float thr = choose_threshold_for_rate(h, hit);

    Meas m{};
    const int N = (int)h.size();
    const int numBins = 1<<kBits;

    Point2D* d_in=nullptr; uint32_t* d_codes=nullptr;
    cudaMalloc(&d_in,    N*sizeof(Point2D));
    cudaMalloc(&d_codes, N*sizeof(uint32_t));
    cudaMemcpy(d_in, h.data(), N*sizeof(Point2D), cudaMemcpyHostToDevice);

    // events
    cudaEvent_t e0,e1,e2,e3,e4; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3); cudaEventCreate(&e4);

    // (1) morton codes
    dim3 t1(256), b1((N+255)/256);
    cudaEventRecord(e0);
    genMortonCodesKernel<<<b1,t1>>>(d_in, d_codes, N);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&m.codes_ms, e0, e1);

    // (2) offsets via helper (measure with chrono)
    const auto t_start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<int> d_offsets(numBins + 1);
    thrust::device_vector<int> d_sizes  (numBins);
    computeBinOffsets(d_codes, N, kBits,
                      thrust::raw_pointer_cast(d_offsets.data()),
                      thrust::raw_pointer_cast(d_sizes.data()));
    const auto t_end = std::chrono::high_resolution_clock::now();
    m.offsets_ms = std::chrono::duration<float,std::milli>(t_end - t_start).count();

    // (3) scatter to per-bin contiguous buffer
    Point2D* d_tmp=nullptr;
    cudaMalloc(&d_tmp, N*sizeof(Point2D));
    thrust::device_vector<int> d_binCursor = d_offsets; // copy
    cudaEventRecord(e2);
    scatterToBins<<<b1,t1>>>(d_in, d_tmp, d_codes,
                             thrust::raw_pointer_cast(d_binCursor.data()),
                             N, (1<<kBits) - 1);
    cudaEventRecord(e3); cudaEventSynchronize(e3);
    cudaEventElapsedTime(&m.scatter_ms, e2, e3);

    // (4) per-bin aggregation over slices
    int* d_count=nullptr; float* d_sum=nullptr; float* d_max=nullptr;
    cudaMalloc(&d_count, numBins*sizeof(int));
    cudaMalloc(&d_sum,   numBins*sizeof(float));
    cudaMalloc(&d_max,   numBins*sizeof(float));
    cudaMemset(d_count, 0, numBins*sizeof(int));
    cudaMemset(d_sum,   0, numBins*sizeof(float));
    std::vector<float> h_min(numBins, -INFINITY);
    cudaMemcpy(d_max, h_min.data(), numBins*sizeof(float), cudaMemcpyHostToDevice);

    dim3 t(block), b(numBins);
    size_t shmem = (sizeof(float)*2 + sizeof(int))* (size_t)block;
    cudaEventRecord(e4);
    binsAgg_slices<<<b,t,shmem>>>(d_tmp,
                                  thrust::raw_pointer_cast(d_offsets.data()),
                                  numBins, thr,
                                  d_count, d_sum, d_max);
    cudaEvent_t e5; cudaEventCreate(&e5);
    cudaEventRecord(e5); cudaEventSynchronize(e5);
    cudaEventElapsedTime(&m.agg_ms, e4, e5);

    m.e2e_ms = m.codes_ms + m.offsets_ms + m.scatter_ms + m.agg_ms;

    std::vector<int>    h_count(numBins);
    std::vector<float>  h_sum  (numBins);
    std::vector<float>  h_max  (numBins);
    cudaMemcpy(h_count.data(), d_count, numBins*sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum.data(),   d_sum,   numBins*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max.data(),   d_max,   numBins*sizeof(float), cudaMemcpyDeviceToHost);

    long long cnt=0; double s=0.0; float mx=-INFINITY;
    for(int b2=0;b2<numBins;++b2){ cnt += h_count[b2]; s += (double)h_sum[b2]; mx = fmaxf(mx, h_max[b2]); }
    m.count_sum = cnt;
    m.avg_temp = (cnt>0)? (s / (double)cnt) : 0.0;
    m.max_temp = mx;

    // cleanup
    cudaFree(d_in); cudaFree(d_codes); cudaFree(d_tmp);
    cudaFree(d_count); cudaFree(d_sum); cudaFree(d_max);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);
    return m;
}

// -------------- CSV helpers --------------
static void write_header(std::ofstream& os){
    os<<"impl,dist,kBits,N,hit_rate,"
      <<"codes_ms,offsets_ms,scatter_ms,agg_ms,e2e_ms,"
      <<"count_sum,avg_temp,max_temp\n";
}
static void write_row(std::ofstream& os, const std::string& impl,
                      const Args& a, const Meas& m){
    os<<impl<<","<<a.dist<<","<<a.kBits<<","<<a.N<<","<<std::fixed<<std::setprecision(2)<<a.hit<<","
      <<std::setprecision(4)<<m.codes_ms<<","<<m.offsets_ms<<","<<m.scatter_ms<<","<<m.agg_ms<<","<<m.e2e_ms<<","
      <<m.count_sum<<","<<std::setprecision(6)<<m.avg_temp<<","<<m.max_temp<<"\n";
}

// -------------- main --------------
int main(int argc, char** argv){
    Args a = parse_args(argc, argv);

    // Prepare dataset (host)
    auto h = make_dataset(a.N, a.dist, a.seed, a.kBits);

    // Ensure CSV dir exists
    {
        std::filesystem::path p(a.csv);
        if(p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    }

    // Open CSV
    std::ofstream os(a.csv);
    write_header(os);

    // Warm-up
    cudaFree(0);

    for(int r=0;r<a.repeat;++r){
        Meas m_atomic = run_atomic_round(h, a.kBits, a.hit, a.block);
        Meas m_slices = run_slices_round(h, a.kBits, a.hit, a.block);

        write_row(os, "atomic", a, m_atomic);
        write_row(os, "slices", a, m_slices);

        std::cout<<"[Q7] round "<<(r+1)<<"/"<<a.repeat
                 <<" | atomic e2e="<<m_atomic.e2e_ms<<" ms"
                 <<" | slices e2e="<<m_slices.e2e_ms<<" ms\n";
    }

    std::cout<<"[Q7] CSV written: "<<a.csv<<"\n";
    return 0;
}
