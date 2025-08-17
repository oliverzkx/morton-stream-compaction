// q1_microbench.cu
//
// Minimal Q1/Q2 micro-benchmark driver with CSV output.
//
// Usage (build & run):
//   # Q1: only PlanA/PlanB (no naive baseline)
//   ./q1_microbench > csv/q1_breakdown.csv
//
//   # Q2: PlanA/PlanB + Naive baseline
//   ./q1_microbench --q2 > csv/q2_breakdown.csv
//
// Common options:
//   ./q1_microbench [--N 20000000] [--k 8] [--dist uniform|clustered|skewed]
//                   [--rates 0.05,0.50,0.95]
//
// CSV Columns:
//   plan,variant,kBits,N,hit_rate,
//   codes_ms,hist_ms,scan_ms,scatter_ms,count_ms,reduce_ms,write_ms,compact_ms,
//   kernel_ms,e2e_ms,total
//
// Notes:
//  - Requires the following symbols provided by your project:
//      * struct Point2D                         (from your types.h)
//      * enum class BinKernel {Shared,Warp,Bitmask}   (from bin_kernel.h)
//      * print_csv_header() and choose_threshold_for_rate() (benchmark_utils.h/.cu)
//      * BreakdownPlanA / BreakdownPlanB + testPlanA/B_breakdown(...) (stream_compaction_bin.h/.cu)
//      * testNaiveGPUCompaction(...) (stream_compaction.h/.cu)

#include "q1_microbench.h"       // brings: Args, print_csv_header(), choose_threshold_for_rate(), types
#include <iomanip>               // std::setprecision
#include <sstream>               // parsing --rates
#include "stream_compaction.h"   // testNaiveGPUCompaction(...)
#include "stream_compaction_bin.h" // BreakdownPlanA/B, testPlanA/B_breakdown(...)

// ----------------------------- global mode ------------------------------
static bool g_q2_mode = false; // default: Q1 mode

// ----------------------------- CLI parsing ------------------------------

static std::vector<double> parse_rates(const std::string& s) {
  std::vector<double> r;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) r.push_back(std::stod(tok));
  }
  if (r.empty()) r = {0.05, 0.50, 0.95};
  return r;
}

static Args parse_cli(int argc, char** argv) {
  Args a; // has defaults from q1_microbench.h
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << k << "\n"; std::exit(2); }
      return argv[++i];
    };
    if (k == "--N")         a.N      = static_cast<std::size_t>(std::stoll(next()));
    else if (k == "--k")    a.kBits  = std::stoi(next());
    else if (k == "--dist") a.dist   = next();
    else if (k == "--rates")a.rates  = parse_rates(next());
    else if (k == "--q2")   g_q2_mode = true;
    else if (k == "-h" || k == "--help") {
      std::cout <<
        "Usage: " << argv[0] <<
        " [--N 20000000] [--k 8] [--dist uniform|clustered|skewed] [--rates 0.05,0.50,0.95] [--q2]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown arg: " << k << "\n";
      std::exit(2);
    }
  }
  return a;
}

// -------------------------- dataset generation -------------------------

// Clamp helper for integer grid
static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Generate a uniform dataset on a 2D grid; temp ~ U[0,1)
std::vector<Point2D> make_uniform_dataset(std::size_t N, uint32_t seed=1234) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> XY(0, 65535);   // fits in 16 bits
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

// Clustered: choose K random centers; samples are Gaussians around centers
std::vector<Point2D> make_clustered_dataset(std::size_t N,
                                                   int K = 32,
                                                   float sigma = 300.f,
                                                   uint32_t seed=5678) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> XYc(0, 65535);
  std::normal_distribution<float> N01(0.f, 1.f);
  std::uniform_real_distribution<float> F01(0.f, 1.f);

  std::vector<std::pair<int,int>> centers;
  centers.reserve(K);
  for (int i = 0; i < K; ++i) centers.emplace_back(XYc(rng), XYc(rng));

  std::uniform_int_distribution<int> pick(0, K-1);

  std::vector<Point2D> v(N);
  for (std::size_t i = 0; i < N; ++i) {
    auto c = centers[pick(rng)];
    float xf = static_cast<float>(c.first)  + sigma * N01(rng);
    float yf = static_cast<float>(c.second) + sigma * N01(rng);
    Point2D p{};
    p.x = static_cast<float>(clampi((int)std::lround(xf), 0, 65535));
    p.y = static_cast<float>(clampi((int)std::lround(yf), 0, 65535));
    p.vx = 0.f; p.vy = 0.f;
    p.temp = F01(rng);
    v[i] = p;
  }
  return v;
}

// Skewed: put a large fraction into a tiny window; rest uniform
std::vector<Point2D> make_skewed_dataset(std::size_t N,
                                                double heavy_frac = 0.9,
                                                int window = 1024,
                                                uint32_t seed=9012) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> XY(0, 65535);
  std::uniform_real_distribution<float> F01(0.f, 1.f);
  std::bernoulli_distribution heavy(heavy_frac);

  // Pick a random small window
  int cx = XY(rng), cy = XY(rng);
  std::uniform_int_distribution<int> Xw(std::max(0, cx-window), std::min(65535, cx+window));
  std::uniform_int_distribution<int> Yw(std::max(0, cy-window), std::min(65535, cy+window));

  std::vector<Point2D> v(N);
  for (std::size_t i = 0; i < N; ++i) {
    bool h = heavy(rng);
    int xi = h ? Xw(rng) : XY(rng);
    int yi = h ? Yw(rng) : XY(rng);
    Point2D p{};
    p.x = static_cast<float>(xi);
    p.y = static_cast<float>(yi);
    p.vx = 0.f; p.vy = 0.f;
    p.temp = F01(rng);
    v[i] = p;
  }
  return v;
}

std::vector<Point2D> make_dataset(std::size_t N, const std::string& dist) {
  if (dist == "uniform")   return make_uniform_dataset(N);
  if (dist == "clustered") return make_clustered_dataset(N);
  if (dist == "skewed")    return make_skewed_dataset(N);
  std::cerr << "Unknown distribution '" << dist << "', using uniform.\n";
  return make_uniform_dataset(N);
}

// ---------------------------- Q1 benchmark -----------------------------

static void run_q1_microbench(const std::vector<Point2D>& input, int kBits,
                              const std::vector<double>& hit_rates) {
  print_csv_header();

  const int N = static_cast<int>(input.size());

  for (double h : hit_rates) {
    const float thr = choose_threshold_for_rate(input, h);

    // ---- Plan B : binned-atomic with precomputed offsets ----
    BreakdownPlanB pb{};
    testPlanB_breakdown(input, thr, kBits, pb, /*host_output*/nullptr);

    // plan,variant,kBits,N,hit_rate,
    std::cout << "PlanB,atomic," << kBits << "," << N << "," << h << ","
              // codes,hist,scan,scatter
              << pb.codes_ms << ","   // codes_ms
              << 0.0         << ","   // hist_ms (N/A)
              << 0.0         << ","   // scan_ms (N/A)
              << 0.0         << ","   // scatter_ms (N/A)
              // count,reduce,write,compact
              << pb.count_ms  << ","
              << pb.reduce_ms << ","
              << pb.write_ms  << ","
              << 0.0          << ","  // compact_ms (N/A)
              // kernel,e2e,total
              << pb.kernel_ms << ","
              << pb.e2e_ms    << ","
              << pb.total_valid
              << "\n";

    // ---- Plan A : per-bin Shared / Warp / Bitmask ----
    for (auto kind : {BinKernel::Shared, BinKernel::Warp, BinKernel::Bitmask}) {
      BreakdownPlanA pa{};
      testPlanA_breakdown(input, thr, kBits, kind, pa, /*host_output*/nullptr);

      const char* v = (kind==BinKernel::Shared ? "shared"
                    : (kind==BinKernel::Warp ? "warp" : "bitmask"));

      std::cout << "PlanA," << v << "," << kBits << "," << N << "," << h << ","
                // codes,hist,scan,scatter
                << pa.codes_ms    << ","
                << pa.hist_ms     << ","
                << pa.scan_ms     << ","
                << pa.scatter_ms  << ","
                // count,reduce,write (N/A for Plan A), compact
                << 0.0            << ","
                << 0.0            << ","
                << 0.0            << ","
                << pa.compact_ms  << ","
                // kernel,e2e,total
                << pa.kernel_ms   << ","
                << pa.e2e_ms      << ","
                << pa.total_out
                << "\n";
    }
  }
}

// Q2: kernel-only vs end-to-end（含 Naive / Baseline-thrust / PlanB-atomic / PlanA 三变体）
// 约定：所有“没有该阶段”的列统一输出 0.0；CSV 表头沿用 Q1/Q2 相同格式。
static void run_q2_microbench(const std::vector<Point2D>& input,
                              int kBits,
                              const std::vector<double>& hit_rates)
{
    using std::cout;
    const int N = static_cast<int>(input.size());

    // 统一 CSV 小数格式
    cout.setf(std::ios::fixed);
    cout << std::setprecision(6);

    // ✅ 在该 N 上预分配并复用 Thrust 缓冲区
    thrust::device_vector<Point2D> d_in(input.size());
    thrust::device_vector<Point2D> d_out(input.size());

    for (double h : hit_rates) {
        const float thr = choose_threshold_for_rate(input, h);

        // 1) Naive（保持不变）
        {
            std::vector<Point2D> out_naive;
            float ms_k = 0.f, ms_e = 0.f;
            testNaiveGPUCompaction(input, thr, out_naive, &ms_k, &ms_e);
            cout << "Naive,atomic," << kBits << "," << N << "," << h << ","
                 << 0.0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << ","
                 << 0.0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << ","
                 << ms_k << "," << ms_e << "," << out_naive.size() << "\n";
        }

        // 2) Baseline — Thrust（改用复用版）
        {
            float ms_k = 0.f, ms_e = 0.f; size_t tot = 0;
            run_thrust_baseline_with_buffers(input, thr, d_in, d_out, ms_k, ms_e, tot);
            cout << "Baseline,thrust," << kBits << "," << N << "," << h << ","
                 << 0.0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << ","
                 << 0.0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << ","
                 << ms_k << "," << ms_e << "," << tot << "\n";
        }

        // 3) PlanB（不变）
        {
            BreakdownPlanB pb{}; std::vector<Point2D> out_b;
            testPlanB_breakdown(input, thr, kBits, pb, &out_b);
            cout << "PlanB,atomic," << kBits << "," << N << "," << h << ","
                 << pb.codes_ms << "," << 0.0 << "," << 0.0 << "," << 0.0 << ","
                 << pb.count_ms << "," << pb.reduce_ms << "," << pb.write_ms << "," << 0.0 << ","
                 << pb.kernel_ms << "," << pb.e2e_ms << "," << out_b.size() << "\n";
        }

        // 4) PlanA（不变，三变体）
        for (auto kind : {BinKernel::Shared, BinKernel::Warp, BinKernel::Bitmask}) {
            BreakdownPlanA pa{}; std::vector<Point2D> out_a;
            testPlanA_breakdown(input, thr, kBits, kind, pa, &out_a);
            const char* vname = (kind==BinKernel::Shared ? "shared" :
                                 (kind==BinKernel::Warp ? "warp" : "bitmask"));
            cout << "PlanA," << vname << "," << kBits << "," << N << "," << h << ","
                 << pa.codes_ms << "," << pa.hist_ms << "," << pa.scan_ms << "," << pa.scatter_ms << ","
                 << 0.0 << "," << 0.0 << "," << 0.0 << "," << pa.compact_ms << ","
                 << pa.kernel_ms << "," << pa.e2e_ms << "," << out_a.size() << "\n";
        }
    }
}



// 更公平的 Thrust 基线：
// - 不在计时窗口里做设备分配/构造（由外部传入可复用的缓冲区）
// - 计时范围：H2D(拷入) + copy_if(kernel) + D2H(仅有效数量)；含 1 次暖机
static void run_thrust_baseline_with_buffers(
    const std::vector<Point2D>& h_input,
    float thr,
    thrust::device_vector<Point2D>& d_in,   // 预分配，大小 == N
    thrust::device_vector<Point2D>& d_out,  // 预分配，大小 == N
    /* OUT */ float& kernel_ms,
    /* OUT */ float& e2e_ms,
    /* OUT */ size_t& total_valid)
{
    using T = Point2D;
    const size_t N = h_input.size();

    // --- 可选暖机：一次不计时的 copy_if，减少首次抖动
    {
        // 拷入
        thrust::copy(h_input.begin(), h_input.end(), d_in.begin());
        // kernel
        auto warm_end = thrust::copy_if(
            d_in.begin(), d_in.end(), d_out.begin(),
            [thr] __device__ (const T& p) { return p.temp > thr; }
        );
        (void)warm_end;
        cudaDeviceSynchronize();
    }

    // 事件
    cudaEvent_t e0, e1, k0, k1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&k0); cudaEventCreate(&k1);

    // --- E2E 计时开始
    cudaEventRecord(e0);

    // H2D：仅拷入
    thrust::copy(h_input.begin(), h_input.end(), d_in.begin());

    // kernel-only：copy_if
    cudaEventRecord(k0);
    auto end_it = thrust::copy_if(
        d_in.begin(), d_in.end(), d_out.begin(),
        [thr] __device__ (const T& p) { return p.temp > thr; }
    );
    cudaEventRecord(k1);
    cudaEventSynchronize(k1);

    total_valid = static_cast<size_t>(end_it - d_out.begin());

    // D2H：只回传有效数量
    if (total_valid > 0) {
        std::vector<T> h_out(total_valid);
        thrust::copy(d_out.begin(), d_out.begin() + total_valid, h_out.begin());
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    cudaEventElapsedTime(&kernel_ms, k0, k1);
    cudaEventElapsedTime(&e2e_ms,    e0, e1);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(k0); cudaEventDestroy(k1);
}


// -------------------------------- main ---------------------------------

int main(int argc, char** argv) {
    const Args args = parse_cli(argc, argv);

    // ✅ 无论 Q1 / Q2，都先打印一次 CSV 表头
    print_csv_header();

    // dataset sizes
    std::vector<std::size_t> sizes;
    if (args.N > 0) {
        sizes = {args.N}; // user specified a single N
    } else {
        sizes = {1000000, 5000000, 10000000, 20000000}; // default set
    }

    for (std::size_t N : sizes) {
        // generate dataset
        std::vector<Point2D> input = make_dataset(N, args.dist);

        if (g_q2_mode) {
            // Run Q2 experiment: kernel-only vs E2E
            run_q2_microbench(input, args.kBits, args.rates);
        } else {
            // Run Q1 experiment: per-stage breakdown
            run_q1_microbench(input, args.kBits, args.rates);
        }
    }

    return 0;
}