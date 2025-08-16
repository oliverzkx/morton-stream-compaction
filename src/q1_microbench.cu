// q1_microbench.cu
//
// Minimal Q1 micro-benchmark driver with CSV output.
// Usage:
//   ./q1_microbench [--N 20000000] [--k 8] [--dist uniform|clustered|skewed]
//                   [--rates 0.05,0.50,0.95]
//
// Columns printed (CSV):
//   plan,variant,kBits,N,hit_rate,
//   codes_ms,hist_ms,scan_ms,scatter_ms,count_ms,reduce_ms,write_ms,compact_ms,
//   kernel_ms,e2e_ms,total
//
// Notes:
//  - Requires the following symbols provided by your project:
//      * struct Point2D        (from types.h)
//      * enum class BinKernel  (from bin_kernel.h)
//      * print_csv_header_q1() and choose_threshold_for_rate() (benchmark_utils.h/.cu)
//      * BreakdownPlanA / BreakdownPlanB + testPlanA/B_breakdown(...) (stream_compaction_bin.h/.cu)
//  - This file intentionally keeps dataset generation simple. Replace or hook
//    into your existing dataset loader if you already have one.

#include "q1_microbench.h"


// ----------------------------- CLI parsing -----------------------------


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
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << k << "\n"; std::exit(2); }
      return argv[++i];
    };
    if (k == "--N")        a.N     = static_cast<std::size_t>(std::stoll(next()));
    else if (k == "--k")   a.kBits = std::stoi(next());
    else if (k == "--dist")a.dist  = next();
    else if (k == "--rates")a.rates = parse_rates(next());
    else if (k == "-h" || k == "--help") {
      std::cout <<
        "Usage: " << argv[0]
        << " [--N 20000000] [--k 8] [--dist uniform|clustered|skewed]"
           " [--rates 0.05,0.50,0.95]\n";
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
static std::vector<Point2D> make_uniform_dataset(std::size_t N, uint32_t seed=1234) {
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
static std::vector<Point2D> make_clustered_dataset(std::size_t N,
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
static std::vector<Point2D> make_skewed_dataset(std::size_t N,
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

static std::vector<Point2D> make_dataset(std::size_t N, const std::string& dist) {
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
              << ""          << ","   // hist_ms (N/A)
              << ""          << ","   // scan_ms (N/A or uncollected)
              << ""          << ","   // scatter_ms (N/A)
              // count,reduce,write,compact
              << pb.count_ms  << ","
              << pb.reduce_ms << ","
              << pb.write_ms  << ","
              << ""           << ","  // compact_ms (N/A)
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
                << ""             << ","
                << ""             << ","
                << ""             << ","
                << pa.compact_ms  << ","
                // kernel,e2e,total
                << pa.kernel_ms   << ","
                << pa.e2e_ms      << ","
                << pa.total_out
                << "\n";
    }
  }
}

// -------------------------------- main ---------------------------------

int main(int argc, char** argv) {
  const Args args = parse_cli(argc, argv);

  // 1) Build synthetic dataset
  std::vector<Point2D> input = make_dataset(args.N, args.dist);

  // 2) Run Q1 (micro-bench)
  run_q1_microbench(input, args.kBits, args.rates);

  return 0;
}
