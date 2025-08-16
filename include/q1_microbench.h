#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "common.h"              // Point2D definition
#include "benchmark_utils.h"      // print_csv_header(), choose_threshold_for_rate()
#include "stream_compaction_bin.h" // BreakdownPlanA/B, testPlanA/B_breakdown()

struct Args {
  std::size_t N       = 20'000'000;        // default dataset size
  int         kBits   = 8;                 // default: 256 bins
  std::string dist    = "uniform";         // uniform | clustered | skewed
  std::vector<double> rates{0.05, 0.50, 0.95}; // desired hit rates
};

static std::vector<double> parse_rates(const std::string& s);

static Args parse_cli(int argc, char** argv);

static inline int clampi(int v, int lo, int hi);

static std::vector<Point2D> make_uniform_dataset(std::size_t N, uint32_t seed);

static std::vector<Point2D> make_clustered_dataset(std::size_t N,
                                                   int K ,
                                                   float sigma ,
                                                   uint32_t seed);

static std::vector<Point2D> make_skewed_dataset(std::size_t N,
                                                double heavy_frac,
                                                int window ,
                                                uint32_t seed);

static std::vector<Point2D> make_dataset(std::size_t N, const std::string& dist);

static void run_q1_microbench(const std::vector<Point2D>& input, int kBits,
                              const std::vector<double>& hit_rates);

static void run_q2_microbench(const std::vector<Point2D>& input,
                              int kBits,
                              const std::vector<double>& hit_rates);