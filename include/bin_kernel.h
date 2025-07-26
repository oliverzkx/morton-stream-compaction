/**
 * @file bin_kernel.h
 * @brief Enumeration of single-bin stream-compaction kernel variants.
 *
 * This header defines an enumeration that lists every CUDA kernel
 * implementation available for compacting points inside one bin.
 * Each variant follows a different optimization strategy:
 *   • Shared   – uses on-chip shared memory to minimise global accesses.  
 *   • Warp     – relies on warp-level primitives (e.g., ballot / prefix sum).  
 *   • Bitmask  – compacts with per-thread validity bits and prefix sums.  
 *   • Auto     – selects the best variant at runtime according to heuristics.
 *
 * The enum is consumed by higher-level code (e.g., stream_compaction_bin.cu)
 * to dispatch the appropriate kernel based on user configuration or autotune
 * results.
 *
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-07-26
 */
#pragma once

/** @enum BinKernel
 *  @brief Available kernel strategies for single-bin compaction.
 */
enum class BinKernel { Shared,  ///< Shared-memory implementation
                       Warp,    ///< Warp-level (intrinsics-based) implementation
                       Bitmask, ///< Bitmask + prefix-sum implementation
                       Auto     ///< Runtime-selected best implementation
};
