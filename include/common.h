/**
 * @file common.h
 * @brief Common data types, constants, and helper predicates used
 *        throughout the Morton-curve stream-compaction project.
 *
 * The header defines:
 *   • Fundamental compile-time constants (e.g. BLOCK_SIZE).  
 *   • Core point structures in both single- and double-precision.  
 *   • A Morton-code/point pairing used for locality sorting.  
 *   • Lightweight host/device helpers and predicates that identify
 *     “hot” points whose temperature exceeds a user-supplied threshold.
 *
 * All entities are simple PODs or inline functions so the header can be
 * included by both host-only and CUDA device code without side effects.
 *
 * @author Kaixiang Zou
 * @version 1.2
 * @date 2025-07-26
 */
#pragma once

#include <vector>
#include <string>

/// Default CUDA thread-block size used across kernels
#define BLOCK_SIZE 256

// ────────────────────────────────────────────────────────────────
// Morton encoding
// ────────────────────────────────────────────────────────────────

/**
 * @brief Encode a 2-D integer coordinate (x, y) into a 32-bit Morton code.
 *
 * The implementation (in morton.cu) interleaves bits of @p x and @p y
 * to produce a Z-order index that preserves spatial locality.
 *
 * @param x The x-coordinate (least-significant 16 bits are used).
 * @param y The y-coordinate (least-significant 16 bits are used).
 * @return 32-bit Morton code.
 */
__host__ __device__
unsigned int morton2D_encode(unsigned int x, unsigned int y);

// ────────────────────────────────────────────────────────────────
// Point structures
// ────────────────────────────────────────────────────────────────

/**
 * @brief Single-precision point sample in a 2-D flow field.
 *
 * Contains spatial coordinates, velocity components, and temperature.
 */
struct Point2D {
    float  x,  y;   ///< Cartesian position
    float  vx, vy;  ///< Velocity components
    float  temp;    ///< Temperature value
};

/**
 * @brief Double-precision variant of Point2D.
 */
struct Point2D_double {
    double x,  y;   ///< Cartesian position
    double vx, vy;  ///< Velocity components
    double temp;    ///< Temperature value
};

/**
 * @brief Pairing of a pre-computed Morton code with its corresponding point.
 *
 * Used by sorting stages to arrange points along a Z-order curve so that
 * spatially adjacent points end up close in memory.
 */
struct MortonCodePoint {
    unsigned int morton;  ///< Morton code generated from (x, y)
    Point2D      point;   ///< Associated point sample
};

// ────────────────────────────────────────────────────────────────
// “Hot-point” helpers
// ────────────────────────────────────────────────────────────────

/**
 * @brief Check whether a point’s temperature exceeds @p threshold.
 *
 * @param p         Point sample to test.
 * @param threshold Temperature threshold.
 * @return          True if @p p.temp > @p threshold.
 */
__host__ __device__ inline
bool isHotPoint(const Point2D& p, float threshold) {
    return p.temp > threshold;
}

/**
 * @brief Functor version of isHotPoint (single-precision).
 *
 * Suitable for algorithms such as `thrust::copy_if`.
 */
struct isHotPredicate {
    float threshold;

    __host__ __device__
    bool operator()(const Point2D& p) const {
        return p.temp > threshold;
    }
};

/**
 * @brief Double-precision overload of isHotPoint.
 */
__host__ __device__ inline
bool isHotPoint(const Point2D_double& p, double threshold) {
    return p.temp > threshold;
}

/**
 * @brief Functor version of isHotPoint (double-precision).
 */
struct isHotPredicate_double {
    double threshold;

    __host__ __device__
    bool operator()(const Point2D_double& p) const {
        return p.temp > threshold;
    }
};
