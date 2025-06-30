/**
 * @file common.h
 * @brief Common data structures and function declarations for the Morton Stream Compaction project.
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-06-23
 */


#pragma once
#include <string>
#include <vector>
#include <string>

#define BLOCK_SIZE 256

// Declaration of the Morton encoding function (defined in morton.cu)
__host__ __device__
unsigned int morton2D_encode(unsigned int x, unsigned int y);

/**
 * @brief Structure representing a point in a 2D field.
 *
 * This structure stores spatial coordinates (x, y),
 * wind velocity components (vx, vy), and temperature.
 */
struct Point2D {
    float x, y; ///< x,y-coordinate
    float vx, vy; ///< Velocity in X,y direction
    float temp; ///< Temperature at the point
};

/**
 * @brief Structure combining a Morton code with a 2D point.
 *
 * Used for spatial locality sorting using Morton (Z-order) codes.
 */
struct MortonCodePoint {
    unsigned int morton; ///< Morton code computed from (x, y)
    Point2D point; ///< Associated 2D point data
};

/// Check if a point has temperature higher than the given threshold
__host__ __device__ inline
bool isHotPoint(const Point2D& p, float threshold) {
    return p.temp > threshold;
}

/// Predicate to check if a point is a hot point (used in thrust::copy_if)
struct isHotPredicate {
    float threshold;

    __host__ __device__
    bool operator()(const Point2D& p) const {
        return p.temp > threshold;
    }
};