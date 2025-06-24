/**
 * @file utils.cu
 * @brief Utility functions for point generation, binary display, and Morton sorting.
 *        Includes both CPU (std::sort) and GPU (Thrust) Morton sort implementations.
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-06-23
 */

#include "common.h"
#include <iostream>
#include <bitset>
#include <vector>
#include <random>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <algorithm>

/**
 * @brief Print a 32-bit unsigned integer in binary format with an optional label.
 *
 * This function prints the binary representation of an unsigned integer, useful
 * for visualizing bit manipulations such as Morton encoding.
 *
 * @param val   The unsigned integer value to print in binary.
 * @param label Optional label string printed before the binary number.
 */
void print_binary(unsigned int val, const std::string& label) {
    std::bitset<32> bits(val);
    std::cout << label << " = " << bits << " (decimal: " << val << ")" << std::endl;
}


/**
 * @brief Generate a 2D grid of random points with wind and temperature data.
 * 
 * This function creates a set of Point2D objects arranged in a grid of (width × height),
 * with optional spacing and controllable random seed for reproducibility.
 * 
 * @param width   Number of points in the horizontal direction (X).
 * @param height  Number of points in the vertical direction (Y).
 * @param spacing Distance between adjacent grid points. Default is 1.0f.
 * @param seed    Optional random seed. If seed < 0, system time is used for randomness.
 * 
 * @return std::vector<Point2D>  A list of generated 2D points with velocity and temperature values.
 */
std::vector<Point2D> generatePoints(int width, int height, float spacing, int seed)
{
    std::vector<Point2D> points;
    points.reserve(width * height);  // Preallocate space

    // Initialize RNG with optional seed
    std::default_random_engine rng;
    if (seed < 0)
        rng.seed(static_cast<unsigned int>(time(0)));  // Use current time
    else
        rng.seed(static_cast<unsigned int>(seed));     // Use provided seed

    std::uniform_real_distribution<float> wind_dist(-5.0f, 5.0f);    // Horizontal/vertical wind
    std::uniform_real_distribution<float> temp_dist(10.0f, 40.0f);   // Temperature range

    // Loop over the grid and generate points
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Point2D p;
            p.x = x * spacing;               // Set x-coordinate
            p.y = y * spacing;               // Set y-coordinate
            p.vx = wind_dist(rng);           // Random wind in X
            p.vy = wind_dist(rng);           // Random wind in Y
            p.temp = temp_dist(rng);         // Random temperature

            points.push_back(p);
        }
    }

    return points;
}

/**
 * @brief Print a list of 2D points including position, wind, and temperature.
 *
 * This function prints out each point in a given list, including its (x, y) position,
 * wind velocity vector (vx, vy), and temperature value. It optionally takes a title
 * to label the output section.
 *
 * @param points A vector of Point2D containing all the points to print.
 * @param title  Optional title string to label the print section.
 */
void printPointList(const std::vector<Point2D>& points, const std::string& title) {
    // Print a title header if provided
    if (!title.empty()) {
        std::cout << "🌀 " << title << " (" << points.size() << " points):" << std::endl;
    }
    for (const auto& p : points) {
        std::cout << std::fixed << std::setprecision(2)
                  << "(" << p.x << ", " << p.y << ")"    // Position
                  << "  V=(" << p.vx << ", " << p.vy << ")"  // Wind vector
                  << "  Temp=" << p.temp << std::endl;   // Temperature
    }
    std::cout << std::endl;
}

/**
 * @brief Debug helper to visualize the bit expansion used in Morton encoding.
 * 
 * This function demonstrates how bits are interleaved during Morton encoding
 * by printing the intermediate binary results after each step.
 * 
 * Useful for learning and verifying bit-level logic.
 */
void binary_test()
{
    // you can change the number in what you need
    unsigned int x = 0; 
    print_binary(x, "original x");

    x = (x | (x << 8)) & 0x00FF00FF;
    print_binary(x, "after <<8 and & 0x00FF00FF");

    x = (x | (x << 4)) & 0x0F0F0F0F;
    print_binary(x, "after <<4 and & 0x0F0F0F0F");

    x = (x | (x << 2)) & 0x33333333;
    print_binary(x, "after <<2 and & 0x33333333");

    x = (x | (x << 1)) & 0x55555555;
    print_binary(x, "after <<1 and & 0x55555555");
}



/**
 * @brief Sort 2D points by Morton code using Thrust. Backend can be CPU or GPU.
 * 
 * @param points Host vector of points (will be sorted in-place)
 * @param useGPU If true, run on GPU (device_vector); else run on CPU (host_vector)
 */
void sort_by_morton_thrust(std::vector<Point2D>& points, bool useGPU) {
    size_t N = points.size();

    if (useGPU) {
        // --- GPU version ---
        thrust::host_vector<unsigned int> h_morton(N);
        thrust::host_vector<Point2D> h_points = points;

        for (size_t i = 0; i < N; ++i) {
            h_morton[i] = morton2D_encode((unsigned int)(h_points[i].x),
                                          (unsigned int)(h_points[i].y));
        }

        // Copy to GPU
        thrust::device_vector<unsigned int> d_morton = h_morton;
        thrust::device_vector<Point2D> d_points = h_points;

        // GPU sort
        thrust::sort_by_key(thrust::device, d_morton.begin(), d_morton.end(), d_points.begin());

        // Copy back
        thrust::copy(d_points.begin(), d_points.end(), points.begin());

    } else {
        // --- CPU version (host backend) ---
        thrust::host_vector<unsigned int> h_morton(N);
        thrust::host_vector<Point2D> h_points = points;

        for (size_t i = 0; i < N; ++i) {
            h_morton[i] = morton2D_encode((unsigned int)(h_points[i].x),
                                          (unsigned int)(h_points[i].y));
        }

        // CPU sort using Thrust (OpenMP-like backend)
        thrust::sort_by_key(thrust::host, h_morton.begin(), h_morton.end(), h_points.begin());

        // Copy back
        thrust::copy(h_points.begin(), h_points.end(), points.begin());
    }
}


/**
 * @brief Sort a list of 2D points by their Morton code using std::sort on CPU.
 *
 * @param points Reference to the vector of points to sort in-place.
 */
void sort_by_morton(std::vector<Point2D>& points) {
    // Step 1: Encode each point with its Morton code
    std::vector<MortonCodePoint> mcps;
    mcps.reserve(points.size());

    for (const auto& p : points) {
        mcps.push_back({
            morton2D_encode(static_cast<unsigned int>(p.x),
                            static_cast<unsigned int>(p.y)),
            p
        });
    }

    // Step 2: Sort by Morton code
    std::sort(mcps.begin(), mcps.end(),
              [](const MortonCodePoint& a, const MortonCodePoint& b) {
                  return a.morton < b.morton;
              });

    // Step 3: Copy sorted results back
    for (size_t i = 0; i < points.size(); ++i) {
        points[i] = mcps[i].point;
    }
}

/**
 * Compact the input point list on CPU by filtering out points 
 * whose temperature is below the given threshold.
 *
 * This function implements stream compaction on CPU by applying
 * the `isHotPoint` predicate to each point. Points that pass the 
 * predicate (i.e., temperature > threshold) are retained.
 *
 * @param points The input list of 2D points with wind and temperature data.
 * @param threshold The temperature threshold to consider a point "hot".
 * @return std::vector<Point2D> A compacted list containing only hot points.
 */
std::vector<Point2D> compact_stream_cpu(const std::vector<Point2D>& points, float threshold) {
    std::vector<Point2D> compacted;
    compacted.reserve(points.size()); // Preallocate full size (safe)

    for (const auto& p : points) {
        if (isHotPoint(p, threshold)) {
            compacted.push_back(p); // Keep only hot points
        }
    }

    return compacted;
}

/**
 * Perform stream compaction on a list of 2D points using Thrust,
 * filtering out points whose temperature is below the given threshold.
 *
 * This function uses thrust::copy_if with a predicate (isHotPredicate)
 * to retain only the "hot" points. It supports both GPU (device_vector)
 * and CPU (host_vector) backends by setting the useGPU flag.
 *
 * @param input The input std::vector of Point2D elements to be filtered.
 * @param threshold The temperature threshold; only points with temp > threshold are kept.
 * @param useGPU If true, the GPU backend (device_vector) is used; otherwise CPU backend (host_vector).
 * @return std::vector<Point2D> A compacted list of hot points.
 */
std::vector<Point2D> compact_points_thrust(const std::vector<Point2D>& input, float threshold, bool useGPU) {
    // Step 1: copy input data to Thrust vector
    if (useGPU) {
        thrust::device_vector<Point2D> d_input = input;
        thrust::device_vector<Point2D> d_output(input.size());

        // Step 2: perform thrust::copy_if with isHotPredicate
        auto end_it = thrust::copy_if(
            d_input.begin(), d_input.end(),
            d_output.begin(),
            isHotPredicate{threshold}
        );

        // Step 3: compute size of result
        size_t numValid = end_it - d_output.begin();

        // Step 4: copy back to host
        std::vector<Point2D> result(numValid);
        thrust::copy(d_output.begin(), d_output.begin() + numValid, result.begin());
        return result;
    } else {
        // Same logic, but on CPU using host_vector
        thrust::host_vector<Point2D> h_input = input;
        thrust::host_vector<Point2D> h_output(input.size());

        auto end_it = thrust::copy_if(
            h_input.begin(), h_input.end(),
            h_output.begin(),
            isHotPredicate{threshold}
        );

        size_t numValid = end_it - h_output.begin();

        std::vector<Point2D> result(numValid);
        thrust::copy(h_output.begin(), h_output.begin() + numValid, result.begin());
        return result;
    }
}