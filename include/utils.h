/**
 * @file utils.h
 * @brief Helper routines for the Morton-curve stream-compaction project.
 *
 * This header gathers assorted utilities:
 *   • Bit-level debugging helpers for Morton encoding.  
 *   • Point-set generation, printing, and CPU/GPU sorting.  
 *   • CPU and Thrust-based reference compaction functions.  
 *   • A simple CUDA-device picker that favours multiprocessor count.  
 *
 * None of the functions allocate long-lived resources; they are intended
 * for quick experiments, unit tests, and demo visualisation.
 *
 * Author:  Kaixiang Zou  
 * Version: 1.1  
 * Date:    2025-07-26
 */
#pragma once

#include <iostream>
#include <bitset>
#include <vector>
#include <string>

#include "common.h"  // Point2D definition

// ────────────────────────────────────────────────────────────────
// Bit-level / debug helpers
// ────────────────────────────────────────────────────────────────

/**
 * @brief Visualise the bit-expansion steps used in 2-D Morton encoding.
 *
 * Prints intermediate masks and results to <tt>stdout</tt>.
 */
void binary_test();

/**
 * @brief Print an <tt>unsigned int</tt> as binary, prefixed by @p label.
 *
 * @param val   Value to display.
 * @param label Optional label shown before the binary string.
 */
void print_binary(unsigned int val, const std::string& label = "");

// ────────────────────────────────────────────────────────────────
// Point generation & display
// ────────────────────────────────────────────────────────────────

/**
 * @brief Generate a 2-D Cartesian grid of points with velocity and temperature.
 *
 * @param width     Grid width  (points along x).
 * @param height    Grid height (points along y).
 * @param spacing   Physical spacing between adjacent points.
 * @param seed      RNG seed; <0 ➔ use std::random_device.
 * @return          Vector of generated points (host-side).
 */
std::vector<Point2D> generatePoints(int   width,
                                    int   height,
                                    float spacing = 1.0f,
                                    int   seed    = -1);

/**
 * @brief Print a subset of points with position, velocity, and temperature.
 *
 * @param points    Input vector.
 * @param title     Header string printed before the list.
 * @param maxPrint  Maximum elements to display (useful for large sets).
 */
void printPointList(const std::vector<Point2D>& points,
                    const std::string&          title,
                    int                         maxPrint = 20);

// ────────────────────────────────────────────────────────────────
// CPU / GPU sorting by Morton code
// ────────────────────────────────────────────────────────────────

/**
 * @brief Sort @p points in-place on the CPU using std::sort and Morton keys.
 */
void sort_by_morton(std::vector<Point2D>& points);

/**
 * @brief Sort @p points by Morton code using Thrust.
 *
 * If @p useGPU is <tt>true</tt>, launch a device policy; otherwise fall back
 * to Thrust’s host policy (useful for debugging on non-CUDA machines).
 *
 * @param points  Input/output vector (modified in-place).
 * @param useGPU  <tt>true</tt> ➔ thrust::device; <tt>false</tt> ➔ thrust::host.
 */
void sort_by_morton_thrust(std::vector<Point2D>& points,
                           bool                  useGPU = true);

// ────────────────────────────────────────────────────────────────
// Reference stream-compaction (CPU + Thrust back-end)
// ────────────────────────────────────────────────────────────────

/**
 * @brief CPU reference implementation: remove points below @p threshold.
 *
 * @param points     Input vector.
 * @param threshold  Predicate threshold applied to Point2D::temp.
 * @return           Compacted vector (host-side).
 */
std::vector<Point2D> compact_stream_cpu(const std::vector<Point2D>& points,
                                        float                      threshold);

/**
 * @brief Thrust-based compaction (device or host back-end).
 *
 * @param input     Input vector.
 * @param threshold Predicate threshold.
 * @param useGPU    <tt>true</tt> ➔ device back-end; <tt>false</tt> ➔ host.
 * @return          Compacted vector.
 */
std::vector<Point2D> compact_points_thrust(const std::vector<Point2D>& input,
                                           float                       threshold,
                                           bool                        useGPU = true);

// ────────────────────────────────────────────────────────────────
// CUDA device helper
// ────────────────────────────────────────────────────────────────

/**
 * @brief Choose the “best” CUDA device (highest multiprocessor count).
 *
 * @param verbose If <tt>true</tt>, print the selected device’s properties.
 * @return        Device ID to pass to <tt>cudaSetDevice</tt>.
 */
int chooseCudaCard(bool verbose = true);
