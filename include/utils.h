/**
 * @file utils.h
 * @brief Utility functions for Morton Stream Compaction project, including point generation, display, and sorting helpers.
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-06-23
 */


#pragma once
#include <iostream>
#include <bitset>
#include <vector>
#include <string>


/// Debug function to visualize bit expansion in Morton encoding
void binary_test();

/// Generate a 2D grid of points with wind and temperature
std::vector<Point2D> generatePoints(int width, int height, float spacing = 1.0f, int seed = -1);

/// Print a list of points with position, wind, and temperature
void printPointList(const std::vector<Point2D>& points, const std::string& title = "");

/// Print an unsigned int in binary format with a label
void print_binary(unsigned int val, const std::string& label = "");

/// Sort points by Morton code on CPU using std::sort
void sort_by_morton(std::vector<Point2D>& points);

/// Sort points by Morton code using Thrust (GPU or CPU)
void sort_by_morton_thrust(std::vector<Point2D>& points, bool useGPU = true);

/// Compact the list of points by removing those below the temperature threshold
std::vector<Point2D> compact_stream_cpu(const std::vector<Point2D>& points, float threshold);

/// Perform stream compaction using Thrust (GPU or CPU backend)
std::vector<Point2D> compact_points_thrust(const std::vector<Point2D>& input, float threshold, bool useGPU = true);
