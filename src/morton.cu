/**
 * @file morton.cu
 * @brief Morton encoding implementation to convert 2D (x, y) into 1D Z-order code.
 * @author Kaixiang Zou
 * @version 1.0
 * @date 2025-06-23
 */


#include "common.h"

/**
 * @brief Encode 2D coordinates (x, y) into a 1D Morton (Z-order) code.
 *
 * This function interleaves the bits of x and y to generate a single
 * Morton code that preserves spatial locality in memory layout.
 * 
 * The implementation supports up to 16-bit integers (i.e., values < 65536).
 *
 * @param x X coordinate (unsigned int)
 * @param y Y coordinate (unsigned int)
 * @return The computed Morton code (unsigned int)
 */
__host__ __device__
unsigned int morton2D_encode(unsigned int x, unsigned int y) {
    // Spread out the bits of x so that there are 0s between each bit
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    // Do the same for y
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    // Interleave x and y: y goes to the odd bits, x to the even bits
    return (y << 1) | x;
}
