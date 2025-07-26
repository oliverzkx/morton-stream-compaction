/**
 * @file morton.cu
 * @brief Encode 2-D integer coordinates into a 32-bit Morton (Z-order) code.
 *
 * The routine interleaves the low-order bits of <tt>x</tt> and <tt>y</tt>
 * so that spatially adjacent points map to nearby 1-D positions—useful for
 * cache-friendly layouts and binning algorithms.
 *
 * The algorithm here supports 16-bit inputs (values < 65536); higher bits
 * are ignored by the masking operations.
 *
 * Author  : Kaixiang Zou  
 * Version : 1.1  
 * Date    : 2025-07-26
 */

#include "common.h"

/**
 * @brief Convert 2-D coordinates to a 32-bit Morton code.
 *
 * @param x X coordinate (unsigned int, ≤ 65535).
 * @param y Y coordinate (unsigned int, ≤ 65535).
 * @return  Interleaved Morton code.
 */
__host__ __device__
unsigned int morton2D_encode(unsigned int x, unsigned int y)
{
    /* ---- bit-interleave x ---- */
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    /* ---- bit-interleave y ---- */
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    /* ---- merge: y bits occupy odd positions, x bits even ---- */
    return (y << 1) | x;
}
