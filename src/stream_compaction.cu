// stream_compaction.cu
#include <cuda_runtime.h>
#include "stream_compaction.h"
#include "common.h"          // <- contains typedef Point, predicate, etc.
#include <iostream>

__device__ __constant__ float d_threshold;
__constant__ double d_threshold_double;


// -------------------- device predicate --------------------
/**
 * @brief Device-side predicate function.
 *        Determines whether a given Point meets the compaction condition.
 * @param p Input point
 * @return true if the point should be retained, false otherwise
 */
/*
__device__ inline bool isHotPredicateDevice(const Point2D& p)
{
    isHotPredicate pred{30.0f}; 
    return pred(p);
}
*/

__device__ inline bool isHotPredicateDevice(const Point2D& p)
{
    return isHotPoint(p, d_threshold);  // 使用 constant memory 中的 threshold
}

__device__ inline bool isHotPredicateDevice(const Point2D_double& p) {
    return isHotPoint(p, d_threshold_double);  // d_threshold_double is __constant__ double
}

// -------------------- naïve kernel ------------------------
/**
 * @brief Naïve stream compaction kernel using a global atomic counter.
 *
 * Each thread checks if its input element satisfies a predicate.
 * If yes, it atomically increments a global counter and writes
 * the value to the corresponding output position.
 *
 * @param in         Input array of Point
 * @param out        Output array to store compacted results
 * @param N          Number of input elements
 * @param d_counter  Global counter for tracking valid output position
 */
__global__ void streamCompactNaive(const Point2D* in,
                                   Point2D*       out,
                                   int          N,
                                   int*         d_counter)   // global write index
{
    // Compute thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    Point2D val = in[tid];
    if (isHotPredicateDevice(val))
    {
        // Atomically get output position
        int pos = atomicAdd(d_counter, 1);   // global counter ++
        out[pos] = val;                      // write to compacted array
    }
}

// -------------------- host wrapper ------------------------
/**
 * @brief Host wrapper for the naïve GPU stream compaction.
 *
 * This function allocates and manages the counter on the device,
 * launches the kernel, and copies the final result count back to host.
 *
 * @param d_in        Device input array
 * @param d_out       Device output array
 * @param N           Number of input elements
 * @param h_outCount  [out] Number of elements written to d_out
 */
void compactNaiveGPU(const Point2D* d_in, Point2D* d_out, int N, int& h_outCount)
{
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // int threads = 256;
    // int blocks  = (N + threads - 1) / threads;
    // streamCompactNaive<<<blocks, threads>>>(d_in, d_out, N, d_counter);

    // 🔧 Standard dim3 configuration
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE  - 1) / BLOCK_SIZE );

    streamCompactNaive<<<dimGrid, dimBlock>>>(d_in, d_out, N, d_counter);

    cudaMemcpy(&h_outCount, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_counter);
}

// ==============================================
// Shared Memory Stream Compaction Kernel (Block Scan)
// ==============================================

/**
 * @brief Shared memory-based CUDA kernel for stream compaction using block-level scan.
 *
 * Each block performs a scan on its own data using shared memory to determine
 * which elements satisfy a predicate and writes them to output.
 * This is a first-pass kernel that produces block-local compaction.
 *
 * @param in            Input array of Point2D
 * @param out           Output array (compact result, written by each block)
 * @param N             Number of input elements
 * @param threshold     Threshold for the hot predicate (e.g., temperature)
 * @param block_counts  Output array storing how many items each block compacts
 */
__global__ void streamCompactShared(const Point2D* in,
                                    Point2D*       out,
                                    int            N,
                                    float          threshold,
                                    int*           block_counts)
{
    extern __shared__ unsigned int s_flags[];  // Shared memory: flags for predicate
    Point2D* s_data = (Point2D*)&s_flags[blockDim.x];  // Followed by shared data

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: load input and evaluate predicate
    unsigned int flag = 0;
    Point2D val;
    if (gid < N) {
        val = in[gid];
        flag = isHotPoint(val, threshold) ? 1 : 0;
    }
    s_flags[tid] = flag;
    s_data[tid] = val;

    __syncthreads();

    // Step 2: exclusive scan on flags array in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        unsigned int temp = 0;
        if (tid >= stride) {
            temp = s_flags[tid - stride];
        }
        __syncthreads();
        s_flags[tid] += temp;
        __syncthreads();
    }

    // Step 3: write valid results to global output array
    if (gid < N && flag == 1) {
        int pos = s_flags[tid] - 1;
        out[blockIdx.x * blockDim.x + pos] = val;
    }

    // Step 4: last thread writes total count of valid elements in this block
    if (tid == blockDim.x - 1) {
        block_counts[blockIdx.x] = s_flags[tid];
    }
}

/**
 * @brief Host-side wrapper for shared-memory stream compaction kernel.
 *
 * This function allocates device memory for block-local counters,
 * launches the shared-memory compaction kernel, and collects the total
 * number of compacted elements by summing each block's output count.
 *
 * @param d_in         Device input array
 * @param d_out        Device output array
 * @param N            Number of input elements
 * @param threshold    Predicate threshold (e.g., temperature > threshold)
 * @param h_outCount   [out] Total number of valid elements written to d_out
 */
void compactSharedGPU(const Point2D* d_in, Point2D* d_out, int N,
                      float threshold, int& h_outCount)
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Allocate device memory for per-block counts
    int* d_block_counts;
    cudaMalloc(&d_block_counts, dimGrid.x * sizeof(int));  // use dimGrid.x blocks

    // Calculate shared memory size: flags + Point2D array
    size_t shared_mem_bytes = dimBlock.x * (sizeof(unsigned int) + sizeof(Point2D));

    // Launch shared memory kernel
    streamCompactShared<<<dimGrid, dimBlock, shared_mem_bytes>>>(
        d_in, d_out, N, threshold, d_block_counts);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launch error: %s\n", cudaGetErrorString(err));
    }

    // Copy per-block results back to host
    std::vector<int> h_block_counts(dimGrid.x);
    cudaMemcpy(h_block_counts.data(), d_block_counts,
               dimGrid.x * sizeof(int), cudaMemcpyDeviceToHost);

    // Sum block counts to get total compacted output count
    h_outCount = 0;
    for (int i = 0; i < dimGrid.x; ++i) {
        h_outCount += h_block_counts[i];
    }

    // Free temporary device memory
    cudaFree(d_block_counts);
}


/**
 * @brief Test the naive GPU stream compaction method with timing.
 *
 * This function allocates memory, launches the naive compaction kernel,
 * copies results back to host, and reports execution time.
 *
 * @param input        Input vector of Point2D elements.
 * @param threshold    Temperature threshold for filtering.
 * @param output       Vector to store the compacted results.
 */
/*
void testNaiveGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    int N = input.size();
    Point2D* d_input = nullptr;
    Point2D* d_output = nullptr;
    int compactedCount = 0;

    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, N * sizeof(Point2D));

    // 💡 Copy threshold value to constant memory before kernel launch
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // ⏱️ Start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 🚀 Launch naive compaction kernel
    compactNaiveGPU(d_input, d_output, N, compactedCount);

    // ⏱️ Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "⏱️ Naive GPU compaction time: " << milliseconds << " ms\n";

    output.resize(compactedCount);
    cudaMemcpy(output.data(), d_output, compactedCount * sizeof(Point2D), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
*/

/**
 * @brief Test the naive GPU stream compaction method with full timing.
 *
 * This function allocates device memory, transfers input data,
 * launches the naive compaction kernel, retrieves results, and
 * reports the total GPU execution time including memory operations.
 *
 * @param input        Input vector of Point2D elements.
 * @param threshold    Temperature threshold for filtering.
 * @param output       Vector to store the compacted results.
 */
void testNaiveGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    std::cout << "\n[GPU] Testing Naive Stream Compaction..." << std::endl;

    int N = input.size();
    Point2D* d_input = nullptr;
    Point2D* d_output = nullptr;
    int compactedCount = 0;

    // Create CUDA event timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ✅ Start full GPU timing here (includes malloc, memcpy, kernel, copy back, free)
    cudaEventRecord(start);

    // Allocate and copy input
    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, N * sizeof(Point2D));

    // Copy threshold to constant memory
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // Call your original kernel (compactedCount passed by reference)
    compactNaiveGPU(d_input, d_output, N, compactedCount);

    // Resize and copy result back to host
    output.resize(compactedCount);
    cudaMemcpy(output.data(), d_output, compactedCount * sizeof(Point2D), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);

    // ✅ Stop full GPU timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "⏱️ Naive GPU total time (including memory ops): " << milliseconds << " ms\n";
    std::cout << "📌 Compacted count: " << compactedCount << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



/**
 * @brief Test the shared memory + block scan GPU stream compaction method        with timing.
 *
 * This function allocates memory, launches the optimized shared memory compaction kernel,
 * copies results back to host, and reports execution time.
 *
 * @param input        Input vector of Point2D elements.
 * @param threshold    Temperature threshold for filtering.
 * @param output       Vector to store the compacted results.
 */
/*
void testSharedGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    int N = input.size();
    Point2D* d_input = nullptr;
    Point2D* d_output = nullptr;
    int compactedCount = 0;

    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, N * sizeof(Point2D));


    // 💡 Copy threshold value to constant memory before kernel launch
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // ⏱️ Start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 🚀 Launch shared memory compaction kernel
    compactSharedGPU(d_input, d_output, N, threshold, compactedCount);

    // ⏱️ Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "⏱️ Shared Memory GPU compaction time: " << milliseconds << " ms\n";

    output.resize(compactedCount);
    cudaMemcpy(output.data(), d_output, compactedCount * sizeof(Point2D), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
*/

/**
 * @brief Test the shared memory + block scan GPU stream compaction method with timing.
 *
 * This function allocates memory, launches the optimized shared memory compaction kernel,
 * copies results back to host, and reports full GPU execution time including memory operations.
 *
 * @param input        Input vector of Point2D elements.
 * @param threshold    Temperature threshold for filtering.
 * @param output       Vector to store the compacted results.
 */
void testSharedGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    std::cout << "\n[GPU] Testing Shared Memory Stream Compaction..." << std::endl;

    int N = input.size();
    Point2D* d_input = nullptr;
    Point2D* d_output = nullptr;
    int compactedCount = 0;

    // ⏱️ Start full GPU timing (includes malloc, memcpy, kernel, and result copy)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate and copy input
    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, N * sizeof(Point2D));

    // Copy threshold to constant memory
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // Launch your original kernel (with threshold as argument)
    compactSharedGPU(d_input, d_output, N, threshold, compactedCount);

    // Copy result back to host
    output.resize(compactedCount);
    cudaMemcpy(output.data(), d_output, compactedCount * sizeof(Point2D), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);

    // ⏱️ Stop timing and print result
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "⏱️ Shared Memory GPU total time (including memory ops): " << milliseconds << " ms\n";
    std::cout << "📌 Compacted count: " << compactedCount << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// ==============================================
// Warp Shuffle Stream Compaction Kernel (Intra-Warp Prefix Scan + Ballot)
// ==============================================

/**
 * @brief Stream-compaction kernel that keeps “hot” points using a hybrid
 *        shuffle prefix-scan and warp ballot.
 *
 * Workflow per thread:
 * 1.  Evaluate @p isHotPredicateDevice to decide whether the element is kept.
 * 2.  Perform an intra-warp exclusive prefix sum of the keep votes via
 *     @c __shfl_up_sync.
 * 3.  Obtain the warp-wide keep count with @c __ballot_sync + @c __popc
 *     (works for partial warps).
 * 4.  Lane 0 of every warp writes its count to @p warp_sum[] in shared memory.
 * 5.  Each thread computes its final write position:
 *        write_idx = global_offset + block_offset + prefix - 1
 * 6.  Threads with @p keep == true write their @p Point2D to the compacted
 *     output array.
 *
 * @param d_input     Device pointer to the input @p Point2D array
 * @param d_output    Device pointer to the compacted output array
 * @param d_count     Device pointer to a single int that receives the final
 *                    number of compacted elements
 * @param num_points  Total number of input elements
 */
__global__ void compactPointsWarpShuffle(
    Point2D*  d_input,
    Point2D*  d_output,
    int*      d_count,
    int       num_points)
{
    /* ------------------------------------------------------------------ *
     * 0. Initialise per-warp counters in shared memory                    *
     * ------------------------------------------------------------------ */
    __shared__ int warp_sum[BLOCK_SIZE / 32];          // max 32 warps / block
    if (threadIdx.x < BLOCK_SIZE / 32) warp_sum[threadIdx.x] = 0;
    __syncthreads();

    /* ------------------------------------------------------------------ *
     * 1. Compute predicate                                                *
     * ------------------------------------------------------------------ */
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_points) return;

    Point2D pt   = d_input[gid];
    bool    keep = isHotPredicateDevice(pt);
    int     vote = keep ? 1 : 0;

    /* ------------------------------------------------------------------ *
     * 2. Warp-level prefix scan using shuffles                            *
     * ------------------------------------------------------------------ */
    int lane = threadIdx.x & 31;        // lane ID inside warp
    int warp = threadIdx.x >> 5;        // warp ID inside block

    int prefix = vote;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
        int n = __shfl_up_sync(0xffffffff, prefix, off);
        if (lane >= off) prefix += n;
    }                                   // prefix is 1-based for kept threads

    /* ------------------------------------------------------------------ *
     * 3. Warp total via ballot + popc (handles partial warps)            *
     * ------------------------------------------------------------------ */
    unsigned int mask       = __ballot_sync(0xffffffff, keep);
    int          warp_total = __popc(mask);            // kept count in warp

    if (lane == 0) warp_sum[warp] = warp_total;
    __syncthreads();

    /* ------------------------------------------------------------------ *
     * 4. Block-exclusive prefix of warp totals                           *
     * ------------------------------------------------------------------ */
    int block_offset = 0;
    for (int i = 0; i < warp; ++i)
        block_offset += warp_sum[i];

    int local_pos = block_offset + prefix - 1;         // 0-based index inside block

    /* ------------------------------------------------------------------ *
     * 5. Reserve global output space once per block                      *
     * ------------------------------------------------------------------ */
    __shared__ int global_offset;
    if (threadIdx.x == 0)
    {
        int block_total = 0;
        int warps_in_block = (blockDim.x + 31) >> 5;
        for (int i = 0; i < warps_in_block; ++i)
            block_total += warp_sum[i];

        global_offset = atomicAdd(d_count, block_total);
    }
    __syncthreads();                                   // global_offset ready

    /* ------------------------------------------------------------------ *
     * 6. Write kept element                                              *
     * ------------------------------------------------------------------ */
    if (keep)
        d_output[global_offset + local_pos] = pt;
}

/**
 * @brief Host wrapper for warp-shuffle-based stream compaction.
 *
 * Launches the compactPointsWarpShuffle CUDA kernel to compact the input
 * array of Point2D based on the isHotPredicateDevice condition.
 *
 * @param d_input Device pointer to input Point2D array
 * @param d_output Device pointer to output Point2D array
 * @param d_count Device pointer to int that will hold the total number of compacted points
 * @param num_points Number of input points
 */
void compact_points_warp(
    Point2D* d_input,
    Point2D* d_output,
    int* d_count,
    int num_points
) {
    const dim3 blockDim(BLOCK_SIZE);
    const dim3 gridDim((num_points + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Initialize compacted count to zero
    cudaMemset(d_count, 0, sizeof(int));

    compactPointsWarpShuffle<<<gridDim, blockDim>>>(
        d_input, d_output, d_count, num_points
    );
    cudaDeviceSynchronize();
}



/**
 * @brief Test warp-shuffle-based GPU compaction using given input and threshold.
 *
 * @param input Vector of input points
 * @param threshold Threshold to determine "hot" points
 * @param output Vector to store compacted results
 */
/*
void testWarpGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    std::cout << "\n[GPU] Testing Warp Shuffle Stream Compaction..." << std::endl;

    int N = input.size();

    // Allocate device memory
    Point2D* d_input;
    Point2D* d_output;
    int* d_count;
    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMalloc(&d_output, N * sizeof(Point2D));
    cudaMalloc(&d_count, sizeof(int));

    // Copy input to device
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);

    // ✅ Copy threshold to constant memory
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch wrapper
    compact_points_warp(d_input, d_output, d_count, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "    [Timing] Elapsed time: " << milliseconds << " ms\n";

    // Copy results back
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    output.resize(h_count);
    cudaMemcpy(output.data(), d_output, h_count * sizeof(Point2D), cudaMemcpyDeviceToHost);

    std::cout << "    [Result] Compacted count = " << h_count << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
*/

/**
 * @brief Test warp-shuffle-based GPU compaction using given input and threshold.
 *
 * @param input Vector of input points
 * @param threshold Threshold to determine "hot" points
 * @param output Vector to store compacted results
 */
void testWarpGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    std::cout << "\n[GPU] Testing Warp Shuffle Stream Compaction..." << std::endl;

    int N = input.size();

    // CUDA event timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);  // ✅ Start timing before any GPU activity

    // Allocate device memory
    Point2D* d_input;
    Point2D* d_output;
    int* d_count;
    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMalloc(&d_output, N * sizeof(Point2D));
    cudaMalloc(&d_count, sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);

    // Copy threshold to constant memory
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // Launch warp-based compaction kernel
    compact_points_warp(d_input, d_output, d_count, N);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "⏱️ Warp Shuffle GPU total time (including memory ops): " << milliseconds << " ms\n";

    // Copy results back to host
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    output.resize(h_count);
    cudaMemcpy(output.data(), d_output, h_count * sizeof(Point2D), cudaMemcpyDeviceToHost);

    std::cout << "📌 Compacted count: " << h_count << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



// ==============================================
// Bitmask Stream Compaction Kernel (Ballot + popc)
// ==============================================

/**
 * @brief CUDA kernel for stream compaction using warp-level bitmask voting.
 *
 * Each thread determines whether its assigned point meets the compaction condition.
 * A warp-level bitmask is generated using __ballot_sync, and each thread calculates
 * its compacted output index using __popc. This method minimizes shared memory use
 * and offers high performance for warp-sized compaction.
 *
 * @param d_input Pointer to device-side input array of Point2D
 * @param d_output Pointer to device-side output array for compacted points
 * @param d_count Pointer to device-side int used to store final compacted count
 * @param num_points Total number of input points
 */
__global__ void compactPointsBitmask(
    const Point2D* d_input,
    Point2D* d_output,
    int* d_count,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point2D pt = d_input[idx];
    bool isHot = isHotPredicateDevice(pt);  // user-defined predicate

    unsigned int lane_id = threadIdx.x % 32;
    unsigned int warp_id = threadIdx.x / 32;

    // Warp-wide vote: bitmask of active threads
    unsigned int mask = __ballot_sync(0xffffffff, isHot);

    // Count how many threads before me voted true
    int local_pos = __popc(mask & ((1u << lane_id) - 1));

    // Warp leader reserves space for this warp in global output
    __shared__ int warp_offsets[BLOCK_SIZE / 32];
    int total_in_warp = __popc(mask);

    if (lane_id == 0) {
        warp_offsets[warp_id] = atomicAdd(d_count, total_in_warp);
    }

    __syncthreads();

    // Write point to compacted array if hot
    if (isHot) {
        int output_index = warp_offsets[warp_id] + local_pos;
        d_output[output_index] = pt;
    }
}

/**
 * @brief CUDA kernel for stream compaction using warp-level bitmask voting, writing to surface memory.
 *
 * Each thread checks if its input point meets a "hot" condition (temp > d_threshold).
 * Warp-level bitmask voting is used to compute output offsets.
 * Each hot point is written as float4(vx, vy, temp, 1.0) to a 2D surface.
 *
 * @param d_input         Device pointer to input Point2D array
 * @param surfaceOutput   CUDA surface object to write output
 * @param d_count         Device pointer to int tracking compacted count
 * @param num_points      Number of input points
 * @param surface_width   Width of the surface in elements (used to compute x/y)
 */
__global__ void compactPointsBitmaskSurface(
    const Point2D* d_input,
    cudaSurfaceObject_t surfaceOutput,
    int* d_count,
    int num_points,
    int surface_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point2D pt = d_input[idx];
    bool isHot = pt.temp > d_threshold;

    unsigned int lane_id = threadIdx.x % 32;
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int mask = __ballot_sync(0xffffffff, isHot);
    int local_pos = __popc(mask & ((1U << lane_id) - 1));
    int total_in_warp = __popc(mask);

    __shared__ int warp_offsets[BLOCK_SIZE / 32];
    if (lane_id == 0) {
        warp_offsets[warp_id] = atomicAdd(d_count, total_in_warp);
    }
    __syncthreads();

    if (isHot) {
        int global_output_idx = warp_offsets[warp_id] + local_pos;
        int x = global_output_idx % surface_width;
        int y = global_output_idx / surface_width;

        float4 outVal = make_float4(pt.vx, pt.vy, pt.temp, 1.0f);
        surf2Dwrite(outVal, surfaceOutput, x * sizeof(float4), y);
    }
}


/**
 * @brief Host wrapper to launch bitmask-based stream compaction kernel.
 *
 * Configures and launches compactPointsBitmask kernel to perform compaction
 * using warp-level ballot and popc instructions.
 *
 * @param d_input Device pointer to input array of Point2D
 * @param d_output Device pointer to output array for compacted results
 * @param d_count Device pointer to an int for storing compacted count
 * @param num_points Number of input points
 */
void compact_points_bitmask(
    const Point2D* d_input,
    Point2D* d_output,
    int* d_count,
    int num_points
) {
    const dim3 blockDim(BLOCK_SIZE);
    const dim3 gridDim((num_points + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMemset(d_count, 0, sizeof(int));

    compactPointsBitmask<<<gridDim, blockDim>>>(
        d_input, d_output, d_count, num_points
    );
    cudaDeviceSynchronize();
}

/**
 * @brief Host wrapper for launching compactPointsBitmaskSurface kernel.
 *
 * @param d_input        Device input pointer
 * @param surfaceOutput  Bound CUDA surface object (must be created in host)
 * @param d_count        Device int pointer (should be zeroed before call)
 * @param num_points     Number of input points
 * @param surface_width  Width of the 2D surface in float4 elements
 */
void compact_points_bitmask_surface(
    const Point2D* d_input,
    cudaSurfaceObject_t surfaceOutput,
    int* d_count,
    int num_points,
    int surface_width
) {
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((num_points + blockDim.x - 1) / blockDim.x);

    cudaMemset(d_count, 0, sizeof(int));  // ensure count starts from 0

    compactPointsBitmaskSurface<<<gridDim, blockDim>>>(
        d_input, surfaceOutput, d_count, num_points, surface_width
    );
    cudaDeviceSynchronize();
}

/**
 * @brief Test bitmask-based GPU stream compaction.
 *
 * @param input Input vector of Point2D
 * @param threshold Threshold value to determine "hot" points
 * @param output Output vector to store compacted result
 */
/*
void testBitmaskGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    std::cout << "\n[GPU] Testing Bitmask Stream Compaction..." << std::endl;

    int N = input.size();

    // Allocate device memory
    Point2D* d_input;
    Point2D* d_output;
    int* d_count;
    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMalloc(&d_output, N * sizeof(Point2D));
    cudaMalloc(&d_count, sizeof(int));

    // Copy input to device
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);

    // ✅ Copy threshold to constant memory
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // Set threshold if needed — skip this if you hardcoded in device predicate
    // setHotPredicateThreshold(threshold);  // optional if threshold is global

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch compaction
    compact_points_bitmask(d_input, d_output, d_count, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "    [Timing] Elapsed time: " << milliseconds << " ms\n";

    // Copy result count and data back to host
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    output.resize(h_count);
    cudaMemcpy(output.data(), d_output, h_count * sizeof(Point2D), cudaMemcpyDeviceToHost);

    std::cout << "    [Result] Compacted count = " << h_count << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
*/

/**
 * @brief Test bitmask-based GPU stream compaction.
 *
 * @param input Input vector of Point2D
 * @param threshold Threshold value to determine "hot" points
 * @param output Output vector to store compacted result
 */
void testBitmaskGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    std::cout << "\n[GPU] Testing Bitmask Stream Compaction..." << std::endl;

    int N = input.size();

    // CUDA timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);  // ✅ Start timing before all GPU operations

    // Allocate device memory
    Point2D* d_input;
    Point2D* d_output;
    int* d_count;
    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMalloc(&d_output, N * sizeof(Point2D));
    cudaMalloc(&d_count, sizeof(int));

    // Copy input to device
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);

    // Copy threshold to constant memory
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // Launch bitmask-based compaction kernel
    compact_points_bitmask(d_input, d_output, d_count, N);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "⏱️ Bitmask GPU total time (including memory ops): " << milliseconds << " ms\n";

    // Copy result count and compacted data back to host
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    output.resize(h_count);
    cudaMemcpy(output.data(), d_output, h_count * sizeof(Point2D), cudaMemcpyDeviceToHost);

    std::cout << "📌 Compacted count: " << h_count << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * @brief Test bitmask stream compaction using CUDA surface memory.
 *
 * Measures total GPU execution time including memory allocation, data transfer,
 * kernel launch, and surface operations.
 *
 * @param input Input vector of Point2D
 * @param threshold Threshold for hot point selection
 * @param surface_width Width of the 2D surface used to store results
 */
void testBitmaskSurfaceGPUCompaction(const std::vector<Point2D>& input, float threshold, int surface_width) {
    std::cout << "\n[GPU] Testing Bitmask Surface Stream Compaction..." << std::endl;

    int N = input.size();
    int surface_height = (N + surface_width - 1) / surface_width;

    // ✅ Start timing before ALL GPU operations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate device input
    Point2D* d_input;
    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);

    // Allocate compacted count
    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    // Create CUDA array and surface object
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, surface_width, surface_height, cudaArraySurfaceLoadStore);

    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaSurfaceObject_t surfaceObject = 0;
    cudaCreateSurfaceObject(&surfaceObject, &resDesc);

    // Copy threshold to constant memory
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // Launch compaction into surface
    compact_points_bitmask_surface(d_input, surfaceObject, d_count, N, surface_width);

    // Copy compacted count back to host ✅
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaDestroySurfaceObject(surfaceObject);
    cudaFreeArray(cuArray);
    cudaFree(d_input);
    cudaFree(d_count);

    // ✅ Stop timing after ALL GPU operations
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "⏱️ Bitmask Surface GPU total time (including memory ops): " << milliseconds << " ms\n";

    // ✅ Print final count (aligned with other outputs)
    std::cout << "📌 Compacted count: " << h_count << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


/**
 * @brief GPU bitmask stream compaction using float precision.
 *
 * This function launches the compact_points_bitmask kernel, measures
 * the GPU execution time (including data transfer), and copies the
 * result back to host.
 *
 * @param input     Input vector of Point2D
 * @param threshold Temperature threshold for hot points
 * @param blockSize Thread block size to use in kernel launch
 * @param time_ms   Output variable to store measured time in ms
 * @param output    Output vector storing compacted result
 */
void bitmask_stream_compaction_gpu_float(const std::vector<Point2D>& input,
                                         float threshold,
                                         int blockSize,
                                         float& time_ms,
                                         std::vector<Point2D>& output) {
    int N = input.size();

    // --- Setup CUDA timer ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- Allocate memory ---
    Point2D* d_input = nullptr;
    Point2D* d_output = nullptr;
    int* d_count = nullptr;

    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMalloc(&d_output, N * sizeof(Point2D));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    // --- Copy input ---
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);

    // --- Copy threshold to constant memory ---
    cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));

    // --- Kernel launch configuration ---
    int gridSize = (N + blockSize - 1) / blockSize;
    //compact_points_bitmask<<<gridSize, blockSize>>>(d_input, d_output, d_count, N);
    compactPointsBitmask<<<gridSize, blockSize>>>(d_input, d_output, d_count, N);

    // --- Copy results back ---
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    output.resize(h_count);
    cudaMemcpy(output.data(), d_output, h_count * sizeof(Point2D), cudaMemcpyDeviceToHost);

    // --- Stop timing ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    // --- Clean up ---
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



/**
 * @brief CUDA kernel for stream compaction using warp-level bitmask voting (double version).
 *
 * Each thread determines whether its assigned point meets the compaction condition.
 * A warp-level bitmask is generated using __ballot_sync, and each thread calculates
 * its compacted output index using __popc. This version works on Point2D_double structure
 * and uses double-precision threshold comparison.
 *
 * @param d_input Pointer to device-side input array of Point2D_double
 * @param d_output Pointer to device-side output array for compacted points
 * @param d_count Pointer to device-side int used to store final compacted count
 * @param num_points Total number of input points
 */
__global__ void compact_points_bitmask_double(
    const Point2D_double* d_input,
    Point2D_double* d_output,
    int* d_count,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point2D_double pt = d_input[idx];
    bool isHot = isHotPredicateDevice(pt);  // must be double version
    //printf("debug isHot: idx = %d, isHot = %d, temp = %f\n", idx, isHot, pt.temp);

    unsigned int lane_id = threadIdx.x % 32;
    unsigned int warp_id = threadIdx.x / 32;

    // Warp-wide vote: bitmask of active threads
    unsigned int mask = __ballot_sync(0xffffffff, isHot);

    // Count how many threads before me voted true
    int local_pos = __popc(mask & ((1u << lane_id) - 1));

    // Warp leader reserves space for this warp in global output
    __shared__ int warp_offsets[BLOCK_SIZE / 32];
    int total_in_warp = __popc(mask);

    if (lane_id == 0) {
        warp_offsets[warp_id] = atomicAdd(d_count, total_in_warp);
    }

    __syncthreads();

    // Write point to compacted array if hot
    if (isHot) {
        int output_index = warp_offsets[warp_id] + local_pos;
        d_output[output_index] = pt;
    }
}

/**
 * @brief Device-side predicate function for identifying "hot" points (double precision).
 *
 * This function determines whether a given 2D point exceeds the temperature threshold.
 * It compares the temperature field `temp` of a Point2D_double against a constant
 * double-precision threshold value `d_threshold_double`, which resides in device constant memory.
 *
 * @param pt The input point of type Point2D_double.
 * @return true if the point's temperature is greater than the threshold; false otherwise.
 */
/*
__device__ bool isHotPredicateDevice(Point2D_double pt) {
    return pt.temp > d_threshold_double;
}
*/


/**
 * @brief GPU stream compaction using warp-level bitmask (double-precision version).
 *
 * This function allocates memory on the GPU, transfers input data,
 * launches a bitmask-based stream compaction kernel operating on double-precision
 * Point2D_double structures, and retrieves the compacted results.
 *
 * @param input Input vector of Point2D_double
 * @param threshold Temperature threshold to determine "hot" points
 * @param blockSize CUDA thread block size
 * @param time_ms Output variable to store measured kernel time (in milliseconds)
 * @param output Output vector of compacted Point2D_double results
 */
void bitmask_stream_compaction_gpu_double(const std::vector<Point2D_double>& input,
                                          double threshold,
                                          int blockSize,
                                          float& time_ms,
                                          std::vector<Point2D_double>& output) {
    int N = input.size();
    std::cout << "[DEBUG] Launching DOUBLE precision kernel with threshold = " << threshold << "\n";

    // --- Setup CUDA timer ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- Allocate memory ---
    Point2D_double* d_input = nullptr;
    Point2D_double* d_output = nullptr;
    int* d_count = nullptr;

    cudaMalloc(&d_input, N * sizeof(Point2D_double));
    cudaMalloc(&d_output, N * sizeof(Point2D_double));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    // --- Error check for allocations ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc or cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // --- Copy input ---
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D_double), cudaMemcpyHostToDevice);

    // --- Copy threshold to constant memory ---
    cudaMemcpyToSymbol(d_threshold_double, &threshold, sizeof(double));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpy or cudaMemcpyToSymbol failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // --- Kernel launch configuration ---
    int gridSize = (N + blockSize - 1) / blockSize;
    compact_points_bitmask_double<<<gridSize, blockSize>>>(d_input, d_output, d_count, N);

    // --- Check kernel launch ---
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // --- Wait for kernel to complete ---
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // --- Copy results back ---
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_count > N) {
        std::cerr << "[ERROR] Invalid output count: h_count = " << h_count << " > N = " << N << std::endl;
        return;
    }

    output.resize(h_count);
    cudaMemcpy(output.data(), d_output, h_count * sizeof(Point2D_double), cudaMemcpyDeviceToHost);

    // --- Final error check for memcpy ---
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpy output failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // --- Stop timing ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    // --- Clean up ---
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

