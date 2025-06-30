// stream_compaction.cu
#include <cuda_runtime.h>
#include "stream_compaction.h"
#include "common.h"          // <- contains typedef Point, predicate, etc.
#include <iostream>

// -------------------- device predicate --------------------
/**
 * @brief Device-side predicate function.
 *        Determines whether a given Point meets the compaction condition.
 * @param p Input point
 * @return true if the point should be retained, false otherwise
 */
__device__ inline bool isHotPredicateDevice(const Point2D& p)
{
    isHotPredicate pred{30.0f}; 
    return pred(p);
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

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    streamCompactNaive<<<blocks, threads>>>(d_in, d_out, N, d_counter);

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
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // Allocate device memory for per-block counts
    int* d_block_counts;
    cudaMalloc(&d_block_counts, blocks * sizeof(int));

    // Calculate shared memory size: flags + Point2D array
    size_t shared_mem_bytes = threads * (sizeof(unsigned int) + sizeof(Point2D));

    // Launch shared memory kernel
    streamCompactShared<<<blocks, threads, shared_mem_bytes>>>(
        d_in, d_out, N, threshold, d_block_counts);

    // Copy per-block results back to host
    std::vector<int> h_block_counts(blocks);
    cudaMemcpy(h_block_counts.data(), d_block_counts,
               blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Sum block counts to get total compacted output count
    h_outCount = 0;
    for (int i = 0; i < blocks; ++i) {
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
void testNaiveGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    int N = input.size();
    Point2D* d_input = nullptr;
    Point2D* d_output = nullptr;
    int compactedCount = 0;

    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, N * sizeof(Point2D));

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

/**
 * @brief Test the shared memory + block scan GPU stream compaction method with timing.
 *
 * This function allocates memory, launches the optimized shared memory compaction kernel,
 * copies results back to host, and reports execution time.
 *
 * @param input        Input vector of Point2D elements.
 * @param threshold    Temperature threshold for filtering.
 * @param output       Vector to store the compacted results.
 */
void testSharedGPUCompaction(const std::vector<Point2D>& input, float threshold, std::vector<Point2D>& output) {
    int N = input.size();
    Point2D* d_input = nullptr;
    Point2D* d_output = nullptr;
    int compactedCount = 0;

    cudaMalloc(&d_input, N * sizeof(Point2D));
    cudaMemcpy(d_input, input.data(), N * sizeof(Point2D), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, N * sizeof(Point2D));

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
