#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>

#include "CudaCommon.cuh"
#include "CudaKernels.cuh"
#include "ArenaMesh.cuh"
#include "CarArenaCollision.cuh"
#include "RLEnvironment.cuh"
#include "StructAllocate.cuh"

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : sims(sims), numB(numB), numO(numO), seed(seed)
    , cars(sims * (numB + numO))
    , m_arena(MESH_PATH)
    , m_state(sims, numB, numO, seed)
{
    // Copy arena mesh and game state to device
    cudaMallocCpy(d_arena, &m_arena);
    cudaMallocCpy(d_state, &m_state);

    // Allocate collision workspace (hitCount, triCounts, triOffsets, cellMin, cellMax)
    cudaMallocSOA(m_space, {1, cars, cars + 1, cars, cars});
    cudaMallocCpy(d_space, &m_space);

    // Allocate CUB temp storage for prefix sum
    d_cubTemp = nullptr;
    cubTempBytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_cubTemp, cubTempBytes,
        m_space.triCounts, m_space.triOffsets, cars + 1);
    CUDA_CHECK(cudaMalloc(&d_cubTemp, cubTempBytes));

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
}

float* RLEnvironment::step()
{
    int blockSize = 256;
    int gridSize = (cars + blockSize - 1) / blockSize;

    // Reset hit counter
    cudaMemsetAsync(m_space.hitCount, 0, sizeof(int));

    // 1. Broad phase - compute cell bounds and triangle counts per car
    carArenaCollisionKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space);

    // 2. Prefix sum to get offsets for thread mapping
    cub::DeviceScan::ExclusiveSum(
        d_cubTemp, cubTempBytes,
        m_space.triCounts, m_space.triOffsets, cars + 1);

    // 3. Get total triangles (last element of prefix sum)
    int totalTris;
    cudaMemcpy(&totalTris, m_space.triOffsets + cars, sizeof(int), cudaMemcpyDeviceToHost);

    // 4. Narrow phase - one thread per (car, triangle) pair
    if (totalTris > 0)
    {
        int npBlocks = (totalTris + blockSize - 1) / blockSize;
        carArenaNarrowPhaseKernel<<<npBlocks, blockSize>>>(
            d_state, d_arena, d_space, cars, totalTris);
    }

    cudaDeviceSynchronize();

    // Debug: print stats every 1000 frames
    static int frame = 0;
    if (frame++ % 1000 == 0)
    {
        int hitCount;
        cudaMemcpy(&hitCount, m_space.hitCount, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Frame %d: totalTris=%d, npBlocks=%d, hitCount=%d\n",
               frame, totalTris, totalTris > 0 ? (totalTris + blockSize - 1) / blockSize : 0, hitCount);
    }

    return d_output;
}

float* RLEnvironment::reset()
{
    int blockSize = 32;
    int gridSize = (sims + blockSize - 1) / blockSize;

    resetKernel<<<gridSize, blockSize>>>(d_state);
    cudaDeviceSynchronize();

    return d_output;
}
