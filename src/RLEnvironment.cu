#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>

#include "CudaCommon.cuh"
#include "CudaKernels.cuh"
#include "ArenaMesh.cuh"
#include "CarArenaCollision.cuh"
#include "RLEnvironment.cuh"
#include "StructAllocate.cuh"

// Fixed temp buffer size for CUB scan (generous, avoids query overhead)
constexpr size_t CUB_TEMP_BYTES = 1 << 20;  // 1MB

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : sims(sims), numB(numB), numO(numO), seed(seed)
    , cars(sims * (numB + numO))
    , cubBytes(CUB_TEMP_BYTES)
    , m_arena(MESH_PATH)
    , m_state(sims, numB, numO, seed)
{
    // Copy arena mesh and game state to device
    cudaMallocCpy(d_arena, &m_arena);
    cudaMallocCpy(d_state, &m_state);

    // Allocate collision workspace
    cudaMallocSOA(m_space, {1, cars, cars + 1, cars, cars});
    cudaMallocCpy(d_space, &m_space);

    // Allocate CUB temp storage
    CUDA_CHECK(cudaMalloc(&d_cubBuf, cubBytes));

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
}

float* RLEnvironment::step()
{
    int blockSize = 256;
    int gridSize = (cars + blockSize - 1) / blockSize;

    // Reset hit counter
    cudaMemsetAsync(m_space.numHit, 0, sizeof(int));

    // Broad phase - compute cell bounds and triangle counts per car
    carArenaCollisionKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space);

    // Prefix sum to get offsets for thread mapping
    cub::DeviceScan::ExclusiveSum(d_cubBuf, cubBytes, m_space.numTri, m_space.triOff, cars + 1);

    // Get total triangles (last element of prefix sum)
    int totalTris;
    cudaMemcpy(&totalTris, m_space.triOff + cars, sizeof(int), cudaMemcpyDeviceToHost);

    // Narrow phase - one thread per (car, triangle) pair
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
        cudaMemcpy(&hitCount, m_space.numHit, sizeof(int), cudaMemcpyDeviceToHost);
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
