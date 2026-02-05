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
    , m_broadTris(0), m_narrowTris(0), m_maxBroad(0), m_nHit(0)
{
    // Copy arena mesh and game state to device
    cudaMallocCpy(d_arena, &m_arena);
    cudaMallocCpy(d_state, &m_state);

    // Estimate max tris from broad phase (worst case: all cars in densest group)
    // Use a generous upper bound based on average tris per group * cars * safety factor
    m_maxBroad = (m_arena.nTris / m_arena.nGroups + 1) * cars * 4;

    // Allocate collision workspace
    // Fields: numHit(1), numTri(cars), triOff(cars+1), groupIdx(cars),
    //         carAABBMin(cars), carAABBMax(cars), narrowCount(1),
    //         compactCarIdx(maxBroad), compactTriIdx(maxBroad)
    cudaMallocSOA(m_space, {1, cars, cars + 1, cars,
                           cars, cars, 1,
                           m_maxBroad, m_maxBroad});
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

    // Reset counters
    cudaMemsetAsync(m_space.numHit, 0, sizeof(int));
    cudaMemsetAsync(m_space.narrowCount, 0, sizeof(int));

    // Broad phase - compute group bounds, triangle counts, and car AABBs
    carArenaCollisionKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space);

    // Prefix sum to get offsets for thread mapping
    cub::DeviceScan::ExclusiveSum(d_cubBuf, cubBytes, m_space.numTri, m_space.triOff, cars + 1);

    // Get total triangles from broad phase
    cudaMemcpy(&m_broadTris, m_space.triOff + cars, sizeof(int), cudaMemcpyDeviceToHost);

    if (m_broadTris > 0)
    {
        // Warp-level AABB filter + compaction (single kernel, one atomic per warp)
        gridSize = (m_broadTris + blockSize - 1) / blockSize;
        carArenaFilterCompactKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space, m_broadTris);

        // Get total AABB-overlapping triangles from atomic counter
        cudaMemcpy(&m_narrowTris, m_space.narrowCount, sizeof(int), cudaMemcpyDeviceToHost);

        if (m_narrowTris > 0)
        {
            // Narrow phase - full SAT test on AABB-overlapping triangles only
            gridSize = (m_narrowTris + blockSize - 1) / blockSize;
            carArenaNarrowPhaseKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space, m_narrowTris);
        }
    }

    cudaDeviceSynchronize();

    // Debug: print stats every 1000 frames
    static int frame = 0;
    if (++frame % 1000 == 0)
    {
        cudaMemcpy(&m_nHit, m_space.numHit, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Frame %d: broad=%d narrow=%d hits=%d\n", frame, m_broadTris, m_narrowTris, m_nHit);
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
