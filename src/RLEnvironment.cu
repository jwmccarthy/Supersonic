#include <cuda_runtime.h>
#include <iostream>

#include "CudaCommon.cuh"
#include "CudaKernels.cuh"
#include "StructAllocate.cuh"
#include "ArenaMesh.cuh"
#include "CarArenaCollision.cuh"
#include "RLEnvironment.cuh"

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : sims(sims), numB(numB), numO(numO), seed(seed)
    , cars(sims * (numB + numO))
    , m_arena(MESH_PATH)
    , m_state(sims, numB, numO, seed)
{
    // Copy arena mesh and game state to device
    cudaMallocCpy(d_arena, &m_arena);
    cudaMallocCpy(d_state, &m_state);

    // Allocate intermediate workspaces
    cudaMallocSOA(m_space, cars * MAX_PER_CAR);
    cudaMallocCpy(d_space, &m_space);

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_output, sims * sizeof(float)));
}

float* RLEnvironment::step()
{
    int blockSize = 128;
    int gridSize = (cars + blockSize - 1) / blockSize;

    // Reset counters
    cudaMemset(&d_space->broadDone, 0, sizeof(int));
    cudaMemset(&d_space->narrowHits, 0, sizeof(int));

    // Broad phase tail-launches narrow phase
    carArenaCollisionKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space);
    cudaDeviceSynchronize();

    // Debug: accumulate SAT hit stats
    int totalPairs, satHits;
    cudaMemcpy(&totalPairs, &d_space->count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&satHits, &d_space->narrowHits, sizeof(int), cudaMemcpyDeviceToHost);
    debugTotalPairs += totalPairs;
    debugSatHits += satHits;

    return d_output;
}

void RLEnvironment::printSatStats()
{
    if (debugTotalPairs > 0)
    {
        std::cerr << "SAT: " << debugSatHits << "/" << debugTotalPairs
                  << " pairs had contacts (" << (100.0 * debugSatHits / debugTotalPairs) << "%)" << std::endl;
    }
}

float* RLEnvironment::reset()
{
    int blockSize = 32;
    int gridSize = (sims + blockSize - 1) / blockSize;

    resetKernel<<<gridSize, blockSize>>>(d_state);
    cudaDeviceSynchronize();

    return d_output;
}