#include <cuda_runtime.h>

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

    // Allocate collision workspace (counts per car, pre-allocated pair slots)
    cudaMallocSOA(m_space, {cars, cars * MAX_PAIRS_PER_CAR});
    cudaMallocCpy(d_space, &m_space);

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
}

float* RLEnvironment::step()
{
    int blockSize = 256;
    int gridSize = (cars + blockSize - 1) / blockSize;

    carArenaCollisionKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_space);
    cudaDeviceSynchronize();

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
