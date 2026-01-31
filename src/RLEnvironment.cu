#include <cuda_runtime.h>

#include "CudaCommon.cuh"
#include "CudaKernels.cuh"
#include "ArenaMesh.cuh"
#include "CarArenaCollision.cuh"
#include "RLEnvironment.cuh"

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : sims(sims), numB(numB), numO(numO), seed(seed)
    , m_arena(MESH_PATH)
    , m_state(sims, numB, numO, seed)
{
    // Copy arena mesh and game state to device
    cudaMallocCpy(d_arena, &m_arena);
    cudaMallocCpy(d_state, &m_state);

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    // Debug accumulator
    CUDA_CHECK(cudaMalloc(&d_debug, sizeof(int)));
}

float* RLEnvironment::step()
{
    int blockSize = 128;
    int gridSize = (sims * (numB + numO) + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaMemset(d_debug, 0, sizeof(int)));
    carArenaCollisionKernel<<<gridSize, blockSize>>>(d_state, d_arena, d_debug);
    cudaDeviceSynchronize();

    return d_output;
}

float* RLEnvironment::reset()
{
    int blockSize = 32;
    int gridSize = (sims + blockSize - 1) / blockSize;

    resetKernel<<<gridSize, blockSize>>>(d_state);

    return d_output;
}