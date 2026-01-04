#include <cuda_runtime.h>

#include "CudaCommon.hpp"
#include "CudaKernels.cuh"
#include "ArenaMesh.cuh"
#include "RLEnvironment.hpp"

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : sims(sims), numB(numB), numO(numO), seed(seed), 
      m_arena(MESH_PATH), 
      m_state(sims, numB, numO, seed)
{
    // Copy arena mesh and game state to device
    cudaMallocCpy(d_arena, &m_arena);
    cudaMallocCpy(d_state, &m_state);

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
}

float* RLEnvironment::step()
{
    // TODO: implement step
    return d_output;
}

float* RLEnvironment::reset()
{
    int blockSize = 32;
	int gridSize = (sims + blockSize - 1) / blockSize;

    resetKernel<<<gridSize, blockSize>>>(d_state);

    return d_output;
}