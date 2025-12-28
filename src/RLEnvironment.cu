#include <cuda_runtime.h>

#include "CudaCommon.hpp"
#include "RLEnvironment.hpp"
#include "CudaKernels.hpp"

RLEnvironment::RLEnvironment(int sims, int numB, int numO, int seed)
    : sims(sims), numB(numB), numO(numO), seed(seed),
      m_gameState(sims, numB, numO, seed)
{    
    // Allocate device ptr and copy from the member variable
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(GameState)));
    
    // Copy the struct (which contains the valid pointers) to the device
    CUDA_CHECK(cudaMemcpy(d_state, &m_gameState, sizeof(GameState), 
                    cudaMemcpyHostToDevice));

    // Allocate output buffer on device
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
}

float* RLEnvironment::step()
{
    // One thread per collision pair
    int nCar   = numB + numO;
    int nPairs = nCar * (nCar - 1) / 2;
    int nTotal = sims * nPairs;

    int blockSize = 256;
    int gridSize  = (nTotal + blockSize - 1) / blockSize;

    collisionKernel<<<gridSize, blockSize>>>(d_state);
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