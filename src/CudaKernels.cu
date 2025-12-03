#include <cuda_runtime.h>

#include "CudaKernels.hpp"
#include "GameState.hpp"
#include "StateReset.hpp"
#include "Collisions.hpp"

__global__ void resetKernel(GameState* state)
{
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims) return;

    resetToKickoff(state, simIdx);
}

__global__ void collisionTestKernel(GameState* state)
{
    int numCars = state->nCar;
    int numCols = numCars * (numCars - 1) / 2;
    
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (globalIdx >= state->sims * numCols) return;
    
    int simIdx = globalIdx / numCols;
    int colIdx = globalIdx % numCols;
    
    // Map linear index to unique pair (i, j) where i < j
    int i = (int)(sqrtf(2.0f * colIdx + 0.25f) + 0.5f);
    int j = colIdx - i * (i - 1) / 2;
    
    // Base index for the car state arrays
    const int carBase = simIdx * numCars;

    // Car pair positions and rotations
    float4 posA = state->cars.position[carBase + j];
    float4 rotA = state->cars.rotation[carBase + j];
    float4 posB = state->cars.position[carBase + i];
    float4 rotB = state->cars.rotation[carBase + i];

    bool overlap = carCarCollision(posA, rotA, posB, rotB);

    if (overlap)
    {
        // printf("Collision detected: Cars %d and %d in sim %d\n", i, j, simIdx);
    }
}