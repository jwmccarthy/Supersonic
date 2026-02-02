#include <cuda_runtime.h>

#include "CudaKernels.cuh"
#include "CudaMath.cuh"
#include "RLConstants.cuh"
#include "GameState.cuh"
#include "ArenaMesh.cuh"
#include "StateReset.cuh"
#include "CarArenaCollision.cuh"

__global__ void resetKernel(GameState* state)
{
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims) return;

    randomizeInitialPositions(state, simIdx);
}

__global__ void carArenaNarrowPhaseKernel(GameState* state, ArenaMesh* arena, Workspace* space)
{
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= space->count) return;

    carArenaNarrowPhase(state, arena, space, pairIdx);
}

__global__ void carArenaBroadPhaseKernel(GameState* state, ArenaMesh* arena, Workspace* space)
{
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCars = state->sims * state->nCar;

    // Reset count (single thread)
    if (carIdx == 0) space->count = 0;
    __syncthreads();

    // Broad phase
    if (carIdx < totalCars)
    {
        carArenaBroadPhase(state, arena, space, carIdx);
    }

    // Last block tail-launches narrow phase
    __shared__ int lastBlock;
    if (threadIdx.x == 0)
    {
        lastBlock = atomicAdd(&space->broadDone, 1) == (gridDim.x - 1);
    }
    __syncthreads();

    if (lastBlock && threadIdx.x == 0)
    {
        int blockSize = 128;
        int gridSize = (space->count + blockSize - 1) / blockSize;
        if (gridSize > 0)
        {
            carArenaNarrowPhaseKernel<<<gridSize, blockSize>>>(state, arena, space);
        }
    }
}