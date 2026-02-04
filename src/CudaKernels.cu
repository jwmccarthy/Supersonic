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

__global__ void carArenaNarrowPhaseKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space)
{
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= *space->numPairs) return;

    // TODO: Implement narrow phase collision detection
}

__global__ void carArenaCollisionKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space)
{
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCars = state->sims * state->nCar;

    // Broad phase
    if (carIdx < totalCars)
    {
        carArenaBroadPhase(state, arena, space, carIdx);
    }

    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int nPairs = *space->numPairs;
        
        if (nPairs > 0)
        {
            int blockSize = 128;
            int gridSize = (nPairs + blockSize - 1) / blockSize;

            carArenaNarrowPhaseKernel<<<gridSize, blockSize>>>(state, arena, space);
        }

        // Reset counters for next invocation
        *space->blockCnt = 0;
        *space->numPairs = 0;
    }
}