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

__global__ void carArenaBroadPhaseKernel(GameState* state, ArenaMesh* arena, Workspace* space)
{
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= state->sims * state->nCar) return;

    carArenaBroadPhase(state, arena, space, carIdx);
}

__global__ void carArenaNarrowPhaseKernel(GameState* state, ArenaMesh* arena, Workspace* space)
{
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= space->count) return;

    carArenaNarrowPhase(state, arena, space, pairIdx);
}