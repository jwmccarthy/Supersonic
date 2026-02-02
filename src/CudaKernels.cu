#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "CudaKernels.cuh"
#include "CudaMath.cuh"
#include "RLConstants.cuh"
#include "GameState.cuh"
#include "ArenaMesh.cuh"
#include "StateReset.cuh"
#include "CarArenaCollision.cuh"

namespace cg = cooperative_groups;

__global__ void resetKernel(GameState* state)
{
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims) return;

    randomizeInitialPositions(state, simIdx);
}

__global__ void carArenaCollisionKernel(GameState* state, ArenaMesh* arena, Workspace* space)
{
    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int str = gridDim.x * blockDim.x;

    // Reset count
    if (idx == 0) space->count = 0;

    grid.sync();

    // Broad phase - grid-stride loop
    for (int carIdx = idx; carIdx < state->sims * state->nCar; carIdx += str)
    {
        carArenaBroadPhase(state, arena, space, carIdx);
    }

    grid.sync();

    // Narrow phase - grid-stride loop
    for (int pairIdx = idx; pairIdx < space->count; pairIdx += str)
    {
        carArenaNarrowPhase(state, arena, space, pairIdx);
    }
}