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

    // Reset count
    if (idx == 0) space->count = 0;

    grid.sync();

    // Broad phase
    if (idx < state->sims * state->nCar)
    {
        carArenaBroadPhase(state, arena, space, idx);
    }

    grid.sync();

    // Narrow phase
    if (idx < space->count)
    {
        carArenaNarrowPhase(state, arena, space, idx);
    }
}