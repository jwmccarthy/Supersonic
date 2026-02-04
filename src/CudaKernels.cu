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

    resetToKickoff(state, simIdx);
}

// Binary search to find which car owns this thread
__device__ int findCarIdx(int* triOffsets, int totalCars, int threadIdx)
{
    int lo = 0, hi = totalCars;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (triOffsets[mid + 1] <= threadIdx)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__global__ void carArenaNarrowPhaseKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int totalTris)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalTris) return;

    // Find which car this thread belongs to
    int totalCars = state->sims * state->nCar;
    int carIdx = findCarIdx(space->triOff, totalCars, tid);
    int localTriIdx = tid - space->triOff[carIdx];

    carArenaNarrowPhase(state, arena, space, carIdx, localTriIdx);
}

__global__ void carArenaCollisionKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space)
{
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCars = state->sims * state->nCar;

    // Broad phase - compute cell bounds and triangle counts
    if (carIdx < totalCars)
    {
        carArenaBroadPhase(state, arena, space, carIdx);
    }
}