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

// Binary search to find which car owns this thread
__device__ int findCarIdx(int* triOffsets, int totalCars, int threadIdx)
{
    int lo = 0, hi = totalCars;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        
        if (triOffsets[mid + 1] <= threadIdx)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

// AABB filter kernel: test car AABB vs triangle AABB
__global__ void carArenaAABBFilterKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int totalBroadTris)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalBroadTris) return;

    int totalCars = state->sims * state->nCar;
    int carIdx = findCarIdx(space->triOff, totalCars, tid);
    int localTriIdx = tid - space->triOff[carIdx];

    carArenaAABBFilter(arena, space, carIdx, localTriIdx, tid);
}

// Compaction kernel: write surviving (carIdx, triIdx) pairs
__global__ void carArenaCompactKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int totalBroadTris)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalBroadTris) return;

    int totalCars = state->sims * state->nCar;
    int carIdx = findCarIdx(space->triOff, totalCars, tid);
    int localTriIdx = tid - space->triOff[carIdx];

    carArenaCompact(arena, space, carIdx, localTriIdx, tid);
}

// Narrow phase kernel: full SAT test on AABB-overlapping triangles only
__global__ void carArenaNarrowPhaseKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int totalNarrowTris)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalNarrowTris) return;

    carArenaNarrowPhase(state, arena, space, tid);
}

__global__ void carArenaCollisionKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space)
{
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCars = state->sims * state->nCar;

    // Broad phase - compute group bounds and triangle counts
    if (carIdx < totalCars)
    {
        carArenaBroadPhase(state, arena, space, carIdx);
    }
}
