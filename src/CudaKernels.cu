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

// Warp-level AABB filter and compaction kernel
// Tests car AABB vs triangle AABB and compacts passing pairs using warp primitives
__global__ void carArenaFilterCompactKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int totalBroadTris)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x & 31;

    bool valid = tid < totalBroadTris;
    bool passes = false;
    int carIdx = 0, triIdx = 0;

    if (valid)
    {
        // Find which car this thread belongs to
        int totalCars = state->sims * state->nCar;
        carIdx = findCarIdx(space->triOff, totalCars, tid);
        int localTriIdx = tid - space->triOff[carIdx];

        // Get triangle index
        int4 groupIdx = space->groupIdx[carIdx];
        int groupFlat = arena->flatGroupIdx(groupIdx.x, groupIdx.y, groupIdx.z);
        int triBeg = __ldg(&arena->triPre[groupFlat]);
        triIdx = __ldg(&arena->triIdx[triBeg + localTriIdx]);

        // Load car AABB
        float4 carMin = space->carAABBMin[carIdx];
        float4 carMax = space->carAABBMax[carIdx];

        // Load triangle AABB
        float4 triMin = __ldg(&arena->aabbMin[triIdx]);
        float4 triMax = __ldg(&arena->aabbMax[triIdx]);

        // AABB overlap test
        passes = (carMin.x <= triMax.x && carMax.x >= triMin.x) &&
                 (carMin.y <= triMax.y && carMax.y >= triMin.y) &&
                 (carMin.z <= triMax.z && carMax.z >= triMin.z);
    }

    // Warp ballot - get mask of passing threads
    unsigned mask = __ballot_sync(0xFFFFFFFF, passes);
    int warpPasses = __popc(mask);

    // Warp-level prefix sum for local offset within warp
    unsigned lowerMask = (1u << laneId) - 1;
    int localOffset = __popc(mask & lowerMask);

    // Lane 0 atomically allocates space for entire warp
    int warpBase = 0;
    if (laneId == 0 && warpPasses > 0)
    {
        warpBase = atomicAdd(space->narrowCount, warpPasses);
    }

    // Broadcast warpBase to all lanes
    warpBase = __shfl_sync(0xFFFFFFFF, warpBase, 0);

    // Write to compact arrays
    if (passes)
    {
        int outIdx = warpBase + localOffset;
        space->compactCarIdx[outIdx] = carIdx;
        space->compactTriIdx[outIdx] = triIdx;
    }
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
