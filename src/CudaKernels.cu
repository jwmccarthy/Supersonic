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
    int totalCars,
    int totalTris)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalTris) return;

    // Find which car this thread belongs to
    int carIdx = findCarIdx(space->triOffsets, totalCars, tid);
    int localTriIdx = tid - space->triOffsets[carIdx];

    carArenaNarrowPhase(state, arena, space, carIdx, localTriIdx);
}

// Simple prefix sum kernel (single block, for small arrays)
__global__ void prefixSumKernel(int* triCounts, int* triOffsets, int n, int* totalOut)
{
    extern __shared__ int temp[];

    int tid = threadIdx.x;

    // Load into shared memory
    temp[tid] = (tid < n) ? triCounts[tid] : 0;
    __syncthreads();

    // Exclusive scan (Blelloch)
    for (int stride = 1; stride < n; stride *= 2)
    {
        int val = 0;
        if (tid >= stride)
            val = temp[tid - stride];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write output (exclusive prefix sum)
    if (tid < n)
        triOffsets[tid] = temp[tid] - triCounts[tid];
    if (tid == 0)
    {
        triOffsets[n] = temp[n - 1];  // Total at end
        *totalOut = temp[n - 1];
    }
}

__global__ void carArenaCollisionKernel(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int* totalTris)
{
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCars = state->sims * state->nCar;

    // Broad phase - compute cell bounds and triangle counts
    if (carIdx < totalCars)
    {
        carArenaBroadPhase(state, arena, space, carIdx);
    }
}