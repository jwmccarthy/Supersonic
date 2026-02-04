#pragma once

#include "CudaCommon.cuh"
#include "CudaMath.cuh"
#include "RLConstants.cuh"
#include "GameState.cuh"
#include "ArenaMesh.cuh"

struct AABB
{
    float4 min;
    float4 max;
};

__device__ __forceinline__ AABB getCarAABB(ArenaMesh* arena, float4 pos, float4 rot)
{
    float4 aabbMin = ARENA_MAX;
    float4 aabbMax = ARENA_MIN;

    // Check world space car corners
    #pragma unroll 8
    for (int i = 0; i < 8; ++i)
    {
        float4 local = {
            CAR_OFFSETS.x + CAR_HALF_EX.x * ((i & 1) ? 1.f : -1.f),
            CAR_OFFSETS.y + CAR_HALF_EX.y * ((i & 2) ? 1.f : -1.f),
            CAR_OFFSETS.z + CAR_HALF_EX.z * ((i & 4) ? 1.f : -1.f), 0.0f
        };

        float4 world = vec3::add(pos, quat::toWorld(local, rot));
        aabbMin = vec3::min(aabbMin, world);
        aabbMax = vec3::max(aabbMax, world);
    }

    return { aabbMin, aabbMax };
}

// Broad phase: compute cell bounds and count triangles (no AABB tests yet)
__device__ __forceinline__ void carArenaBroadPhase(GameState* state, ArenaMesh* arena, Workspace* space, int carIdx)
{
    float4 pos = __ldg(&state->cars.position[carIdx]);
    float4 rot = __ldg(&state->cars.rotation[carIdx]);

    auto [aabbMin, aabbMax] = getCarAABB(arena, pos, rot);

    int3 cellMin = arena->getCellIdx(aabbMin);
    int3 cellMax = arena->getCellIdx(aabbMax);

    // Count total triangles in overlapping cells
    int triCount = 0;
    for (int x = cellMin.x; x <= cellMax.x; ++x)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        int cellIdx = arena->flatCellIdx(x, y, z);
        triCount += __ldg(&arena->triPre[cellIdx + 1]) - __ldg(&arena->triPre[cellIdx]);
    }

    // Store for narrow phase
    space->triCounts[carIdx] = triCount;
    space->cellMin[carIdx] = cellMin;
    space->cellMax[carIdx] = cellMax;
}

// Narrow phase: one thread per (car, triangle) pair - does AABB test
__device__ __forceinline__ void carArenaNarrowPhase(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int carIdx,
    int localTriIdx)
{
    float4 pos = __ldg(&state->cars.position[carIdx]);
    float4 rot = __ldg(&state->cars.rotation[carIdx]);

    auto [aabbMin, aabbMax] = getCarAABB(arena, pos, rot);
    auto [minX, minY, minZ, minW] = aabbMin;
    auto [maxX, maxY, maxZ, maxW] = aabbMax;

    int3 cellMin = space->cellMin[carIdx];
    int3 cellMax = space->cellMax[carIdx];

    // Find the triangle corresponding to localTriIdx
    int idx = 0;
    for (int x = cellMin.x; x <= cellMax.x; ++x)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        int cellIdx = arena->flatCellIdx(x, y, z);
        int triBeg = __ldg(&arena->triPre[cellIdx]);
        int triEnd = __ldg(&arena->triPre[cellIdx + 1]);
        int cellTriCount = triEnd - triBeg;

        if (localTriIdx < idx + cellTriCount)
        {
            // Found the cell containing our triangle
            int t = triBeg + (localTriIdx - idx);
            int triIdx = __ldg(&arena->triIdx[t]);

            float4 triMin = __ldg(&arena->aabbMin[triIdx]);
            float4 triMax = __ldg(&arena->aabbMax[triIdx]);

            bool hit = (
                minX <= triMax.x && maxX >= triMin.x &&
                minY <= triMax.y && maxY >= triMin.y &&
                minZ <= triMax.z && maxZ >= triMin.z
            );

            if (hit)
            {
                atomicAdd(space->hitCount, 1);
            }
            return;
        }
        idx += cellTriCount;
    }
}