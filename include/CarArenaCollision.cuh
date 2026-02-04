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

__device__ __forceinline__ void carArenaBroadPhase(GameState* state, ArenaMesh* arena, Workspace* space, int carIdx)
{
    // Cached access of car state
    float4 pos = __ldg(&state->cars.position[carIdx]);
    float4 rot = __ldg(&state->cars.rotation[carIdx]);

    // Compute 3D cell bounds given car position & rotation
    auto [aabbMin, aabbMax] = getCarAABB(arena, pos, rot);

    // Get bound grid cell indices
    int3 cellMin = arena->getCellIdx(aabbMin);
    int3 cellMax = arena->getCellIdx(aabbMax);

    auto [minX, minY, minZ, minW] = aabbMin;
    auto [maxX, maxY, maxZ, maxW] = aabbMax;

    // Pre-allocated slot base for this car
    int slotBase = carIdx * MAX_PAIRS_PER_CAR;
    int count = 0;

    for (int x = cellMin.x; x <= cellMax.x; ++x)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        int cellIdx = arena->flatCellIdx(x, y, z);

        // Cached loads of tri index bounds
        int triBeg = __ldg(&arena->triPre[cellIdx]);
        int triEnd = __ldg(&arena->triPre[cellIdx + 1]);

        for (int t = triBeg; t < triEnd; ++t)
        {
            int triIdx = __ldg(&arena->triIdx[t]);

            // Triangle AABB
            float4 triMin = __ldg(&arena->aabbMin[triIdx]);
            float4 triMax = __ldg(&arena->aabbMax[triIdx]);

            // Test AABB overlap
            bool hit = (
                minX <= triMax.x && maxX >= triMin.x &&
                minY <= triMax.y && maxY >= triMin.y &&
                minZ <= triMax.z && maxZ >= triMin.z
            );

            if (hit && count < MAX_PAIRS_PER_CAR)
            {
                space->pairs[slotBase + count++] = make_int2(carIdx, triIdx);
            }
        }
    }

    space->counts[carIdx] = count;
}