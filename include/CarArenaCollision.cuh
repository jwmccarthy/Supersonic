#pragma once

#include "CudaCommon.cuh"
#include "CudaMath.cuh"
#include "RLConstants.cuh"
#include "GameState.cuh"
#include "ArenaMesh.cuh"
#include "Reflection.hpp"

struct AABB
{
    float4 min;
    float4 max;
};

__device__ __forceinline__ Triangle getTriVerts(ArenaMesh* arena, int t)
{
    int4 tri = __ldg(&arena->tris[t]);
    return {
        __ldg(&arena->verts[tri.x]),
        __ldg(&arena->verts[tri.y]),
        __ldg(&arena->verts[tri.z])
    };
}

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

__device__ __forceinline__ void carArenaBroadPhase(GameState* state, ArenaMesh* arena, int carIdx, int* debug)
{
    // Cached access of car state
    float4 pos = __ldg(&state->cars.position[carIdx]);
    float4 rot = __ldg(&state->cars.rotation[carIdx]);

    // Compute 3D cell bounds given car position & rotation
    auto [aabbMin, aabbMax] = getCarAABB(arena, pos, rot);

    // Get bound grid cell indices
    int3 cellMin = arena->getCellIdx(aabbMin);
    int3 cellMax = arena->getCellIdx(aabbMax);

    int overlaps = 0;

    for (int x = cellMin.x; x <= cellMax.x; ++x)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        int cellIdx = arena->flatCellIdx(x, y, z);

        // Cached loads of tri index bounds
        int triBeg = __ldg(&arena->triPre[cellIdx]);
        int triEnd = __ldg(&arena->triPre[cellIdx + 1]);

        for (int t = 0; t < triEnd - triBeg; ++t)
        {
            int triIdx = __ldg(&arena->triIdx[triBeg + t]);

            // Triangle AABB
            float4 triMin = __ldg(&arena->aabbMin[triIdx]);
            float4 triMax = __ldg(&arena->aabbMax[triIdx]);

            // Test AABB overlap
            overlaps += vec3::lte(aabbMin, triMax) && vec3::gte(aabbMax, triMin);
        }
    }

    // Keep nu
    state->cars.numTris[carIdx] = overlaps;
}
