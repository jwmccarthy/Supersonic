#pragma once

#include "CudaCommon.cuh"
#include "CudaMath.cuh"
#include "RLConstants.cuh"
#include "GameState.cuh"
#include "ArenaMesh.cuh"

struct CellBounds
{
    int3 min;
    int3 max;
};

__device__ __forceinline__ Triangle getTriVerts(ArenaMesh* arena, int i)
{
    int triIdx = arena->triIdx[i];
    int4 verts = arena->tris[triIdx];
    return { 
        arena->verts[verts.x],
        arena->verts[verts.y],
        arena->verts[verts.z]
    };
}

__device__ __forceinline__ CellBounds getCellBounds(ArenaMesh* arena, float4 pos, float4 rot)
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

    // Get bound grid cell indices
    int3 cellMin = arena->getCellIdx(aabbMin);
    int3 cellMax = arena->getCellIdx(aabbMax);

    return { cellMin, cellMax };
}

__device__ __forceinline__ void carArenaBroadPhase(GameState* state, ArenaMesh* arena, int carIdx)
{
    // Cached access of car state
    float4 pos = state->cars.position[carIdx];
    float4 rot = state->cars.rotation[carIdx];

    // Compute 3D cell bounds given car position & rotation
    auto [cellMin, cellMax] = getCellBounds(arena, pos, rot);

    for (int x = cellMin.x; x <= cellMax.x; ++x)
    for (int y = cellMin.y; y <= cellMax.y; ++y)
    for (int z = cellMin.z; z <= cellMax.z; ++z)
    {
        int cellIdx = arena->flatCellIdx(x, y, z);

        // Cached loads of tri index bounds
        int triBeg = arena->triPre[cellIdx];
        int triEnd = arena->triPre[cellIdx + 1];
        
        for (int t = 0; t < triEnd - triBeg; ++t)
        {
            // Fetch world space tri vertices
            auto [v0, v1, v2] = getTriVerts(arena, triBeg + t);

            // Triangle AABB
            float4 triMin = vec3::min(vec3::min(v0, v1), v2);
            float4 triMax = vec3::max(vec3::max(v0, v1), v2);
        }
    }
    
    // printf("Car: %d | Overlaps: %d\n", carIdx, overlapsDEBUG);
}