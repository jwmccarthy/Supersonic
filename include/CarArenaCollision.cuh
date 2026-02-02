#pragma once

#include "CudaCommon.cuh"
#include "CudaMath.cuh"
#include "RLConstants.cuh"
#include "GameState.cuh"
#include "ArenaMesh.cuh"

constexpr int MAX_PER_CAR = 8;

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

__device__ __forceinline__ void carArenaBroadPhase(
    GameState* state, 
    ArenaMesh* arena, 
    Workspace* space,
    int carIdx)
{
    // Cached access of car state
    float4 pos = __ldg(&state->cars.position[carIdx]);
    float4 rot = __ldg(&state->cars.rotation[carIdx]);

    // Compute 3D cell bounds given car position & rotation
    auto [aabbMin, aabbMax] = getCarAABB(arena, pos, rot);

    // Get bound grid cell indices
    int3 cellMin = arena->getCellIdx(aabbMin);
    int3 cellMax = arena->getCellIdx(aabbMax);

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

            bool overlap = vec3::lte(aabbMin, triMax) && vec3::gte(aabbMax, triMin);

            if (overlap)
            {
                // Atomic index for candidate pairs
                int idx = atomicAdd(&space->count, 1);
                space->pairs[idx] = {carIdx, triIdx};
            }
        }
    }
}

__device__ __forceinline__ void carArenaNarrowPhase(
    GameState* state,
    ArenaMesh* arena,
    Workspace* space,
    int pairIdx)
{
    // Cached read of pair indices
    auto [carIdx, triIdx] = __ldg(&space->pairs[pairIdx]);

    // Car state
    float4 carPos = __ldg(&state->cars.position[carIdx]);
    float4 carRot = __ldg(&state->cars.rotation[carIdx]);

    // Box center in world space (car position + rotated offset)
    float4 boxCenter = vec3::add(carPos, quat::toWorld(CAR_OFFSETS, carRot));

    // Get triangle vertices in world space
    auto [v0, v1, v2] = getTriVerts(arena, triIdx);

    // Transform triangle to box-local space (box centered at origin, axis-aligned)
    float4 t0 = quat::toLocal(vec3::sub(v0, boxCenter), carRot);
    float4 t1 = quat::toLocal(vec3::sub(v1, boxCenter), carRot);
    float4 t2 = quat::toLocal(vec3::sub(v2, boxCenter), carRot);

    // Triangle edges (in local space)
    float4 e0 = vec3::sub(t1, t0);
    float4 e1 = vec3::sub(t2, t1);
    float4 e2 = vec3::sub(t0, t2);

    // Test separation on axis; returns true if separated
    auto separated = [&](float4 axis) {
        float lenSq = vec3::dot(axis, axis);
        if (lenSq < 1e-8f) return false;

        float p0 = vec3::dot(t0, axis);
        float p1 = vec3::dot(t1, axis);
        float p2 = vec3::dot(t2, axis);
        float triMin = fminf(fminf(p0, p1), p2);
        float triMax = fmaxf(fmaxf(p0, p1), p2);

        float boxRadius = fabsf(axis.x) * CAR_HALF_EX.x
                        + fabsf(axis.y) * CAR_HALF_EX.y
                        + fabsf(axis.z) * CAR_HALF_EX.z;

        return triMin > boxRadius || triMax < -boxRadius;
    };

    // Box face normals (axis-aligned in local space)
    bool sepX = fminf(fminf(t0.x, t1.x), t2.x) > CAR_HALF_EX.x ||
                fmaxf(fmaxf(t0.x, t1.x), t2.x) < -CAR_HALF_EX.x;
    bool sepY = fminf(fminf(t0.y, t1.y), t2.y) > CAR_HALF_EX.y ||
                fmaxf(fmaxf(t0.y, t1.y), t2.y) < -CAR_HALF_EX.y;
    bool sepZ = fminf(fminf(t0.z, t1.z), t2.z) > CAR_HALF_EX.z ||
                fmaxf(fmaxf(t0.z, t1.z), t2.z) < -CAR_HALF_EX.z;

    // Triangle face normal
    bool sepTri = separated(vec3::cross(e0, e1));

    // Cross-product axes (box edges x triangle edges)
    float4 boxX = {1, 0, 0, 0};
    float4 boxY = {0, 1, 0, 0};
    float4 boxZ = {0, 0, 1, 0};

    bool sepXe0 = separated(vec3::cross(boxX, e0));
    bool sepXe1 = separated(vec3::cross(boxX, e1));
    bool sepXe2 = separated(vec3::cross(boxX, e2));
    bool sepYe0 = separated(vec3::cross(boxY, e0));
    bool sepYe1 = separated(vec3::cross(boxY, e1));
    bool sepYe2 = separated(vec3::cross(boxY, e2));
    bool sepZe0 = separated(vec3::cross(boxZ, e0));
    bool sepZe1 = separated(vec3::cross(boxZ, e1));
    bool sepZe2 = separated(vec3::cross(boxZ, e2));

    bool anySeparated = sepX | sepY | sepZ | sepTri |
                        sepXe0 | sepXe1 | sepXe2 |
                        sepYe0 | sepYe1 | sepYe2 |
                        sepZe0 | sepZe1 | sepZe2;

    if (!anySeparated)
    {
        // All 13 axes show overlap - collision detected
        // TODO: Compute contact point, normal, and penetration depth
        atomicAdd(&state->cars.numTris[carIdx], 1);
    }
}