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

    // Track minimum penetration axis
    float minPen = 1e9f;
    float4 minAxis = {0, 0, 0, 0};
    bool separated = false;

    // === Box face normals (X, Y, Z) ===
    // X-axis
    float triMinX = fminf(fminf(t0.x, t1.x), t2.x);
    float triMaxX = fmaxf(fmaxf(t0.x, t1.x), t2.x);
    float penPosX = CAR_HALF_EX.x - triMinX;  // Push in +X
    float penNegX = triMaxX + CAR_HALF_EX.x;  // Push in -X
    separated |= (penPosX < 0) | (penNegX < 0);
    if (penPosX < penNegX && penPosX < minPen) { minPen = penPosX; minAxis = {1, 0, 0, 0}; }
    if (penNegX < penPosX && penNegX < minPen) { minPen = penNegX; minAxis = {-1, 0, 0, 0}; }

    // Y-axis
    float triMinY = fminf(fminf(t0.y, t1.y), t2.y);
    float triMaxY = fmaxf(fmaxf(t0.y, t1.y), t2.y);
    float penPosY = CAR_HALF_EX.y - triMinY;
    float penNegY = triMaxY + CAR_HALF_EX.y;
    separated |= (penPosY < 0) | (penNegY < 0);
    if (penPosY < penNegY && penPosY < minPen) { minPen = penPosY; minAxis = {0, 1, 0, 0}; }
    if (penNegY < penPosY && penNegY < minPen) { minPen = penNegY; minAxis = {0, -1, 0, 0}; }

    // Z-axis
    float triMinZ = fminf(fminf(t0.z, t1.z), t2.z);
    float triMaxZ = fmaxf(fmaxf(t0.z, t1.z), t2.z);
    float penPosZ = CAR_HALF_EX.z - triMinZ;
    float penNegZ = triMaxZ + CAR_HALF_EX.z;
    separated |= (penPosZ < 0) | (penNegZ < 0);
    if (penPosZ < penNegZ && penPosZ < minPen) { minPen = penPosZ; minAxis = {0, 0, 1, 0}; }
    if (penNegZ < penPosZ && penNegZ < minPen) { minPen = penNegZ; minAxis = {0, 0, -1, 0}; }

    // === Triangle face normal ===
    float4 triNorm = vec3::cross(e0, e1);
    float triNormLen = sqrtf(vec3::dot(triNorm, triNorm));
    if (triNormLen > 1e-8f)
    {
        triNorm = vec3::mult(triNorm, 1.0f / triNormLen);
        float p0 = vec3::dot(t0, triNorm);
        float p1 = vec3::dot(t1, triNorm);
        float p2 = vec3::dot(t2, triNorm);
        float triMin = fminf(fminf(p0, p1), p2);
        float triMax = fmaxf(fmaxf(p0, p1), p2);
        float boxRadius = fabsf(triNorm.x) * CAR_HALF_EX.x
                        + fabsf(triNorm.y) * CAR_HALF_EX.y
                        + fabsf(triNorm.z) * CAR_HALF_EX.z;
        float penPos = boxRadius - triMin;
        float penNeg = triMax + boxRadius;
        separated |= (penPos < 0) | (penNeg < 0);
        if (penPos < penNeg && penPos < minPen) { minPen = penPos; minAxis = triNorm; }
        if (penNeg < penPos && penNeg < minPen) { minPen = penNeg; minAxis = vec3::mult(triNorm, -1.0f); }
    }

    // === Cross-product axes (9 total) ===
    float4 axes[9] = {
        {0, -e0.z, e0.y, 0}, {0, -e1.z, e1.y, 0}, {0, -e2.z, e2.y, 0},  // X × edges
        {e0.z, 0, -e0.x, 0}, {e1.z, 0, -e1.x, 0}, {e2.z, 0, -e2.x, 0},  // Y × edges
        {-e0.y, e0.x, 0, 0}, {-e1.y, e1.x, 0, 0}, {-e2.y, e2.x, 0, 0}   // Z × edges
    };

    #pragma unroll
    for (int i = 0; i < 9; ++i)
    {
        float4 axis = axes[i];
        float lenSq = vec3::dot(axis, axis);
        if (lenSq < 1e-8f) continue;

        float invLen = rsqrtf(lenSq);
        axis = vec3::mult(axis, invLen);

        float p0 = vec3::dot(t0, axis);
        float p1 = vec3::dot(t1, axis);
        float p2 = vec3::dot(t2, axis);
        float triMin = fminf(fminf(p0, p1), p2);
        float triMax = fmaxf(fmaxf(p0, p1), p2);
        float boxRadius = fabsf(axis.x) * CAR_HALF_EX.x
                        + fabsf(axis.y) * CAR_HALF_EX.y
                        + fabsf(axis.z) * CAR_HALF_EX.z;

        float penPos = boxRadius - triMin;
        float penNeg = triMax + boxRadius;
        separated |= (penPos < 0) | (penNeg < 0);
        if (penPos < penNeg && penPos < minPen) { minPen = penPos; minAxis = axis; }
        if (penNeg < penPos && penNeg < minPen) { minPen = penNeg; minAxis = vec3::mult(axis, -1.0f); }
    }

    if (!separated)
    {
        // minAxis is contact normal (in car-local space), minPen is penetration depth
        // Transform normal to world space
        float4 worldNormal = quat::toWorld(minAxis, carRot);

        // TODO: Use worldNormal and minPen for collision response
        atomicAdd(&state->cars.numTris[carIdx], 1);
    }
}