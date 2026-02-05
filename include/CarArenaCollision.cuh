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

// Broad phase: compute group bounds and count triangles
__device__ __forceinline__ void carArenaBroadPhase(GameState* state, ArenaMesh* arena, Workspace* space, int carIdx)
{
    float4 pos = __ldg(&state->cars.position[carIdx]);
    float4 rot = __ldg(&state->cars.rotation[carIdx]);

    auto [aabbMin, aabbMax] = getCarAABB(arena, pos, rot);

    int3 cellMin = arena->getCellIdx(aabbMin);
    int3 groupIdx = arena->getGroupIdx(cellMin);
    int groupFlat = arena->flatGroupIdx(groupIdx.x, groupIdx.y, groupIdx.z);

    // Count total triangles in the overlapping group
    int triCount = __ldg(&arena->triPre[groupFlat + 1]) - __ldg(&arena->triPre[groupFlat]);

    // Store for narrow phase
    space->numTri[carIdx] = triCount;
    space->groupIdx[carIdx] = make_int4(groupIdx.x, groupIdx.y, groupIdx.z, 0);
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

    int4 groupIdx = space->groupIdx[carIdx];
    int groupFlat = arena->flatGroupIdx(groupIdx.x, groupIdx.y, groupIdx.z);
    int triBeg = __ldg(&arena->triPre[groupFlat]);
    int triEnd = __ldg(&arena->triPre[groupFlat + 1]);

    if (localTriIdx < triEnd - triBeg)
    {
        int t = triBeg + localTriIdx;
        int triIdx = __ldg(&arena->triIdx[t]);

        float4 triMin = __ldg(&arena->aabbMin[triIdx]);
        float4 triMax = __ldg(&arena->aabbMax[triIdx]);

        bool hit = (
            minX <= triMax.x && maxX >= triMin.x &&
            minY <= triMax.y && maxY >= triMin.y &&
            minZ <= triMax.z && maxZ >= triMin.z
        );

        // if (hit)
        // {
            // SAT: OBB vs Triangle (13 axes)
            // Load triangle vertices via index buffer
            int4 tri = arena->tris[triIdx];
            float4 v0 = arena->verts[tri.x];
            float4 v1 = arena->verts[tri.y];
            float4 v2 = arena->verts[tri.z];

            // Triangle edges and normal
            float4 e0 = vec3::sub(v1, v0);
            float4 e1 = vec3::sub(v2, v1);
            float4 e2 = vec3::sub(v0, v2);
            float4 triN = vec3::cross(e0, e1);

            // Box axes in world space
            float4 bX = quat::toWorld(WORLD_X, rot);
            float4 bY = quat::toWorld(WORLD_Y, rot);
            float4 bZ = quat::toWorld(WORLD_Z, rot);

            // Box center
            float4 center = vec3::add(pos, quat::toWorld(CAR_OFFSETS, rot));

            // Test all 13 potential separating axes
            bool separated = false;

            // Helper: test one axis
            auto testAxis = [&](float4 axis) {
                float len2 = vec3::dot(axis, axis);
                if (len2 < 1e-8f) return;

                // Box projection radius
                float boxR = fabsf(vec3::dot(bX, axis)) * CAR_HALF_EX.x +
                             fabsf(vec3::dot(bY, axis)) * CAR_HALF_EX.y +
                             fabsf(vec3::dot(bZ, axis)) * CAR_HALF_EX.z;

                // Triangle vertex projections relative to box center
                float d0 = vec3::dot(vec3::sub(v0, center), axis);
                float d1 = vec3::dot(vec3::sub(v1, center), axis);
                float d2 = vec3::dot(vec3::sub(v2, center), axis);

                float triMin = fminf(fminf(d0, d1), d2);
                float triMax = fmaxf(fmaxf(d0, d1), d2);

                if (triMin > boxR || triMax < -boxR) separated = true;
            };

            // 3 box face normals
            testAxis(bX);
            testAxis(bY);
            testAxis(bZ);

            // 1 triangle normal
            testAxis(triN);

            // 9 edge cross products
            testAxis(vec3::cross(bX, e0));
            testAxis(vec3::cross(bX, e1));
            testAxis(vec3::cross(bX, e2));
            testAxis(vec3::cross(bY, e0));
            testAxis(vec3::cross(bY, e1));
            testAxis(vec3::cross(bY, e2));
            testAxis(vec3::cross(bZ, e0));
            testAxis(vec3::cross(bZ, e1));
            testAxis(vec3::cross(bZ, e2));

            if (!separated)
            {
                atomicAdd(space->numHit, 1);
            }
        // }
    }
}
