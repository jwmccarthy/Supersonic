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

            // Find tri canonical cell for dedup per-car
            int4 c = __ldg(&arena->triCellMin[triIdx]);
            int3 canon = vec3::max(cellMin, {c.x, c.y, c.z});

            // Triangle AABB
            float4 triMin = __ldg(&arena->aabbMin[triIdx]);
            float4 triMax = __ldg(&arena->aabbMax[triIdx]);

            // Checks prior to emission
            bool isCanon = vec3::eq(canon, {x, y, z});
            bool overlap = vec3::lte(aabbMin, triMax) && vec3::gte(aabbMax, triMin);

            if (isCanon && overlap)
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

    if (separated) return;

    // Contact points in box-local space (max 4)
    float4 contacts[4];
    int numContacts = 0;

    // Determine axis type: 0-5 = box faces (±X,±Y,±Z), 6 = tri face, 7+ = edge-edge
    // minAxis already tells us the direction, check which case we're in
    float ax = fabsf(minAxis.x), ay = fabsf(minAxis.y), az = fabsf(minAxis.z);

    // Box face contact: clip triangle against box face
    if (ax > 0.9f || ay > 0.9f || az > 0.9f)
    {
        // Reference face is the box face aligned with minAxis
        // Clip triangle edges against the 4 side planes of this face

        // Determine which axis is the face normal and the two perpendicular axes
        int faceAxis = (ax > 0.9f) ? 0 : (ay > 0.9f) ? 1 : 2;
        float faceSign = (faceAxis == 0) ? minAxis.x : (faceAxis == 1) ? minAxis.y : minAxis.z;
        float faceD = (faceAxis == 0) ? CAR_HALF_EX.x : (faceAxis == 1) ? CAR_HALF_EX.y : CAR_HALF_EX.z;

        // The two perpendicular half-extents for clipping
        float clipEx[2], clipMin[2], clipMax[2];
        if (faceAxis == 0) { // X face, clip against Y and Z
            clipEx[0] = CAR_HALF_EX.y; clipEx[1] = CAR_HALF_EX.z;
        } else if (faceAxis == 1) { // Y face, clip against X and Z
            clipEx[0] = CAR_HALF_EX.x; clipEx[1] = CAR_HALF_EX.z;
        } else { // Z face, clip against X and Y
            clipEx[0] = CAR_HALF_EX.x; clipEx[1] = CAR_HALF_EX.y;
        }
        clipMin[0] = -clipEx[0]; clipMax[0] = clipEx[0];
        clipMin[1] = -clipEx[1]; clipMax[1] = clipEx[1];

        // Sutherland-Hodgman style clipping of triangle against 4 planes
        // Start with triangle vertices
        float4 poly[6] = {t0, t1, t2};
        int polyN = 3;

        // Clip against each of 4 side planes
        for (int plane = 0; plane < 4 && polyN > 0; ++plane)
        {
            float4 out[6];
            int outN = 0;

            int clipIdx = plane / 2; // 0 or 1
            float boundary = (plane & 1) ? clipMax[clipIdx] : clipMin[clipIdx];
            float sign = (plane & 1) ? 1.0f : -1.0f;

            for (int i = 0; i < polyN && outN < 6; ++i)
            {
                float4 curr = poly[i];
                float4 next = poly[(i + 1) % polyN];

                // Get coordinate based on face axis and clip index
                float currC, nextC;
                if (faceAxis == 0) { currC = (clipIdx == 0) ? curr.y : curr.z; nextC = (clipIdx == 0) ? next.y : next.z; }
                else if (faceAxis == 1) { currC = (clipIdx == 0) ? curr.x : curr.z; nextC = (clipIdx == 0) ? next.x : next.z; }
                else { currC = (clipIdx == 0) ? curr.x : curr.y; nextC = (clipIdx == 0) ? next.x : next.y; }

                bool currIn = (sign * currC) <= (sign * boundary);
                bool nextIn = (sign * nextC) <= (sign * boundary);

                if (currIn && nextIn) {
                    out[outN++] = next;
                } else if (currIn && !nextIn) {
                    float t = (boundary - currC) / (nextC - currC);
                    out[outN++] = vec3::add(curr, vec3::mult(vec3::sub(next, curr), t));
                } else if (!currIn && nextIn) {
                    float t = (boundary - currC) / (nextC - currC);
                    out[outN++] = vec3::add(curr, vec3::mult(vec3::sub(next, curr), t));
                    out[outN++] = next;
                }
            }
            polyN = outN;
            for (int i = 0; i < polyN; ++i) poly[i] = out[i];
        }

        // Project clipped polygon onto reference face and keep points behind it
        float refPlane = faceSign * faceD;
        for (int i = 0; i < polyN && numContacts < 4; ++i)
        {
            float depth;
            if (faceAxis == 0) depth = faceSign * (refPlane - poly[i].x * faceSign);
            else if (faceAxis == 1) depth = faceSign * (refPlane - poly[i].y * faceSign);
            else depth = faceSign * (refPlane - poly[i].z * faceSign);

            if (depth >= 0)
            {
                // Project point onto face
                float4 cp = poly[i];
                if (faceAxis == 0) cp.x = faceSign * faceD;
                else if (faceAxis == 1) cp.y = faceSign * faceD;
                else cp.z = faceSign * faceD;
                contacts[numContacts++] = cp;
            }
        }
    }
    else
    {
        // Triangle face or edge-edge contact - use triangle centroid as single contact
        float4 centroid = vec3::mult(vec3::add(vec3::add(t0, t1), t2), 1.0f/3.0f);
        contacts[0] = centroid;
        numContacts = 1;
    }

    // Transform contacts to world space
    float4 worldNormal = quat::toWorld(minAxis, carRot);
    float4 avgContact = {0, 0, 0, 0};
    for (int i = 0; i < numContacts; ++i)
    {
        float4 worldContact = vec3::add(boxCenter, quat::toWorld(contacts[i], carRot));
        avgContact = vec3::add(avgContact, worldContact);
    }
    if (numContacts > 0) avgContact = vec3::mult(avgContact, 1.0f / numContacts);

    // Force compiler to keep all computations (for profiling)
    float forceKeep = minPen + worldNormal.x + avgContact.x + avgContact.y + avgContact.z;
    atomicAdd(&state->cars.numTris[carIdx], numContacts + (int)(forceKeep * 0.001f));
}