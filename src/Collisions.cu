#include <cfloat>

#include "Collisions.hpp"
#include "CudaMath.hpp"
#include "RLConstants.hpp"

struct SATContext
{
    float4 vecAB;
    float4 axB[3];
};

struct SATResult
{
    float  maxSep  = -FLT_MAX;
    float4 bestAx  = {};
    bool   overlap = true;
};

// Project car extents onto axis
__device__ float projectRadius(float4 L, float4 axX, float4 axY, float4 axZ)
{
    return CAR_HALF_EX.x * fabsf(vec3::dot(L, axX)) +
           CAR_HALF_EX.y * fabsf(vec3::dot(L, axY)) +
           CAR_HALF_EX.z * fabsf(vec3::dot(L, axZ));
}

// Build SAT context from car transforms
__device__ SATContext buildSATContext(float4 posA, float4 rotA, float4 posB, float4 rotB)
{
    // Normalize quaternions and apply hitbox offsets
    rotA = quat::norm(rotA);
    rotB = quat::norm(rotB);
    posA = vec4::add(posA, quat::mult(CAR_OFFSETS, rotA));
    posB = vec4::add(posB, quat::mult(CAR_OFFSETS, rotB));

    // Transform into A's local space
    const float4 conjA = quat::conj(rotA);
    const float4 rotAB = quat::comp(conjA, rotB);

    SATContext ctx;
    ctx.vecAB = quat::mult(vec3::sub(posB, posA), conjA);
    ctx.axB[0] = quat::mult(WORLD_X, rotAB);
    ctx.axB[1] = quat::mult(WORLD_Y, rotAB);
    ctx.axB[2] = quat::mult(WORLD_Z, rotAB);
    return ctx;
}

// Test single SAT axis
__device__ void testAxis(float4 L, bool normalize, const SATContext& ctx, SATResult& res)
{
    // Normalize edge axes
    if (normalize)
    {
        float lenSq = vec3::dot(L, L);
        if (lenSq < 1e-6f) return;
        L = vec3::mult(L, rsqrtf(lenSq));
    }

    // Project center distance onto L
    float d = fabsf(vec3::dot(L, ctx.vecAB));
    float r = projectRadius(L, WORLD_X, WORLD_Y, WORLD_Z) +
              projectRadius(L, ctx.axB[0], ctx.axB[1], ctx.axB[2]);
    float s = d - r;

    if (s > res.maxSep)
    {
        res.maxSep = s;
        res.bestAx = L;
    }

    if (s > 0.0f) res.collides = false;
}

// SAT collision test between two cars
__device__ bool carCarCollision(float4 posA, float4 rotA, float4 posB, float4 rotB)
{
    // Build SAT context
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    
    SATResult res;

    // Test 6 face normals
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        testAxis(WORLD_AXES[i], false, ctx, res);
        testAxis(ctx.axB[i], false, ctx, res);
    }

    // Test 9 edge cross products
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
        testAxis(vec3::cross(WORLD_AXES[i], ctx.axB[j]), true, ctx, res);
    }

    return res.collides;
}
