#include "CollisionsSAT.hpp"

// Transform B into A's local space for SAT
__device__ SATContext buildSATContext(float4 posA, float4 rotA, float4 posB, float4 rotB)
{
    // Normalize and apply hitbox offsets
    rotA = quat::norm(rotA);
    rotB = quat::norm(rotB);
    posA = vec4::add(posA, quat::mult(CAR_OFFSETS, rotA));
    posB = vec4::add(posB, quat::mult(CAR_OFFSETS, rotB));

    // Relative rotation: A^-1 * B
    float4 conjA = quat::conj(rotA);
    float4 rotAB = quat::comp(conjA, rotB);

    // B's axes in A's local space
    SATContext ctx;
    ctx.vecAB = quat::mult(vec3::sub(posB, posA), conjA);
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        ctx.axB[i] = quat::mult(WORLD_AXES[i], rotAB);
    }
    return ctx;
}

// SAT test: 6 face + 9 edge axes
__device__ SATResult carCarSATTest(SATContext& ctx)
{
    SATResult res;

    // Precompute |B axes| for projections
    float3 absB0 = make_float3(fabsf(ctx.axB[0].x), fabsf(ctx.axB[0].y), fabsf(ctx.axB[0].z));
    float3 absB1 = make_float3(fabsf(ctx.axB[1].x), fabsf(ctx.axB[1].y), fabsf(ctx.axB[1].z));
    float3 absB2 = make_float3(fabsf(ctx.axB[2].x), fabsf(ctx.axB[2].y), fabsf(ctx.axB[2].z));

    // Face axes 0-2 (A's local axes)
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float d = getComp(ctx.vecAB, i);
        float rA = CAR_HALF_EX_ARR[i];
        float rB = projectB(absB0, absB1, absB2, i);
        res.overlap &= testAxis(res, WORLD_AXES[i], d, rA, rB, i);
    }

    // Face axes 3-5 (B's local axes)
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float4 L = ctx.axB[i];
        float d = vec3::dot(L, ctx.vecAB);
        float rA = projectOBB(L);
        float rB = CAR_HALF_EX_ARR[i];
        res.overlap &= testAxis(res, L, d, rA, rB, 3 + i);
    }

    if (!res.overlap) return res;

    // Edge axes 6-14 (cross products)
    #pragma unroll
    for (int ai = 0; ai < 3; ai++)
    {
        #pragma unroll
        for (int bi = 0; bi < 3; bi++)
        {
            float4 L = vec3::cross(WORLD_AXES[ai], ctx.axB[bi]);
            float lenSq = vec3::dot(L, L);
            if (lenSq < 1e-6f) continue;  // skip parallel

            L = vec3::mult(L, rsqrtf(lenSq));
            float d = vec3::dot(L, ctx.vecAB);
            float rA = projectOBB(L);
            float rB = CAR_HALF_EX.x * fabsf(vec3::dot(L, ctx.axB[0])) +
                       CAR_HALF_EX.y * fabsf(vec3::dot(L, ctx.axB[1])) +
                       CAR_HALF_EX.z * fabsf(vec3::dot(L, ctx.axB[2]));

            res.overlap &= testAxis(res, L, d, rA, rB, 6 + ai * 3 + bi, SAT_FUDGE);
        }
    }

    return res;
}

// Decode edge axis index to perpendicular axes
__device__ EdgeAxes getEdgeAxes(int axisIdx)
{
    int i = (axisIdx - 6) / 3;
    int j = (axisIdx - 6) % 3;
    return {
        i, (i + 1) % 3, (i + 2) % 3,
        j, (j + 1) % 3, (j + 2) % 3
    };
}
