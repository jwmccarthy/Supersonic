#include "CollisionsSAT.hpp"
#include "CudaMath.hpp"
#include "RLConstants.hpp"

// Build SAT context: transform everything into car A's local space
// This simplifies axis testing since A's axes become world axes
__device__ SATContext buildSATContext(float4 posA, float4 rotA, float4 posB, float4 rotB)
{
    // Normalize quaternions and apply hitbox offsets
    rotA = quat::norm(rotA);
    rotB = quat::norm(rotB);
    posA = vec4::add(posA, quat::mult(CAR_OFFSETS, rotA));
    posB = vec4::add(posB, quat::mult(CAR_OFFSETS, rotB));

    // Compute relative rotation: A^-1 * B
    float4 conjA = quat::conj(rotA);
    float4 rotAB = quat::comp(conjA, rotB);

    // Build context with B's axes in A's local space
    SATContext ctx;
    ctx.vecAB = quat::mult(vec3::sub(posB, posA), conjA);
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        ctx.axB[i] = quat::mult(WORLD_AXES[i], rotAB);
    }
    return ctx;
}

// SAT test between two car OBBs (6 face + 9 edge axes)
__device__ SATResult carCarSATTest(SATContext& ctx)
{
    SATResult res;

    // Precompute absolute values of B's axes for faster projections
    float3 absB0 = make_float3(fabsf(ctx.axB[0].x), fabsf(ctx.axB[0].y), fabsf(ctx.axB[0].z));
    float3 absB1 = make_float3(fabsf(ctx.axB[1].x), fabsf(ctx.axB[1].y), fabsf(ctx.axB[1].z));
    float3 absB2 = make_float3(fabsf(ctx.axB[2].x), fabsf(ctx.axB[2].y), fabsf(ctx.axB[2].z));

    // ============ FACE AXES (0-5) ============
    // A's face axes (WORLD_X, WORLD_Y, WORLD_Z) - simplified projection

    // Axis 0: WORLD_X
    {
        float d = fabsf(ctx.vecAB.x);
        float rA = CAR_HALF_EX.x;
        float rB = CAR_HALF_EX.x * absB0.x + CAR_HALF_EX.y * absB1.x + CAR_HALF_EX.z * absB2.x;
        float depth = rA + rB - d;
        if (depth < 0.0f) res.overlap = false;
        if (depth < res.depth) {
            res.depth = depth;
            res.bestAx = (ctx.vecAB.x >= 0) ? WORLD_X : make_float4(-1, 0, 0, 0);
            res.axisIdx = 0;
        }
    }

    // Axis 1: WORLD_Y
    {
        float d = fabsf(ctx.vecAB.y);
        float rA = CAR_HALF_EX.y;
        float rB = CAR_HALF_EX.x * absB0.y + CAR_HALF_EX.y * absB1.y + CAR_HALF_EX.z * absB2.y;
        float depth = rA + rB - d;
        if (depth < 0.0f) res.overlap = false;
        if (depth < res.depth) {
            res.depth = depth;
            res.bestAx = (ctx.vecAB.y >= 0) ? WORLD_Y : make_float4(0, -1, 0, 0);
            res.axisIdx = 1;
        }
    }

    // Axis 2: WORLD_Z
    {
        float d = fabsf(ctx.vecAB.z);
        float rA = CAR_HALF_EX.z;
        float rB = CAR_HALF_EX.x * absB0.z + CAR_HALF_EX.y * absB1.z + CAR_HALF_EX.z * absB2.z;
        float depth = rA + rB - d;
        if (depth < 0.0f) res.overlap = false;
        if (depth < res.depth) {
            res.depth = depth;
            res.bestAx = (ctx.vecAB.z >= 0) ? WORLD_Z : make_float4(0, 0, -1, 0);
            res.axisIdx = 2;
        }
    }

    // B's face axes (3-5) - need full projection
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        float4 L = ctx.axB[i];
        float d = vec3::dot(L, ctx.vecAB);
        float s = sign(d);
        d = fabsf(d);

        // Project A onto L (A's axes are world-aligned)
        float rA = CAR_HALF_EX.x * fabsf(L.x) + CAR_HALF_EX.y * fabsf(L.y) + CAR_HALF_EX.z * fabsf(L.z);
        // Project B onto L (L is one of B's axes, so projection is just the half-extent)
        float rB = CAR_HALF_EX_ARR[i];

        float depth = rA + rB - d;
        if (depth < 0.0f) res.overlap = false;
        if (depth < res.depth) {
            res.depth = depth;
            res.bestAx = vec3::mult(L, s);
            res.axisIdx = 3 + i;
        }
    }

    // Early exit if separated on face axes
    if (!res.overlap) return res;

    // ============ EDGE AXES (6-14) ============
    // Cross products of A's edges (world axes) with B's edges

    #pragma unroll
    for (int ai = 0; ai < 3; ai++)
    {
        #pragma unroll
        for (int bi = 0; bi < 3; bi++)
        {
            float4 L = vec3::cross(WORLD_AXES[ai], ctx.axB[bi]);
            float lenSq = vec3::dot(L, L);

            // Skip near-parallel edges
            if (lenSq < 1e-6f) continue;

            L = vec3::mult(L, rsqrtf(lenSq));
            float d = vec3::dot(L, ctx.vecAB);
            float s = sign(d);
            d = fabsf(d);

            float rA = CAR_HALF_EX.x * fabsf(L.x) + CAR_HALF_EX.y * fabsf(L.y) + CAR_HALF_EX.z * fabsf(L.z);
            float rB = CAR_HALF_EX.x * fabsf(vec3::dot(L, ctx.axB[0])) +
                       CAR_HALF_EX.y * fabsf(vec3::dot(L, ctx.axB[1])) +
                       CAR_HALF_EX.z * fabsf(vec3::dot(L, ctx.axB[2]));

            float depth = rA + rB - d;
            if (depth < 0.0f) res.overlap = false;

            // Apply fudge factor for edge axes (prefer face contacts)
            float fudge = depth * SAT_FUDGE;
            if (fudge < res.depth) {
                res.depth = depth;
                res.bestAx = vec3::mult(L, s);
                res.axisIdx = 6 + ai * 3 + bi;
            }
        }
    }

    return res;
}

// Decode edge axis index into perpendicular axis indices
// axisIdx 6-14 encodes: (A's edge dir) * 3 + (B's edge dir)
__device__ EdgeAxes getEdgeAxes(int axisIdx)
{
    int i = (axisIdx - 6) / 3;  // A's edge direction
    int j = (axisIdx - 6) % 3;  // B's edge direction
    return {
        i, (i + 1) % 3, (i + 2) % 3,  // A: incident, perp1, perp2
        j, (j + 1) % 3, (j + 2) % 3   // B: incident, perp1, perp2
    };
}
