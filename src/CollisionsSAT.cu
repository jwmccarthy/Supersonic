#include "CollisionsSAT.hpp"
#include "CudaMath.hpp"
#include "RLConstants.hpp"

// Project box half-extents onto axis L
// Returns sum of |L . axis| * extent for each axis
__device__ float projectRadius(float4 L, float4 axX, float4 axY, float4 axZ)
{
    return CAR_HALF_EX.x * fabsf(vec3::dot(L, axX)) +
           CAR_HALF_EX.y * fabsf(vec3::dot(L, axY)) +
           CAR_HALF_EX.z * fabsf(vec3::dot(L, axZ));
}

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

// Test single separating axis
// Updates res if this axis has shallower penetration
__device__ void testAxis(float4 L, int axis, const SATContext& ctx, SATResult& res, bool isEdge)
{
    // Edge axes need normalization (cross products aren't unit length)
    if (isEdge)
    {
        float lenSq = vec3::dot(L, L);
        if (lenSq < 1e-6f)
        {
            return;  // Parallel edges, skip degenerate axis
        }
        L = vec3::mult(L, rsqrtf(lenSq));
    }

    // Orient L to point from A toward B
    float d = vec3::dot(L, ctx.vecAB);
    L = vec3::mult(L, sign(d));

    // Project both car radii onto L
    float r = projectRadius(L, WORLD_X, WORLD_Y, WORLD_Z) +
              projectRadius(L, ctx.axB[0], ctx.axB[1], ctx.axB[2]);

    // Penetration depth: positive = overlap, negative = separation
    float depth = r - d;

    // Edge axes get fudge factor to prefer face contacts when similar depth
    float fudge = isEdge ? (depth * SAT_FUDGE) : depth;

    // Track shallowest penetration (minimum separation axis)
    if (fudge < res.depth)
    {
        res.depth   = depth;
        res.bestAx  = L;
        res.axisIdx = axis;
    }

    // Any negative depth means boxes are separated
    if (depth < 0.0f)
    {
        res.overlap = false;
    }
}

// Full SAT test: 6 face normals + 9 edge cross products
__device__ SATResult carCarSATTest(SATContext& ctx)
{
    SATResult res;

    // Test face normals (3 from A + 3 from B)
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        testAxis(WORLD_AXES[i], i,     ctx, res, false);
        testAxis(ctx.axB[i],    i + 3, ctx, res, false);
    }

    // Test edge-edge cross products (3x3 = 9 axes)
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            testAxis(vec3::cross(WORLD_AXES[i], ctx.axB[j]), 6 + i * 3 + j, ctx, res, true);
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
