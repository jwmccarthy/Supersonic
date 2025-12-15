#include "CollisionsSAT.hpp"
#include "CudaMath.hpp"
#include "RLConstants.hpp"

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

    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        ctx.axB[i] = quat::mult(WORLD_AXES[i], rotAB);
    }

    return ctx;
}

// Test single SAT axis
__device__ void testAxis(
    float4 L, int axis, 
    const SATContext& ctx, 
    SATResult& res, 
    bool isEdgeAxis
)
{
    // Normalize edge axes (face axes already unit length)
    if (isEdgeAxis)
    {
        float lenSq = vec3::dot(L, L);
        if (lenSq < 1e-6f) return;
        L = vec3::mult(L, rsqrtf(lenSq));
    }

    // Sign L according to direction
    float d = vec3::dot(L, ctx.vecAB);
    L = vec3::mult(L, sign(d));

    // Project car radii onto L
    float r = projectRadius(L, WORLD_X, WORLD_Y, WORLD_Z) +
              projectRadius(L, ctx.axB[0], ctx.axB[1], ctx.axB[2]);

    // Compute penetration depth (positive = overlap, negative = separation)
    // Adjust edge-edge separation to favor face contacts
    float depth = r - d;
    float fudge = isEdgeAxis ? (depth * SAT_FUDGE) : depth;

    // Update best axis if this has shallower penetration
    if (fudge < res.depth)
    {
        res.depth = depth;
        res.bestAx = L;
        res.axisIdx = axis;
    }

    // Negative depth means separation (no overlap)
    if (depth < 0.0f) res.overlap = false;
}

// SAT collision test between two cars
__device__ SATResult carCarSATTest(SATContext& ctx)
{   
    SATResult res;

    // Test 6 face normals
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        testAxis(WORLD_AXES[i], i, ctx, res, false);
        testAxis(ctx.axB[i], i + 3, ctx, res, false);
    }

    // Test 9 edge-edge cross products
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            float4 L = vec3::cross(WORLD_AXES[i], ctx.axB[j]);
            testAxis(L, 6 + i * 3 + j, ctx, res, true);
        }
    }

    return res;
}

// Get perpendicular axis indices for edge-edge contact
// axisIdx 6-14 encodes which edges collided: (A's edge dir) * 3 + (B's edge dir)
__device__ EdgeAxes getEdgeAxes(int axisIdx)
{
    // Get indices of edge faces
    int i = (axisIdx - 6) / 3;
    int j = (axisIdx - 6) % 3;

    // Incident and perpendicular axes
    return { 
        i, (i + 1) % 3, (i + 2) % 3, 
        j, (j + 1) % 3, (j + 2) % 3 
    };
}

