#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsSAT.hpp"
#include "CollisionsEdge.hpp"

// Offset along one axis toward the other car's center
__device__ float4 axisOffset(float4 vecAB, float4 axis, int idx)
{
    float s = sign(vec3::dot(vecAB, axis));
    return vec3::mult(axis, s * CAR_HALF_EX_ARR[idx]);
}

// Edge center via sum of offsets along the two perpendicular axes
__device__ float4 edgeCenter(float4 vecAB, const float4* axes, int i1, int i2, float dir = 1.0f)
{
    return vec3::add(
        vec3::mult(axisOffset(vecAB, axes[i1], i1), dir),
        vec3::mult(axisOffset(vecAB, axes[i2], i2), dir)
    );
}

// Get edge center points for both cars
__device__ EdgePoints getEdgeCenters(const SATContext& ctx, const EdgeAxes& ax)
{
    EdgePoints ep;
    ep.pA = edgeCenter(ctx.vecAB, WORLD_AXES, ax.a1, ax.a2);
    ep.pB = edgeCenter(ctx.vecAB, ctx.axB, ax.b1, ax.b2, -1.0f);
    ep.pB = vec3::add(ep.pB, ctx.vecAB);
    return ep;
}

// Compute closest point on edge B (contact point)
__device__ float4 getEdgeContactPoint(const EdgePoints& ec, float4 dirA, float4 dirB)
{
    // Vector between center points
    float4 r = vec3::sub(ec.pB, ec.pA);

    // Closest point parameter for edge B
    float a = vec3::dot(r, dirA);
    float b = vec3::dot(r, dirB);
    float n = vec3::dot(dirA, dirB);
    float d = 1.0f - n * n;

    // Parallel edges - use center point
    if (d <= 1e-6f) return ec.pB;

    // Compute s parameter and return closest point on B
    float s = (n * a - b) / d;
    return vec3::add(ec.pB, vec3::mult(dirB, s));
}

// Edge-edge collision manifold generation
__device__ void generateEdgeEdgeManifold(SATContext& ctx, SATResult& res)
{
    // Get edge axis indices
    EdgeAxes ax = getEdgeAxes(res.axisIdx);

    // Get edge center points
    EdgePoints ec = getEdgeCenters(ctx, ax);

    // Compute contact point (closest point on edge B)
    float4 contact = getEdgeContactPoint(ec, WORLD_AXES[ax.ai], ctx.axB[ax.bi]);
}
