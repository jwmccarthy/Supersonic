#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsSAT.hpp"
#include "CollisionsEdge.hpp"

// Compute offset along axis toward other car's center
// Used to find the edge closest to the other car
__device__ float4 axisOffset(float4 vecAB, float4 axis, int idx)
{
    return vec3::mult(axis, sign(vec3::dot(vecAB, axis)) * CAR_HALF_EX_ARR[idx]);
}

// Compute edge center from two perpendicular axis offsets
// dir = +1 for car A (toward B), -1 for car B (toward A)
__device__ float4 edgeCenter(float4 vecAB, const float4* axes, int i1, int i2, float dir)
{
    return vec3::add(
        vec3::mult(axisOffset(vecAB, axes[i1], i1), dir),
        vec3::mult(axisOffset(vecAB, axes[i2], i2), dir)
    );
}

// Get edge center points for both cars
// These are the midpoints of the colliding edges
__device__ EdgePoints getEdgeCenters(const SATContext& ctx, const EdgeAxes& ax)
{
    EdgePoints ep;
    ep.pA = edgeCenter(ctx.vecAB, WORLD_AXES, ax.a1, ax.a2, +1.0f);
    ep.pB = vec3::add(edgeCenter(ctx.vecAB, ctx.axB, ax.b1, ax.b2, -1.0f), ctx.vecAB);
    return ep;
}

// Find closest point on edge B to edge A
// Uses line-line closest point formula
__device__ float4 getEdgeContactPoint(const EdgePoints& ep, float4 dA, float4 dB)
{
    float4 r = vec3::sub(ep.pB, ep.pA);
    float  a = vec3::dot(r, dA);
    float  b = vec3::dot(r, dB);
    float  n = vec3::dot(dA, dB);
    float  d = 1.0f - n * n;

    // Parallel edges: use center point
    if (d <= 1e-6f)
    {
        return ep.pB;
    }
    return vec3::add(ep.pB, vec3::mult(dB, (n * a - b) / d));
}

// Generate edge-edge contact manifold
// Edge contacts always produce exactly one contact point
__device__ void generateEdgeEdgeManifold(SATContext& ctx, SATResult& res, ContactManifold& m)
{
    EdgeAxes   ax = getEdgeAxes(res.axisIdx);
    EdgePoints ep = getEdgeCenters(ctx, ax);

    m.count     = 1;
    m.points[0] = getEdgeContactPoint(ep, WORLD_AXES[ax.ai], ctx.axB[ax.bi]);
    m.depths[0] = res.depth;
    m.normal    = vec3::mult(res.bestAx, sign(vec3::dot(ctx.vecAB, res.bestAx)));
}
