#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsFace.hpp"
#include "CollisionsSAT.hpp"

// Blend between two axes based on weight (branchless select)
__device__ float4 blendAxes(float4 axA, float4 axB, float b)
{
    return vec3::add(vec3::mult(axA, 1.0f - b), vec3::mult(axB, b));
}

// Find incident face axis (most parallel to reference normal)
__device__ int findIncidentAxis(float4 dir, const float4* axes, float& bestDot)
{
    int bestIdx = 0;
    float maxDot = 0.0f;
    bestDot = 0.0f;

    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        float d = vec3::dot(dir, axes[i]);
        float a = fabsf(d);
        if (a > maxDot)
        {
            maxDot = a;
            bestIdx = i;
            bestDot = d;
        }
    }

    return bestIdx;
}

// Fill 4 face vertices from center and two tangent offsets
__device__ void setFaceVertices(float4 verts[4], float4 center, float4 off1, float4 off2)
{
    verts[0] = vec3::add(center, vec3::add(off1, off2));  // +t1 +t2
    verts[1] = vec3::add(center, vec3::sub(off1, off2));  // +t1 -t2
    verts[2] = vec3::sub(center, vec3::add(off1, off2));  // -t1 -t2
    verts[3] = vec3::sub(center, vec3::sub(off1, off2));  // -t1 +t2
}

// Build reference face from SAT result
__device__ void getReferenceFace(
    const SATContext& ctx, 
    const SATResult& res, 
    ReferenceFace& ref
)
{
    // Build axis indices around reference axis
    int i = res.axisIdx % 3;
    int j = (i + 1) % 3;
    int k = (i + 2) % 3;
    float b = (float)(res.axisIdx >= 3);

    // Get reference face normal (blend between A and B's axis)
    ref.normal = blendAxes(WORLD_AXES[i], ctx.axB[i], b);

    // Point reference normal towards incident car
    float d = vec3::dot(ctx.vecAB, ref.normal);
    float s = sign(d);
    float w = s * (1.0f - b) - s * b;
    ref.normal = vec3::mult(ref.normal, w);

    // Get tangent axes for reference face
    ref.ortho1 = blendAxes(WORLD_AXES[j], ctx.axB[j], b);
    ref.ortho2 = blendAxes(WORLD_AXES[k], ctx.axB[k], b);

    // Get candidate reference face centers
    float4 cA = vec3::mult(WORLD_AXES[i], w * CAR_HALF_EX_ARR[i]);
    float4 cB = vec3::add(ctx.vecAB, vec3::mult(ctx.axB[i], w * CAR_HALF_EX_ARR[i]));

    // Get reference face center & extents
    ref.center = blendAxes(cA, cB, b);
    ref.halfEx = make_float2(CAR_HALF_EX_ARR[j], CAR_HALF_EX_ARR[k]);
}

// Build incident face vertices from reference face
__device__ void getIncidentFace(
    const SATContext& ctx,
    const SATResult& res,
    const ReferenceFace& ref,
    IncidentFace& inc
)
{
    // Determine which box is incident (opposite of reference)
    float b = (float)(res.axisIdx >= 3);

    // Get incident car axes and origin
    const float4* incAxes = (b < 0.5f) ? ctx.axB : WORLD_AXES;
    float4 incOrig = vec3::mult(ctx.vecAB, 1.0f - b);

    // Find most parallel face (axis with largest |dot|)
    float bestDot;
    int bestIdx = findIncidentAxis(ref.normal, incAxes, bestDot);

    // Get face indices tangent to incident axis
    int t1 = (bestIdx + 1) % 3;
    int t2 = (bestIdx + 2) % 3;

    // Compute incident face center
    // If bestDot > 0, axis points same way as ref.normal, so use -axis face
    // If bestDot < 0, axis points opposite to ref.normal, so use +axis face
    float4 incNorm = vec3::mult(incAxes[bestIdx], -sign(bestDot));
    float4 incCent = vec3::add(incOrig, vec3::mult(incNorm, CAR_HALF_EX_ARR[bestIdx]));

    // Compute tangent offsets and fill vertices
    float4 off1 = vec3::mult(incAxes[t1], CAR_HALF_EX_ARR[t1]);
    float4 off2 = vec3::mult(incAxes[t2], CAR_HALF_EX_ARR[t2]);
    setFaceVertices(inc.verts, incCent, off1, off2);
}

// Face-face collision manifold generation (main entry point)
__device__ void generateFaceFaceManifold(
    SATContext& ctx, 
    SATResult& res,
    ContactManifold& contact
)
{
    ReferenceFace ref;
    IncidentFace inc;

    // Build reference and incident faces
    getReferenceFace(ctx, res, ref);
    getIncidentFace(ctx, res, ref, inc);

    // TODO: Project incident vertices to 2D
    // TODO: Clip polygon against reference rectangle
    // TODO: Compute depths and extract contact points
}
