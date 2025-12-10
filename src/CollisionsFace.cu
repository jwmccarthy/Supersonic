#include <stdio.h>

#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsFace.hpp"
#include "CollisionsSAT.hpp"

struct ReferenceFace
{
    float4 normal;
    float4 ortho1;
    float4 ortho2;
    float4 center;
    float2 halfEx;
};

struct IncidentFace
{
    float4 verts[4];
};

__device__ __forceinline__ float4 blendAxes(float4 axA, float4 axB, float b)
{
    return vec3::add(vec3::mult(axA, 1.0f - b), vec3::mult(axB, b));
}

// Face-face collision manifold generation
__device__ void generateFaceFaceManifold(
    SATContext& ctx, SATResult& res, 
    ReferenceFace& ref, IncidentFace& inc
)
{
    // Bulid axis indices around reference axis
    int i = res.axisIdx % 3;
    int j = (i + 1) % 3;
    int k = (i + 2) % 3;
    float b = (float)(res.axisIdx >= 3);

    // Get reference face normal
    float4 axA = WORLD_AXES[i];
    float4 axB = ctx.axB[i];
    ref.normal = blendAxes(axA, axB, b);

    // Point reference normal towards incident car
    float d = vec3::dot(ctx.vecAB, ref.normal);
    float s = (d >= 0.0f) ? 1.0f : -1.0f;
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

    // Get incident car axes and origin
    const float4* incAxes = (b < 0.5f) ? ctx.axB : WORLD_AXES;
    float4 incOrig = vec3::mult(ctx.vecAB, 1.0f - b);

    // Find most anti-parallel face via min dot product
    int minIdx = 0;
    float minDot = FLT_MAX;

    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        float d = vec3::dot(ref.normal, incAxes[i]);
        minIdx = (d < minDot) ? i : minIdx;
        minDot = fminf(minDot, d);
    }

    // Get face indices tangent to min axis
    int t1 = (minIdx + 1) % 3;
    int t2 = (minIdx + 2) % 3;

    // Get car extents from axes
    float incHalfI = CAR_HALF_EX_ARR[minIdx];
    float incHalfT1 = CAR_HALF_EX_ARR[t1];
    float incHalfT2 = CAR_HALF_EX_ARR[t2];

    // Correct sign of min axis
    float incSign = (minDot < 0.0f) ? 1.0f : -1.0f;
    float4 incNorm = vec3::mult(incAxes[minIdx], incSign);
    float4 incCent = vec3::add(incOrig, vec3::mult(incNorm, incHalfI));

    // Combine tangent offsets
    float4 off1 = vec3::mult(incAxes[t1], incHalfT1);
    float4 off2 = vec3::mult(incAxes[t2], incHalfT2);

    // Get incident face vertices (4 corners of rectangle)
    inc.verts[0] = vec3::add(incCent, vec3::add(off1, off2));  // +t1 +t2
    inc.verts[1] = vec3::add(incCent, vec3::sub(off1, off2));  // +t1 -t2
    inc.verts[2] = vec3::sub(incCent, vec3::add(off1, off2));  // -t1 -t2
    inc.verts[3] = vec3::sub(incCent, vec3::sub(off1, off2));  // -t1 +t2
}