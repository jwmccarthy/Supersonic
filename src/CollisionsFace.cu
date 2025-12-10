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

__device__ __forceinline__ float4 blendAxes(float4 axA, float4 axB, float b)
{
    return vec3::add(vec3::mult(axA, 1.0f - b), vec3::mult(axB, b));
}

// Face-face collision manifold generation
__device__ void generateFaceFaceManifold(SATContext& ctx, SATResult& res)
{
    ReferenceFace ref;

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

    printf("Reference face center: (%f, %f, %f)\n", ref.center.x, ref.center.y, ref.center.z);

    // Get incident face vertices

    // Project incident face vertices onto reference face plane

    // Clip projected quad against reference face edges

    // Transform clipped quad back to 3D space

    // Cull points to best 4

    // Output contact manifold
}