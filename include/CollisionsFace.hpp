#pragma once

#include <float.h>
#include <cuda_runtime.h>

#include "CudaMath.hpp"
#include "CollisionsSAT.hpp"

struct ReferenceFace
{
    float4 normal;     // Face normal (points toward incident box)
    float4 ortho1;     // First tangent axis
    float4 ortho2;     // Second tangent axis
    float4 center;     // Face center point
    float  halfEx[2];  // Half-extents along ortho1 and ortho2
};

struct IncidentFace
{
    float4 verts[4];  // 4 corners of the face
};

struct ClipPoint
{
    float2 p;  // 2D point
    float  d;  // Depth below reference face
};

struct ClipPolygon
{
    ClipPoint points[8];
    int count;
};

// Blend between two axes based on weight (branchless select)
__device__ float4 blendAxes(float4 axA, float4 axB, float b);

// Find incident face axis (most parallel to reference normal)
__device__ int findIncidentAxis(float4 dir, const float4* axes, float& bestDot);

// Fill 4 face vertices from center and two tangent offsets
__device__ void setFaceVertices(float4 verts[4], float4 center, float4 off1, float4 off2);

// Build reference face from SAT result
__device__ void getReferenceFace(
    const SATContext& ctx, 
    const SATResult& res, 
    ReferenceFace& ref
);

// Build incident face vertices from reference face
__device__ void getIncidentFace(
    const SATContext& ctx,
    const SATResult& res,
    const ReferenceFace& ref,
    IncidentFace& inc
);

// Face-face collision manifold generation (main entry point)
__device__ void generateFaceFaceManifold(
    SATContext& ctx, 
    SATResult& res,
    ContactManifold& contact
);
