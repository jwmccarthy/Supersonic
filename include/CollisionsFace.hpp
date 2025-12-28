#pragma once

#include <cuda_runtime.h>
#include "CollisionsSAT.hpp"

// Reference face for Sutherland-Hodgman clipping
struct ReferenceFace
{
    float4 normal;  // Face normal (points toward incident box)
    float4 ortho1;  // First tangent axis
    float4 ortho2;  // Second tangent axis
    float4 center;  // Face center point
    float2 halfEx;  // Half-extents along ortho1 and ortho2
};

// Incident face (4 vertices)
struct IncidentFace
{
    float4 verts[4];
};

// 2D clip point with depth
struct ClipPoint
{
    float2 p;  // 2D position on reference plane
    float  d;  // Depth below reference face
};

// Clipped polygon (up to 8 points after clipping)
struct ClipPolygon
{
    ClipPoint points[8];
    int       count;
};

// Blend between two axes (branchless select)
__device__ float4 blendAxes(float4 axA, float4 axB, float b);

// Find incident face axis (most antiparallel to reference normal)
__device__ int findIncidentAxis(float4 dir, const float4* axes, float& bestDot);

// Fill 4 face vertices from center and tangent offsets
__device__ void setFaceVertices(float4 verts[4], float4 c, float4 o1, float4 o2);

// Build reference face from SAT result
__device__ void getReferenceFace(const SATContext& ctx, const SATResult& res, ReferenceFace& ref);

// Build incident face from reference face
__device__ void getIncidentFace(const SATContext& ctx, const SATResult& res, const ReferenceFace& ref, IncidentFace& inc);

// Cull contact points to max 4 (keeps deepest + 3 at 90 degree intervals)
__device__ void cullContactPoints(ContactManifold& m);

// Generate face-face contact manifold (Sutherland-Hodgman clipping)
__device__ void generateFaceFaceManifold(SATContext& ctx, SATResult& res, ContactManifold& m);
