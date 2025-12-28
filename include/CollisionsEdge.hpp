#pragma once

#include <cuda_runtime.h>
#include "CollisionsSAT.hpp"

// Edge center points for closest-point calculation
struct EdgePoints
{
    float4 pA;  // Center of edge on car A
    float4 pB;  // Center of edge on car B
};

// Offset along axis toward other car's center
__device__ float4 axisOffset(float4 vecAB, float4 axis, int idx);

// Compute edge center from perpendicular axis offsets
__device__ float4 edgeCenter(float4 vecAB, const float4* axes, int i1, int i2, float dir);

// Get edge center points for both cars
__device__ EdgePoints getEdgeCenters(const SATContext& ctx, const EdgeAxes& ax);

// Compute closest point on edge B (the contact point)
__device__ float4 getEdgeContactPoint(const EdgePoints& ep, float4 dA, float4 dB);

// Generate edge-edge contact manifold (single contact point)
__device__ void generateEdgeEdgeManifold(SATContext& ctx, SATResult& res, ContactManifold& m);
