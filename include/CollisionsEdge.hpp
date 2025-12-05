#pragma once

#include <cuda_runtime.h>
#include "CollisionsSAT.hpp"

struct EdgePoints
{
    float4 pA;  // Point on edge A
    float4 pB;  // Point on edge B
};

// Offset along one axis toward the other car's center
__device__ float4 axisOffset(float4 vecAB, float4 axis, int idx);

// Edge center via sum of offsets along the two perpendicular axes
__device__ float4 edgeCenter(float4 vecAB, const float4* axes, int i1, int i2, float dir);

// Get edge center points for both cars
__device__ EdgePoints getEdgeCenters(const SATContext& ctx, const EdgeAxes& ax);

// Compute contact point (closest point on edge B)
__device__ float4 getEdgeContactPoint(const EdgePoints& ec, float4 dirA, float4 dirB);

// Edge-edge collision manifold generation
__device__ void generateEdgeEdgeManifold(SATContext& ctx, SATResult& res);
