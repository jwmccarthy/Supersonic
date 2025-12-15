#pragma once

#include <cfloat>
#include <cuda_runtime.h>

struct SATContext
{
    float4 vecAB;
    float4 axB[3];
};

struct SATResult
{
    float  depth   = FLT_MAX;   // Penetration depth (positive = overlap, lower = shallower)
    float4 bestAx  = {};        // Best separating axis
    int    axisIdx = -1;        // Which axis was best (0-14)
    bool   overlap = true;      // Did boxes overlap?
};

struct EdgeAxes
{
    int ai, a1, a2;  // Car A's incident and perpendicular axes
    int bi, b1, b2;  // Car B's incident and perpendicular axes
};

struct ContactManifold
{
    int    count = 0;  // # of points in the manifold
    float  depths[8];  // Penetration depths at points (negative if penetrating)
    float4 points[8];  // World-space contact points
    float4 normal;     // Shared contact normal
};

// Bias toward face axes when depths are similar
__device__ constexpr float SAT_FUDGE = 1.05f;

// Project car extents onto axis
__device__ float projectRadius(float4 L, float4 axX, float4 axY, float4 axZ);

// Build SAT context from car transforms
__device__ SATContext buildSATContext(float4 posA, float4 rotA, float4 posB, float4 rotB);

// Test single SAT axis (isEdgeAxis applies fudge factor per Bullet physics)
__device__ void testAxis(float4 L, int axis, const SATContext& ctx, SATResult& res, bool isEdgeAxis);

// SAT collision test between two cars
__device__ SATResult carCarSATTest(SATContext& ctx);

// Get edge axis indices from SAT result
__device__ EdgeAxes getEdgeAxes(int axisIdx);

