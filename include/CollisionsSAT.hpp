#pragma once

#include <cfloat>
#include <cuda_runtime.h>

// SAT context: transforms car B into car A's local space
struct SATContext
{
    float4 vecAB;   // Vector from A to B in A's local space
    float4 axB[3];  // B's local axes in A's local space
};

// SAT test result
struct SATResult
{
    float  depth   = FLT_MAX;  // Penetration depth (lower = shallower)
    float4 bestAx  = {};       // Best separating axis
    int    axisIdx = -1;       // Which axis was best (0-5: face, 6-14: edge)
    bool   overlap = true;     // Did boxes overlap?
};

// Edge axis indices for edge-edge contact
struct EdgeAxes
{
    int ai, a1, a2;  // A's incident and perpendicular axes
    int bi, b1, b2;  // B's incident and perpendicular axes
};

// Contact manifold (max 4 contact points)
struct ContactManifold
{
    int    count = 0;
    float  depths[4];
    float4 points[4];
    float4 normal;
};

// Bias toward face axes when depths are similar (per Bullet physics)
__device__ constexpr float SAT_FUDGE = 1.05f;

// Build SAT context from car transforms
__device__ SATContext buildSATContext(float4 posA, float4 rotA, float4 posB, float4 rotB);

// Full SAT test between two cars (6 face + 9 edge axes)
__device__ SATResult carCarSATTest(SATContext& ctx);

// Decode edge axis indices from SAT result
__device__ EdgeAxes getEdgeAxes(int axisIdx);
