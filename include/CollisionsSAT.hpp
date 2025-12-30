#pragma once

#include <cfloat>
#include <cuda_runtime.h>
#include "RLConstants.hpp"
#include "CudaMath.hpp"

// B transformed into A's local space
struct SATContext
{
    float4 vecAB;   // A->B vector in A's space
    float4 axB[3];  // B's axes in A's space
};

struct SATResult
{
    float  depth   = FLT_MAX;  // Penetration depth
    float4 bestAx  = {};       // Separating axis
    int    axisIdx = -1;       // Axis index (0-5: face, 6-14: edge)
    bool   overlap = true;
};

struct EdgeAxes
{
    int ai, a1, a2;  // A's incident and perpendicular axes
    int bi, b1, b2;  // B's incident and perpendicular axes
};

struct ContactManifold
{
    int    count = 0;
    float  depths[4];
    float4 points[4];
    float4 normal;
};

// Inline helpers for SAT tests
__device__ __forceinline__ float getComp(float4 v, int i)
{
    return (i == 0) ? v.x : (i == 1) ? v.y : v.z;
}

__device__ __forceinline__ float getComp(float3 v, int i)
{
    return (i == 0) ? v.x : (i == 1) ? v.y : v.z;
}

// Project OBB half-extents onto axis
__device__ __forceinline__ float projectOBB(float4 axis)
{
    return CAR_HALF_EX.x * fabsf(axis.x) +
           CAR_HALF_EX.y * fabsf(axis.y) +
           CAR_HALF_EX.z * fabsf(axis.z);
}

// Project B's OBB for face axis i
__device__ __forceinline__ float projectB(float3 absB0, float3 absB1, float3 absB2, int i)
{
    return CAR_HALF_EX.x * getComp(absB0, i) +
           CAR_HALF_EX.y * getComp(absB1, i) +
           CAR_HALF_EX.z * getComp(absB2, i);
}

// Test axis, update result; returns false if separating
__device__ __forceinline__ bool testAxis(
    SATResult& res, float4 axis, float d, float rA, float rB,
    int axisIdx, float fudge = 1.0f)
{
    float s = sign(d);
    d = fabsf(d);
    float depth = rA + rB - d;

    if (depth * fudge < res.depth) {
        res.depth = depth;
        res.bestAx = vec3::mult(axis, s);
        res.axisIdx = axisIdx;
    }

    return depth > 0.0f;
}

__device__ SATContext buildSATContext(float4 posA, float4 rotA, float4 posB, float4 rotB);

__device__ SATResult carCarSATTest(SATContext& ctx);

__device__ EdgeAxes getEdgeAxes(int axisIdx);
