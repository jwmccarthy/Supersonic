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
    float  maxSep  = -FLT_MAX;
    float4 bestAx  = {};
    int    axisIdx = -1;
    bool   overlap = true;
    bool   refIsA  = true;
};

struct EdgeAxes
{
    int ai, a1, a2;  // Car A's incident and perpendicular axes
    int bi, b1, b2;  // Car B's incident and perpendicular axes
};

// Project car extents onto axis
__device__ float projectRadius(float4 L, float4 axX, float4 axY, float4 axZ);

// Build SAT context from car transforms
__device__ SATContext buildSATContext(float4 posA, float4 rotA, float4 posB, float4 rotB);

// Test single SAT axis
__device__ void testAxis(float4 L, int axis, const SATContext& ctx, SATResult& res, bool normalize);

// SAT collision test between two cars
__device__ SATResult carCarSATTest(SATContext& ctx);

// Get edge axis indices from SAT result
__device__ EdgeAxes getEdgeAxes(int axisIdx);

