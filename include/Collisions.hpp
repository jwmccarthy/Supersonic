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
    bool   overlap = true;
};

// Project car extents onto axis
__device__ float projectRadius(float4 L, float4 axX, float4 axY, float4 axZ);

// Build SAT context from car transforms
__device__ SATContext buildSATContext(float4 posA, float4 rotA, float4 posB, float4 rotB);

// Test single SAT axis
__device__ void testAxis(float4 L, bool normalize, const SATContext& ctx, SATResult& res);

// SAT collision test between two cars
__device__ SATResult carCarSATTest(float4 posA, float4 rotA, float4 posB, float4 rotB);

// Collision manifold generation
__device__ bool carCarCollision(float4 posA, float4 rotA, float4 posB, float4 rotB);