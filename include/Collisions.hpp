#pragma once

#include <cuda_runtime.h>

__device__ bool carCarCollision(float4 posA, float4 rotA, float4 posB, float4 rotB);