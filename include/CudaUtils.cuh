#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ int hash(int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    return (x >> 16) ^ x;
}

__device__ __forceinline__ float hashToRange(int x, float min, float max)
{
    int h = hash(x);
    float t = (h & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    return min + t * (max - min);
}