#include <cuda_runtime.h>

#include "CudaUtils.hpp"

__device__ int hash(int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    return (x >> 16) ^ x;
}

__device__ int2 getLocalPair(const int idx, const int cars)
{
    int pairs = cars * (cars - 1) / 2;

    int simIdx = idx / pairs;
    int locIdx = idx % pairs;

    // Triangle number 1D index -> pair indices
    int iLocal = static_cast<int>((-1.0 + sqrtf(1.0 + 8.0 * locIdx)) / 2.0);
    int jLocal = locIdx - iLocal * (iLocal + 1) / 2 + iLocal + 1;

    return {simIdx * cars + iLocal, simIdx * cars + jLocal};
}