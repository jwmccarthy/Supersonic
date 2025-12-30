#pragma once

#include <cuda_runtime.h>

__device__ int hash(int x);

__device__ int2 getLocalPair(const int idx, const int cars);