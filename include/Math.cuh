#pragma once

#include "CudaCommon.cuh"

template <typename T>
__device__ inline T clamp(T val, int low, int high) {
    return max(min(val, high), low);
}