#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "StateReset.cuh"
#include "CudaCommon.cuh"
#include "GameState.cuh"
#include "RLConstants.cuh"

__global__ void seedKernel(GameState* state, uint64_t seed);

__global__ void resetToKickoffKernel(GameState* state);