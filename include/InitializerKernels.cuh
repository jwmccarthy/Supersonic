#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "StateReset.cuh"
#include "CudaCommon.cuh"
#include "GameState.cuh"
#include "RLConstants.cuh"

__global__ void seedKernel(GameState* state, ulong seed);

__global__ void resetToKickoffKernel(GameState* state);