#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "StateReset.cuh"
#include "CudaCommon.cuh"
#include "GameState.cuh"
#include "RLConstants.cuh"

__global__ void seedKernel(GameState* state, ulong seed) {
    int simIdx = blockIdx.x;
    if (simIdx >= state->simCount) return;

    curand_init(seed, simIdx, 0, &state->rngStates[simIdx]);
}

__global__ void resetToKickoffKernel(GameState* state) {
    int simIdx = blockIdx.x;
    if (simIdx >= state->simCount) return;

    resetToKickoff(state, simIdx);
}