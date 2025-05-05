#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Matrix.cuh"
#include "GameState.cuh"
#include "RLConstants.cuh"
#include "CudaCommon.cuh"

__device__ void resetBall(GameState* state, int simIdx);

__device__ void shuffleKickoffIndices(curandState_t &st, const CarSpawn loc, int teamSize);

__global__ void resetToKickoff(GameState* state, int simIdx);