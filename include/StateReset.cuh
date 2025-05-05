#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Matrix.cuh"
#include "GameState.cuh"
#include "RLConstants.cuh"
#include "CudaCommon.cuh"

__device__ void resetBall(GameState* state, int simIdx);

__device__ void resetCar(GameState* state, int carIdx, const CarSpawn loc, bool invert);

__device__ void shuffleKickoffIndices(curandState_t &st, int* kickoffIndices, int teamSize);

__device__ void resetToKickoff(GameState* state, int simIdx);