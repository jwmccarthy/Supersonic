#pragma once

#include <cuda_runtime.h>

#include "GameState.cuh"

__device__ void resetBall(Ball* ball, int simIdx);

__device__ void resetCar(Cars* cars, int carIdx, int locIdx, bool invert);

__device__ void resetToKickoff(GameState* state, int simIdx);