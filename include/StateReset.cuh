#pragma once

#include <cuda_runtime.h>

#include "GameState.hpp"

__device__ void resetBall(Ball* state, int simIdx);

__device__ void resetCar(Cars* state, int simIdx);

__device__ void resetToKickoff(GameState* state, int simIdx);