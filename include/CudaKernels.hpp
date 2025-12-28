#pragma once

#include <cuda_runtime.h>

#include "GameState.hpp"

__global__ void resetKernel(GameState* state);
__global__ void collisionKernel(GameState* state);
