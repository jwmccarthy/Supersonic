#pragma once

#include <cuda_runtime.h>

#include "GameState.hpp"

__global__ void resetKernel(GameState* state);

// Unified collision kernel: SAT test + manifold generation in one pass
__global__ void collisionKernel(GameState* state);
