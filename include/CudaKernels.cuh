#pragma once

#include <cuda_runtime.h>

#include "GameState.cuh"

__global__ void resetKernel(GameState* state);
