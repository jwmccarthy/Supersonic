#pragma once

#include <cuda_runtime.h>

#include "GameState.hpp"

__global__ void resetKernel(GameState* state);

// SAT test kernel - writes results to Cols SoA
__global__ void satTestKernel(GameState* state);

// Manifold generation kernel - reads from Cols SoA
__global__ void manifoldKernel(GameState* state);