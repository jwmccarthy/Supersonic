#pragma once

#include <cuda_runtime.h>

#include "GameState.cuh"
#include "ArenaMesh.cuh"

__global__ void resetKernel(GameState* state);

__global__ void carArenaNarrowPhaseKernel(GameState* state, ArenaMesh* arena, Workspace* space);

__global__ void carArenaCollisionKernel(GameState* state, ArenaMesh* arena, Workspace* space);