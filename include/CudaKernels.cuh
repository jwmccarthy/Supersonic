#pragma once

#include <cuda_runtime.h>

#include "GameState.cuh"
#include "ArenaMesh.cuh"

__global__ void resetKernel(GameState* state);

__global__ void carArenaCollisionKernel(GameState* state, ArenaMesh* arena, Workspace* space, int* totalTris);

__global__ void prefixSumKernel(int* triCounts, int* triOffsets, int n, int* totalOut);

__global__ void carArenaNarrowPhaseKernel(GameState* state, ArenaMesh* arena, Workspace* space, int totalCars, int totalTris);