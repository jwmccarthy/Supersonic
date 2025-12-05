#pragma once

#include <cuda_runtime.h>

#include "CollisionsSAT.hpp"

// Collision manifold generation
__device__ void carCarCollision(float4 posA, float4 rotA, float4 posB, float4 rotB);
