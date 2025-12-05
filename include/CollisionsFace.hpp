#pragma once

#include <cuda_runtime.h>

#include "CollisionsSAT.hpp"

// Face-face collision manifold generation
__device__ void generateFaceFaceManifold(SATContext& ctx, SATResult& res);

