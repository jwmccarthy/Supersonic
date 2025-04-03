#pragma once

#include "MathTypes.h"

struct PhysicsState {
    // Position in world space
    CudaVec pos = {};

    // Rotation matrix
    CudaRotMat rotMat = CudaRotMat::GetIdentity();

    // Linear velocity
    CudaVec vel = {};

    // Angular velocity (rad/s)
    CudaVec angVel = {};

    // Get a copy of this state rotated about the Z axis
    CUDA_BOTH PhysicsState GetInvertedY() const;
};