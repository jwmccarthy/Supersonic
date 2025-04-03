#include "PhysicsState.h"

CUDA_BOTH PhysicsState PhysicsState::GetInvertedY() const {
    // Invert X and Y, preserve Z
    static const CudaVec INVERT_SCALE(-1, -1, 1);

    PhysicsState inverted = *this;
    
    inverted.pos *= INVERT_SCALE;
    inverted.vel *= INVERT_SCALE;
    inverted.angVel *= INVERT_SCALE;
    
    for (int i = 0; i < 3; i++) {
        inverted.rotMat[i] *= INVERT_SCALE;
    }

    return inverted;
}