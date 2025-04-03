#pragma once

#include <stdint.h>
#include "RLConst.h"
#include "PhysicsState.h"

struct BallState : public PhysicsState {
    // Incremented every update
    // Indicates SetState()
    uint64_t updateCounter = 0;

    CUDA_BOTH BallState() : PhysicsState() {
        pos.z = RLConst::BALL_REST_Z;
    }

    CUDA_BOTH bool Matches(
        const BallState& other, 
        const float marginPos = 0.8f,
        const float marginVel = 0.4f,
        const float marginAngVel = 0.02f
    ) const;
};