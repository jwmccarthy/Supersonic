#include "Ball.h"

CUDA_BOTH bool BallState::Matches(
    const BallState& other,
    const float marginPos,
    const float marginVel,
    const float marginAngVel
) const {
    pos.DistSq(other.pos) < (marginPos * marginPos) &&
    vel.DistSq(other.vel) < (marginVel * marginVel) &&
    angVel.DistSq(other.angVel) < (marginAngVel * marginAngVel);
}