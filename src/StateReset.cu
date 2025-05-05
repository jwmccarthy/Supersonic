#include "StateReset.cuh"

__device__ void resetBall(GameState* state, int simIdx) {
    // Reset ball to be motionless at center of field
    state->ballPosition[simIdx] = make_float4(0, 0, BALL_REST_Z, 0);
    state->ballVelocity[simIdx] = make_float4(0, 0, 0, 0);
    state->ballAngularVelocity[simIdx] = make_float4(0, 0, 0, 0);

    // Set ball rotation to identity
    state->ballRotationF[simIdx] = make_float4(1, 0, 0, 0);
    state->ballRotationR[simIdx] = make_float4(0, 1, 0, 0);
    state->ballRotationU[simIdx] = make_float4(0, 0, 1, 0);
}

__device__ void resetCar(GameState* state, int carIdx, const CarSpawn loc, bool invert) {
    // Get XY position, yaw angle
    float 
        x = invert ? -loc.x   : loc.x,
        y = invert ? -loc.y   : loc.y,
      yaw = invert ? -loc.yaw : loc.yaw;

    // Set car position
    state->carPosition[carIdx] = make_float4(x, y, CAR_REST_Z, 0);

    // Set car rotation
    auto yawRot = Mat3::FromEulerAngles(yaw, 0, 0);
    state->carRotationF[carIdx] = yawRot.f.v;
    state->carRotationR[carIdx] = yawRot.r.v;
    state->carRotationU[carIdx] = yawRot.u.v;

    // Reset car properties
    state->carVelocity[carIdx]        = make_float4(0, 0, 0, 0);
    state->carAngularVelocity[carIdx] = make_float4(0, 0, 0, 0);
    state->carBoostAmount[carIdx]     = SPAWN_BOOST_AMOUNT;
    state->carDemolishTimer[carIdx]   = 0.0f;
    state->carDemoCooldown[carIdx]    = 0.0f;
    state->carIsOnGround[carIdx]      = true;
    state->carIsSupersonic[carIdx]    = false;
    state->carHasJumped[carIdx]       = false;
    state->carHasDoubleJumped[carIdx] = false;
    state->carIsFlipping[carIdx]      = false;
    state->carIsBoosting[carIdx]      = false;
    state->carIsDemolished[carIdx]    = false;
}

__device__ void shuffleKickoffIndices(curandState_t &st, int* kickoffIndices, int teamSize) {
    // Clamp teamSize
    if (teamSize > NUM_KICKOFF_LOCATIONS) {
        teamSize = NUM_KICKOFF_LOCATIONS;
    }

    // Fisher-Yates initialize & shuffle
    for (int i = 0; i < teamSize; ++i) {
        int remaining = NUM_KICKOFF_LOCATIONS - i;

        // Get random offset in [0, remaining-1]
        int offset = curand(&st) % remaining;
        int j = i + offset;

        // “Old” value at j
        // If we wrote it earlier (j < i), use locations[j],
        // Otherwise it’s just j.
        int val_j = (j < i ? kickoffIndices[j] : j);

        // Swap i, j (untouched i implicitly equals i)
        kickoffIndices[i] = val_j;
        kickoffIndices[j] = i;
    }
    // now locations[0...teamSize-1] is a unique random sample
}

__device__ void resetToKickoff(GameState* state, int simIdx) {
    const int nBlue   = state->numBlueCars;
    const int nOrange = state->numOrangeCars;
    const int nTotal  = state->carsPerSim;
    const int baseIdx = simIdx * nTotal;

    // Reset ball to center
    resetBall(state, simIdx);

    // Randomize kickoff indices
    curandState_t st = state->rngStates[simIdx];
    int maxTeamSize = max(nBlue, nOrange);
    int kickoffIndices[NUM_KICKOFF_LOCATIONS];
    shuffleKickoffIndices(st, kickoffIndices, maxTeamSize);

    // Reset blue & orange cars
    for (int i = 0; i < nTotal; i++) {
        int carIdx = baseIdx + i;
        int kickoffIdx = kickoffIndices[i];
        bool invertLoc = (i >= nBlue);

        // Get location from randomized index
        const CarSpawn location = KICKOFF_LOCATIONS[kickoffIdx];
        
        // Reset car to associated random location
        resetCar(state, carIdx, location, invertLoc);
    }

    state->rngStates[simIdx] = st;  // Write back for continued use
}