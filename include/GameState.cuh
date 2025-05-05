#pragma once

#include <cstdint>
#include <curand_kernel.h>

#include "RLConstants.cuh"
#include "CudaCommon.cuh"

#define GAMESTATE_FIELDS(X, SIMS, CARS)                               \
    /* --- Game state (per-sim arrays) -------------------------- */  \
    X(uint32_t*,       phases,                 SIMS)                  \
    X(uint64_t*,       tickCounts,             SIMS)                  \
    X(float*,          gameTime,               SIMS)                  \
    X(uint32_t*,       blueScore,              SIMS)                  \
    X(uint32_t*,       orangeScore,            SIMS)                  \
    X(curandState_t*,  rngStates,              SIMS)                  \
                                                                      \
    /* --- Ball state (1-per-sim arrays) ------------------------ */  \
    X(float4*,         ballPosition,           SIMS)                  \
    X(float4*,         ballVelocity,           SIMS)                  \
    X(float4*,         ballAngularVelocity,    SIMS)                  \
    X(float4*,         ballRotationF,          SIMS)                  \
    X(float4*,         ballRotationR,          SIMS)                  \
    X(float4*,         ballRotationU,          SIMS)                  \
                                                                      \
    /* --- Car kinematics (CARS-per-sim) ------------------------ */  \
    X(float4*,         carPosition,            SIMS * CARS)           \
    X(float4*,         carVelocity,            SIMS * CARS)           \
    X(float4*,         carAngularVelocity,     SIMS * CARS)           \
                                                                      \
    /* --- Car rotation (CARS-per-sim) -------------------------- */  \
    X(float4*,         carRotationF,           SIMS * CARS)           \
    X(float4*,         carRotationR,           SIMS * CARS)           \
    X(float4*,         carRotationU,           SIMS * CARS)           \
                                                                      \
    /* --- Car identifiers (CARS-per-sim) ----------------------- */  \
    X(bool*,           carTeam,                SIMS * CARS)           \
    X(uint32_t*,       carID,                  SIMS * CARS)           \
                                                                      \
    /* --- Car movement (CARS-per-sim) -------------------------- */  \
    X(bool*,           carIsOnGround,          SIMS * CARS)           \
    X(bool*,           carIsSupersonic,        SIMS * CARS)           \
    X(bool*,           carHasJumped,           SIMS * CARS)           \
    X(bool*,           carHasDoubleJumped,     SIMS * CARS)           \
    X(bool*,           carIsFlipping,          SIMS * CARS)           \
    X(float*,          carJumpTime,            SIMS * CARS)           \
    X(float*,          carFlipTime,            SIMS * CARS)           \
    X(float*,          carAirTime,             SIMS * CARS)           \
    X(float*,          carLastJumpTime,        SIMS * CARS)           \
                                                                      \
    /* --- Car boost (CARS-per-sim) ----------------------------- */  \
    X(bool*,           carIsBoosting,          SIMS * CARS)           \
    X(float*,          carBoostAmount,         SIMS * CARS)           \
    X(float*,          carBoostTime,           SIMS * CARS)           \
    X(float*,          carBoostPickupCooldown, SIMS * CARS)           \
                                                                      \
    /* --- Wheels & suspension (4 pads per-sim) ----------------- */  \
    X(bool*,           carWheelContact,        SIMS * NUM_WHEELS)     \
    X(float*,          carSuspensionDistance,  SIMS * NUM_WHEELS)     \
    X(float*,          carRestingPositionZ,    SIMS * NUM_WHEELS)     \
    X(float*,          carSteeringAngle,       SIMS * NUM_WHEELS)     \
    X(float*,          carWheelRotationSpeed,  SIMS * NUM_WHEELS)     \
                                                                      \
    /* --- Demolition state (CARS-per-sim) ---------------------- */  \
    X(bool*,           carIsDemolished,        SIMS * CARS)           \
    X(float*,          carDemolishTimer,       SIMS * CARS)           \
    X(float*,          carDemoCooldown,        SIMS * CARS)           \
                                                                      \
    /* --- Boost pads (per-sim * 34 pads) ----------------------- */  \
    X(bool*,           boostPadIsActive,       SIMS * NUM_BOOST_PADS) \
    X(float*,          boostPadCooldown,       SIMS * NUM_BOOST_PADS)

struct __align__(16) GameState {
    int simCount;
    int numBlueCars;
    int numOrangeCars;
    int carsPerSim;
    int randomSeed;

    #define VIEW_FIELD(type, name, count) \
        type name;
    GAMESTATE_FIELDS(VIEW_FIELD, simCount, carsPerSim)
    #undef VIEW_FIELD
};
