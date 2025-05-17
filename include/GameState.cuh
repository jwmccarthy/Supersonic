#pragma once

#include <cstdint>
#include <curand_kernel.h>

#include "RLConstants.cuh"
#include "CudaCommon.cuh"

#define GAMESTATE_FIELDS(X, SIMS, CARS)                                 \
    /* --------------- Game state (1 per sim) ---------------------- */ \
    X(uint32_t*,       phases,                 SIMS)                    \
    X(uint64_t*,       tickCounts,             SIMS)                    \
    X(float*,          gameTime,               SIMS)                    \
    X(uint32_t*,       blueScore,              SIMS)                    \
    X(uint32_t*,       orangeScore,            SIMS)                    \
    X(curandState_t*,  rngStates,              SIMS)                    \
                                                                        \
    /* --------------- Ball state (1 per sim) ---------------------- */ \
    X(float3*,         ballPosition,           SIMS)                    \
    X(float3*,         ballVelocity,           SIMS)                    \
    X(float3*,         ballAngularVelocity,    SIMS)                    \
    X(float3*,         ballRotationF,          SIMS)                    \
    X(float3*,         ballRotationR,          SIMS)                    \
    X(float3*,         ballRotationU,          SIMS)                    \
                                                                        \
    /* --------------- Car kinematics (CARS per sim) --------------- */ \
    X(float3*,         carPosition,            SIMS * CARS)             \
    X(float3*,         carVelocity,            SIMS * CARS)             \
    X(float3*,         carAngularVelocity,     SIMS * CARS)             \
                                                                        \
    /* --------------- Car rotation (CARS per sim) ----------------- */ \
    X(float3*,         carRotationF,           SIMS * CARS)             \
    X(float3*,         carRotationR,           SIMS * CARS)             \
    X(float3*,         carRotationU,           SIMS * CARS)             \
                                                                        \
    /* --------------- Car identifiers (CARS per sim) -------------- */ \
    X(bool*,           carTeam,                SIMS * CARS)             \
    X(uint32_t*,       carID,                  SIMS * CARS)             \
                                                                        \
    /* --------------- Car movement (CARS per sim) ----------------- */ \
    X(bool*,           carIsOnGround,          SIMS * CARS)             \
    X(bool*,           carIsSupersonic,        SIMS * CARS)             \
    X(bool*,           carHasJumped,           SIMS * CARS)             \
    X(bool*,           carHasDoubleJumped,     SIMS * CARS)             \
    X(bool*,           carIsFlipping,          SIMS * CARS)             \
    X(float*,          carJumpTime,            SIMS * CARS)             \
    X(float*,          carFlipTime,            SIMS * CARS)             \
    X(float*,          carAirTime,             SIMS * CARS)             \
    X(float*,          carLastJumpTime,        SIMS * CARS)             \
                                                                        \
    /* --------------- Car boost (CARS per sim) -------------------- */ \
    X(bool*,           carIsBoosting,          SIMS * CARS)             \
    X(float*,          carBoostAmount,         SIMS * CARS)             \
    X(float*,          carBoostTime,           SIMS * CARS)             \
    X(float*,          carBoostPickupCooldown, SIMS * CARS)             \
                                                                        \
    /* --------------- Wheels & suspension (4 per sim) ------------- */ \
    X(bool*,           carWheelContact,        SIMS * NUM_WHEELS)       \
    X(float*,          carSuspensionDistance,  SIMS * NUM_WHEELS)       \
    X(float*,          carRestingPositionZ,    SIMS * NUM_WHEELS)       \
    X(float*,          carSteeringAngle,       SIMS * NUM_WHEELS)       \
    X(float*,          carWheelRotationSpeed,  SIMS * NUM_WHEELS)       \
                                                                        \
    /* --------------- Demolition state (CARS per sim) ------------- */ \
    X(bool*,           carIsDemolished,        SIMS * CARS)             \
    X(float*,          carDemolishTimer,       SIMS * CARS)             \
    X(float*,          carDemoCooldown,        SIMS * CARS)             \
                                                                        \
    /* --------------- Boost pads (34 per sim) --------------------- */ \
    X(bool*,           boostPadIsActive,       SIMS * TOTAL_NUM_BOOSTS) \
    X(float*,          boostPadCooldown,       SIMS * TOTAL_NUM_BOOSTS)

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
