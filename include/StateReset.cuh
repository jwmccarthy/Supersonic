#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Matrix.cuh"
#include "GameState.cuh"
#include "RLConstants.cuh"
#include "CudaCommon.cuh"

struct CarResetDefaults {
    float4 velocity        = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 angularVelocity = make_float4(0.f, 0.f, 0.f, 0.f);
    float boostAmount      = SPAWN_BOOST_AMOUNT;
    float demolishTimer    = 0.0f;
    float demoCooldown     = 0.0f;
    bool isOnGround        = true;
    bool isSupersonic      = false;
    bool hasJumped         = false;
    bool hasDoubleJumped   = false;
    bool isFlipping        = false;
    bool isBoosting        = false;
    bool isDemolished      = false;
};

__device__ const CarResetDefaults CAR_RESET_DEFAULTS;

__device__ void resetBall(GameState* state, int simIdx);

__device__ void resetCar(GameState* state, int carIdx, const CarSpawn loc, bool invert);

__device__ void shuffleKickoffIndices(curandState_t &st, int* kickoffIndices, int teamSize);

__device__ void resetToKickoff(GameState* state, int simIdx);