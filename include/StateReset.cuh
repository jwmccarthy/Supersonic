#pragma once

#include <cuda_runtime.h>

#include "StateReset.cuh"
#include "GameState.cuh"
#include "CudaUtils.cuh"
#include "RLConstants.cuh"

__device__ __forceinline__ void resetBall(Ball* ball, int simIdx)
{
    ball->position[simIdx] = { 0, 0, BALL_REST_Z, 0 };
    ball->velocity[simIdx] = { 0, 0, 0, 0 };
    ball->angularV[simIdx] = { 0, 0, 0, 0 };
    ball->rotation[simIdx] = { 0, 0, 0, 1 };
}

__device__ __forceinline__ void resetCar(Cars* cars, int carIdx, int locIdx, bool invert)
{
    CarSpawn loc = KICKOFF_LOCATIONS[locIdx % 5];

    float x   = loc.x;
    float y   = loc.y;
    float z   = loc.z;
    float yaw = loc.yaw;

    cars->position[carIdx] = { x, y, z, 0 };
    cars->velocity[carIdx] = { 0, 0, 0, 0 };
    cars->angularV[carIdx] = { 0, 0, 0, 0 };

    // Quaternion rotation via yaw
    cars->rotation[carIdx] = { 0, 0, sinf(yaw / 2), cosf(yaw / 2) };
}

__device__ __forceinline__ void resetToKickoff(GameState* state, int simIdx)
{
    const int sims = state->sims;
    const int numB = state->numB;
    const int numO = state->numO;
    const int nCar = state->nCar;

    // Pseudorandom kickoff permutation
    const int  permIdx = hash(simIdx ^ sims) % 120;
    const int* carLocs = KICKOFF_PERMUTATIONS[permIdx];

    // Ball back to center field
    resetBall(&state->ball, simIdx);

    #pragma unroll 2
    for (int team = 0; team < 2; team++)
    {
        // Invert orange positions
        const bool invert = team;
        const int numCars = team ? numO : numB;
        
        for (int i = 0; i < numCars; i++)
        {
            const int locIdx = carLocs[i];
            const int carIdx = simIdx * nCar + (team * numB + i);

            // Place cars at kickoff positions
            resetCar(&state->cars, carIdx, locIdx, invert);
        }
    }
}

__device__ __forceinline__ void randomizeInitialPositions(GameState* state, int simIdx)
{
    const int sims = state->sims;
    const int numB = state->numB;
    const int numO = state->numO;
    const int nCar = state->nCar;

    // Pseudorandom kickoff permutation
    const int  permIdx = hash(simIdx ^ sims) % 120;
    const int* carLocs = KICKOFF_PERMUTATIONS[permIdx];

    // Ball back to center field
    resetBall(&state->ball, simIdx);

    #pragma unroll 2
    for (int team = 0; team < 2; team++)
    {
        // Invert orange positions
        const bool invert = team;
        const int numCars = team ? numO : numB;
        
        for (int i = 0; i < numCars; i++)
        {
            const int locIdx = carLocs[i];
            const int carIdx = simIdx * nCar + (team * numB + i);

            // Place cars at kickoff positions
            resetCar(&state->cars, carIdx, locIdx, invert);
        }
    }
}