#include <cuda_runtime.h>

#include "StateReset.hpp"
#include "GameState.hpp"
#include "CudaUtils.hpp"
#include "RLConstants.hpp"

__device__ void resetBall(Ball* ball, int simIdx)
{
    ball->position[simIdx] = make_float4(0, 0, BALL_REST_Z, 0);
    ball->velocity[simIdx] = make_float4(0, 0, 0, 0);
    ball->angularV[simIdx] = make_float4(0, 0, 0, 0);
    ball->rotation[simIdx] = make_float4(0, 0, 0, 1);
}

__device__ void resetCar(Cars* cars, int carIdx, int locIdx, bool invert)
{
    CarSpawn loc = KICKOFF_LOCATIONS[locIdx];

    // Invert orange team spawn
    float x   = invert ? -loc.x   : loc.x;
    float y   = invert ? -loc.y   : loc.y;
    float yaw = invert ? -loc.yaw : loc.yaw;

    cars->position[carIdx] = make_float4(x, y, CAR_REST_Z, 0);
    cars->velocity[carIdx] = make_float4(0, 0, 0, 0);
    cars->angularV[carIdx] = make_float4(0, 0, 0, 0);
    
    // Quaternion rotation via yaw
    cars->rotation[carIdx] = make_float4(0, 0, sinf(yaw / 2), cosf(yaw / 2));
}

__device__ void resetToKickoff(GameState* state, int simIdx)
{
    const int numSims = state->sims;
    const int numBCar = state->numB;
    const int numOCar = state->numO;
    const int simCars = state->nCar;

    // Pseudorandom kickoff permutation
    const int  permIdx = hash(simIdx ^ numSims) % 120;
    const int* carLocs = KICKOFF_PERMUTATIONS[permIdx];

    // Ball back to center field
    resetBall(&state->ball, simIdx);

    #pragma unroll 2
    for (int team = 0; team < 2; team++)
    {
        // Invert orange positions
        const bool invert = team;
        const int numCars = team ? numOCar : numBCar;
        
        #pragma unroll
        for (int i = 0; i < numCars; i++)
        {
            const int locIdx = carLocs[i];
            const int carIdx = simIdx * simCars + (team * numBCar + i);

            // Place cars at kickoff positions
            resetCar(&state->cars, carIdx, locIdx, invert);
        }
    }
}