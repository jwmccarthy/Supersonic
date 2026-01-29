#include <cuda_runtime.h>

#include "CudaKernels.cuh"
#include "CudaMath.cuh"
#include "RLConstants.cuh"
#include "GameState.cuh"
#include "ArenaMesh.cuh"
#include "StateReset.cuh"
#include "CarArenaCollision.cuh"

// Reset all simulations to kickoff state
__global__ void resetKernel(GameState* state)
{
    // One sim per thread
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims) return;

    resetToKickoff(state, simIdx);
}

__global__ void carArenaCollisionKernel(GameState* state, ArenaMesh* arena, int* debug)
{
    // 16 threads per car (2 cars per warp)
    int carIdx = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_CAR;
    int laneIdx = threadIdx.x % THREADS_PER_CAR;
    if (carIdx >= state->sims * state->nCar) return;

    carArenaBroadPhase(state, arena, carIdx, laneIdx, debug);
}