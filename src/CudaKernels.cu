#include <cuda_runtime.h>

#include "CudaKernels.cuh"
#include "CudaMath.cuh"
#include "RLConstants.cuh"
#include "GameState.cuh"
#include "StateReset.cuh"

// Reset all simulations to kickoff state
__global__ void resetKernel(GameState* state)
{
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims) return;

    resetToKickoff(state, simIdx);
}