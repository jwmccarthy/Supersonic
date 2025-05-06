#include "InitializerKernels.cuh"

__global__ void seedKernel(GameState* state, uint64_t seed) {
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->simCount) return;

    curand_init(seed, simIdx, 0, &state->rngStates[simIdx]);
}

__global__ void resetToKickoffKernel(GameState* state) {
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->simCount) return;

    resetToKickoff(state, simIdx);
}