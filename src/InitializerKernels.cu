#include "InitializerKernels.cuh"

__global__ void seedKernel(GameState* state, uint64_t seed) {
    // Debug version: only initialize the first element to isolate the problem
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Debug: initializing RNG for simulation 0 only\n");
        printf("Debug: state->simCount = %d\n", state->simCount);
        printf("Debug: state->rngStates = %p\n", state->rngStates);
        
        // Check if rngStates is valid
        if (state->rngStates != nullptr) {
            // Only initialize one state for debugging
            curand_init(seed, 0, 0, &state->rngStates[0]);
            printf("Debug: Successfully initialized first RNG state\n");
        } else {
            printf("Debug: ERROR - rngStates is NULL!\n");
        }
    }
}

__global__ void resetToKickoffKernel(GameState* state) {
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->simCount) return;

    resetToKickoff(state, simIdx);
}