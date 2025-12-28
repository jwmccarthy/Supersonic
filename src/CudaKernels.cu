#include <cuda_runtime.h>

#include "CudaKernels.hpp"
#include "GameState.hpp"
#include "StateReset.hpp"
#include "CollisionsSAT.hpp"
#include "CollisionsEdge.hpp"
#include "CollisionsFace.hpp"

// Reset all simulations to kickoff state
__global__ void resetKernel(GameState* state)
{
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims)
    {
        return;
    }

    resetToKickoff(state, simIdx);
}

// Unified collision kernel: SAT test + manifold generation in one pass
// Each thread handles one car pair end-to-end, keeping data in registers
__global__ void collisionKernel(GameState* state)
{
    int nCar   = state->nCar;
    int nPairs = nCar * (nCar - 1) / 2;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= state->sims * nPairs)
    {
        return;
    }

    int simIdx  = idx / nPairs;
    int pairIdx = idx % nPairs;

    // Map linear pair index to unique (i, j) where i < j
    int i = (int)(sqrtf(2.0f * pairIdx + 0.25f) + 0.5f);
    int j = pairIdx - i * (i - 1) / 2;

    // Load car transforms
    int base = simIdx * nCar;
    float4 posA = state->cars.position[base + i];
    float4 rotA = state->cars.rotation[base + i];
    float4 posB = state->cars.position[base + j];
    float4 rotB = state->cars.rotation[base + j];

    // SAT test (context and result stay in registers)
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult  res = carCarSATTest(ctx);

    // Early exit if no overlap
    if (!res.overlap)
    {
        return;
    }

    // Generate contact manifold based on collision type
    ContactManifold m;
    if (res.axisIdx < 6)
    {
        generateFaceFaceManifold(ctx, res, m);
    }
    else
    {
        generateEdgeEdgeManifold(ctx, res, m);
    }

    // TODO: Store manifold for physics resolution
}
