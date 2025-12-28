#include <cuda_runtime.h>

#include "CudaKernels.hpp"
#include "GameState.hpp"
#include "StateReset.hpp"
#include "CollisionsSAT.hpp"
#include "CollisionsEdge.hpp"
#include "CollisionsFace.hpp"

__global__ void resetKernel(GameState* state)
{
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims) return;

    resetToKickoff(state, simIdx);
}

// SAT test kernel - writes results to Cols SoA
__global__ void satTestKernel(GameState* state)
{
    int numCars = state->nCar;
    int numPairs = numCars * (numCars - 1) / 2;

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx >= state->sims * numPairs) return;

    int simIdx = globalIdx / numPairs;
    int pairIdx = globalIdx % numPairs;

    // Map linear index to unique pair (i, j) where i < j
    int i = (int)(sqrtf(2.0f * pairIdx + 0.25f) + 0.5f);
    int j = pairIdx - i * (i - 1) / 2;

    // Base index for the car state arrays
    const int carBase = simIdx * numCars;

    // Car pair positions and rotations
    float4 posA = state->cars.position[carBase + i];
    float4 rotA = state->cars.rotation[carBase + i];
    float4 posB = state->cars.position[carBase + j];
    float4 rotB = state->cars.rotation[carBase + j];

    // Build SAT context and run test
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult res = carCarSATTest(ctx);

    // Write results to Cols SoA
    Cols& cols = state->cols;
    cols.vecAB[globalIdx] = ctx.vecAB;
    cols.axB0[globalIdx] = ctx.axB[0];
    cols.axB1[globalIdx] = ctx.axB[1];
    cols.axB2[globalIdx] = ctx.axB[2];

    cols.depth[globalIdx] = res.depth;
    cols.bestAx[globalIdx] = res.bestAx;
    cols.axisIdx[globalIdx] = res.axisIdx;
    cols.overlap[globalIdx] = res.overlap;

    cols.carA[globalIdx] = i;
    cols.carB[globalIdx] = j;
}

// Manifold generation kernel - reads from Cols SoA
__global__ void manifoldKernel(GameState* state)
{
    int numCars = state->nCar;
    int numPairs = numCars * (numCars - 1) / 2;

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx >= state->sims * numPairs) return;

    // Read SAT results from Cols SoA
    Cols& cols = state->cols;

    // Skip if no overlap
    if (!cols.overlap[globalIdx]) return;

    // Reconstruct SATContext
    SATContext ctx;
    ctx.vecAB = cols.vecAB[globalIdx];
    ctx.axB[0] = cols.axB0[globalIdx];
    ctx.axB[1] = cols.axB1[globalIdx];
    ctx.axB[2] = cols.axB2[globalIdx];

    // Reconstruct SATResult
    SATResult res;
    res.depth = cols.depth[globalIdx];
    res.bestAx = cols.bestAx[globalIdx];
    res.axisIdx = cols.axisIdx[globalIdx];
    res.overlap = cols.overlap[globalIdx];

    // Generate contact manifold
    ContactManifold contact;

    if (res.axisIdx < 6)
    {
        // Face-face collision
        generateFaceFaceManifold(ctx, res, contact);
    }
    else
    {
        // Edge-edge collision
        generateEdgeEdgeManifold(ctx, res, contact);
    }

    // TODO: Store contact manifold for physics resolution
}