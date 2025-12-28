/**
 * Collision Kernel Benchmark
 *
 * Tests both indexing strategies with randomized car positions
 * to measure real-world performance with divergent simulations.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#include "CudaCommon.hpp"
#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsSAT.hpp"
#include "CollisionsFace.hpp"
#include "CollisionsEdge.hpp"

// ============================================================================
// Randomization kernel - scatter cars across the field
// ============================================================================

__global__ void randomizePositionsKernel(
    float4* positions,
    float4* rotations,
    int totalCars,
    unsigned int seed,
    float spread)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalCars) return;

    // Simple pseudo-random based on seed and index
    unsigned int h = seed + idx * 1234567;
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;

    float r1 = (float)(h & 0xFFFF) / 65535.0f;
    h = h * 1103515245 + 12345;
    float r2 = (float)(h & 0xFFFF) / 65535.0f;
    h = h * 1103515245 + 12345;
    float r3 = (float)(h & 0xFFFF) / 65535.0f;

    // Random position within field bounds
    float x = (r1 - 0.5f) * spread;
    float y = (r2 - 0.5f) * spread;
    float z = CAR_REST_Z;

    // Random yaw rotation
    float yaw = r3 * 2.0f * PI;
    float4 rot = make_float4(0, 0, sinf(yaw * 0.5f), cosf(yaw * 0.5f));

    positions[idx] = make_float4(x, y, z, 0);
    rotations[idx] = rot;
}

// Simple GameState for benchmarking (bypasses reflection system)
struct BenchmarkGameState
{
    int sims;
    int nCar;
    struct {
        float4* position;
        float4* rotation;
        float4* velocity;
        float4* angularV;
    } cars;
};

// ============================================================================
// OLD indexing: adjacent threads = different pairs, same simulation
// ============================================================================

__global__ void collisionKernelOldIndexing(BenchmarkGameState* state)
{
    int nCar   = state->nCar;
    int nPairs = nCar * (nCar - 1) / 2;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= state->sims * nPairs) return;

    // OLD: adjacent threads handle different pairs of same sim
    int simIdx  = idx / nPairs;
    int pairIdx = idx % nPairs;

    int i = (int)(sqrtf(2.0f * pairIdx + 0.25f) + 0.5f);
    int j = pairIdx - i * (i - 1) / 2;

    int    base = simIdx * nCar;
    float4 posA = state->cars.position[base + i];
    float4 posB = state->cars.position[base + j];
    float4 rotA = state->cars.rotation[base + i];
    float4 rotB = state->cars.rotation[base + j];

    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult  res = carCarSATTest(ctx);

    if (!res.overlap) return;

    ContactManifold m;
    if (res.axisIdx < 6) {
        generateFaceFaceManifold(ctx, res, m);
    } else {
        generateEdgeEdgeManifold(ctx, res, m);
    }
}

// ============================================================================
// NEW indexing: adjacent threads = same pair, different simulations
// ============================================================================

__global__ void collisionKernelNewIndexing(BenchmarkGameState* state)
{
    int nCar   = state->nCar;
    int nPairs = nCar * (nCar - 1) / 2;
    int nSims  = state->sims;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nSims * nPairs) return;

    // NEW: adjacent threads handle same pair across different sims
    int simIdx  = idx % nSims;
    int pairIdx = idx / nSims;

    int i = (int)(sqrtf(2.0f * pairIdx + 0.25f) + 0.5f);
    int j = pairIdx - i * (i - 1) / 2;

    int    base = simIdx * nCar;
    float4 posA = state->cars.position[base + i];
    float4 posB = state->cars.position[base + j];
    float4 rotA = state->cars.rotation[base + i];
    float4 rotB = state->cars.rotation[base + j];

    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult  res = carCarSATTest(ctx);

    if (!res.overlap) return;

    ContactManifold m;
    if (res.axisIdx < 6) {
        generateFaceFaceManifold(ctx, res, m);
    } else {
        generateEdgeEdgeManifold(ctx, res, m);
    }
}

// ============================================================================
// Benchmark harness
// ============================================================================

struct BenchmarkConfig
{
    int sims;
    int carsPerSim;
    float spread;        // How spread out cars are (smaller = more collisions)
    int iterations;
    const char* name;
};

void runBenchmark(const BenchmarkConfig& cfg)
{
    printf("\n=== %s ===\n", cfg.name);
    printf("Simulations: %d, Cars/sim: %d, Spread: %.0f, Iterations: %d\n",
           cfg.sims, cfg.carsPerSim, cfg.spread, cfg.iterations);

    int nCar   = cfg.carsPerSim;
    int nPairs = nCar * (nCar - 1) / 2;
    int nTotal = cfg.sims * nPairs;
    int totalCars = cfg.sims * nCar;

    // Manually allocate device memory for car data
    BenchmarkGameState h_state;
    h_state.sims = cfg.sims;
    h_state.nCar = nCar;

    CUDA_CHECK(cudaMalloc(&h_state.cars.position, totalCars * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&h_state.cars.rotation, totalCars * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&h_state.cars.velocity, totalCars * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&h_state.cars.angularV, totalCars * sizeof(float4)));

    // Copy state struct to device
    BenchmarkGameState* d_state;
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(BenchmarkGameState)));
    CUDA_CHECK(cudaMemcpy(d_state, &h_state, sizeof(BenchmarkGameState), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize  = (nTotal + blockSize - 1) / blockSize;
    int randGrid  = (totalCars + blockSize - 1) / blockSize;

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float oldTime = 0, newTime = 0;

    // Get device pointers for position/rotation arrays
    float4* d_positions = h_state.cars.position;
    float4* d_rotations = h_state.cars.rotation;

    // Warm-up and test OLD indexing
    for (int i = 0; i < cfg.iterations + 5; i++)
    {
        // Randomize positions before each iteration
        randomizePositionsKernel<<<randGrid, blockSize>>>(
            d_positions, d_rotations, totalCars, i * 12345, cfg.spread);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (i >= 5)  // Skip warm-up iterations
        {
            CUDA_CHECK(cudaEventRecord(start, 0));
            collisionKernelOldIndexing<<<gridSize, blockSize>>>(d_state);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            oldTime += ms;
        }
        else
        {
            collisionKernelOldIndexing<<<gridSize, blockSize>>>(d_state);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Warm-up and test NEW indexing
    for (int i = 0; i < cfg.iterations + 5; i++)
    {
        // Randomize positions before each iteration
        randomizePositionsKernel<<<randGrid, blockSize>>>(
            d_positions, d_rotations, totalCars, i * 12345 + 99999, cfg.spread);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (i >= 5)
        {
            CUDA_CHECK(cudaEventRecord(start, 0));
            collisionKernelNewIndexing<<<gridSize, blockSize>>>(d_state);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            newTime += ms;
        }
        else
        {
            collisionKernelNewIndexing<<<gridSize, blockSize>>>(d_state);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    oldTime /= cfg.iterations;
    newTime /= cfg.iterations;

    float speedup = (oldTime - newTime) / oldTime * 100.0f;

    printf("\nResults (avg over %d iterations):\n", cfg.iterations);
    printf("  OLD indexing (idx/nPairs): %.3f us\n", oldTime * 1000.0f);
    printf("  NEW indexing (idx%%nSims):  %.3f us\n", newTime * 1000.0f);
    printf("  Speedup: %.1f%%\n", speedup);
    printf("  Pairs tested: %d\n", nTotal);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_state);

    // Clean up device memory
    cudaFree(h_state.cars.position);
    cudaFree(h_state.cars.rotation);
    cudaFree(h_state.cars.velocity);
    cudaFree(h_state.cars.angularV);
}

int main()
{
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d, Max threads/SM: %d\n", prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);

    // Test 1: Sparse (few collisions) - similar to kickoff
    runBenchmark({
        .sims = 512,
        .carsPerSim = 4,
        .spread = 4000.0f,  // Full field spread
        .iterations = 100,
        .name = "Sparse (few collisions)"
    });

    // Test 2: Dense (many collisions) - cars clustered together
    runBenchmark({
        .sims = 512,
        .carsPerSim = 4,
        .spread = 200.0f,   // Tight cluster
        .iterations = 100,
        .name = "Dense (many collisions)"
    });

    // Test 3: Medium density
    runBenchmark({
        .sims = 512,
        .carsPerSim = 4,
        .spread = 500.0f,
        .iterations = 100,
        .name = "Medium density"
    });

    // Test 4: More cars per sim
    runBenchmark({
        .sims = 512,
        .carsPerSim = 8,
        .spread = 1000.0f,
        .iterations = 100,
        .name = "8 cars per sim"
    });

    // Test 5: Many simulations
    runBenchmark({
        .sims = 4096,
        .carsPerSim = 4,
        .spread = 500.0f,
        .iterations = 50,
        .name = "4096 simulations"
    });

    // Test 6: Stress test
    runBenchmark({
        .sims = 8192,
        .carsPerSim = 6,
        .spread = 500.0f,
        .iterations = 20,
        .name = "Stress test (8192 sims, 6 cars)"
    });

    printf("\n=== Summary ===\n");
    printf("NEW indexing groups threads by pair type across simulations.\n");
    printf("This improves coherence even when positions diverge.\n");

    return 0;
}
