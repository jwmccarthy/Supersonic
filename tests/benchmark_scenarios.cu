#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#include "RLEnvironment.hpp"
#include "CudaKernels.hpp"
#include "RLConstants.hpp"
#include "GameState.hpp"

// Rocket League field dimensions (UE units)
constexpr float FIELD_HALF_X = 4096.0f;   // Half length
constexpr float FIELD_HALF_Y = 5120.0f;   // Half width
constexpr float FIELD_Z_MAX  = 2044.0f;   // Ceiling

// Scenario types
enum Scenario {
    KICKOFF,           // Standard kickoff positions
    BALL_CHASE,        // Cars clustered chasing ball
    SPREAD_ROTATION,   // Cars spread across field rotating
    CORNER_PLAY,       // Cars fighting in corner
    GOAL_SCRAMBLE,     // Cars clustered near goal
    AERIAL_CONTEST,    // Cars in air contesting
    DEMO_CHASE,        // Cars chasing for demos
    NUM_SCENARIOS
};

const char* SCENARIO_NAMES[] = {
    "KICKOFF",
    "BALL_CHASE",
    "SPREAD_ROTATION",
    "CORNER_PLAY",
    "GOAL_SCRAMBLE",
    "AERIAL_CONTEST",
    "DEMO_CHASE"
};

// Simple hash for randomization
__device__ __host__ unsigned int hash(unsigned int x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

// Random float in [0, 1]
__device__ float randf(unsigned int& seed) {
    seed = hash(seed);
    return (float)(seed & 0xFFFF) / 65535.0f;
}

// Random float in [min, max]
__device__ float randf(unsigned int& seed, float min, float max) {
    return min + randf(seed) * (max - min);
}

// Quaternion from yaw angle
__device__ float4 quatFromYaw(float yaw) {
    return make_float4(0, 0, sinf(yaw * 0.5f), cosf(yaw * 0.5f));
}

// Set car position and rotation
__device__ void setCar(GameState* state, int simIdx, int carIdx,
                       float x, float y, float z, float yaw) {
    int idx = simIdx * state->nCar + carIdx;
    state->cars.position[idx] = make_float4(x, y, z, 0);
    state->cars.rotation[idx] = quatFromYaw(yaw);
}

// Kernel to set positions based on scenario
__global__ void setScenarioPositions(GameState* state, Scenario scenario,
                                      int tick, unsigned int baseSeed) {
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->sims) return;

    int nCar = state->nCar;
    unsigned int seed = baseSeed + simIdx * 12345 + tick * 67890;

    switch (scenario) {
        case KICKOFF: {
            // Standard kickoff - cars at fixed positions with slight variation
            float offsets[4][2] = {{-2048, -2560}, {2048, -2560}, {-2048, 2560}, {2048, 2560}};
            for (int i = 0; i < nCar && i < 4; i++) {
                float x = offsets[i][0] + randf(seed, -50, 50);
                float y = offsets[i][1] + randf(seed, -50, 50);
                float yaw = atan2f(-y, -x);  // Face center
                setCar(state, simIdx, i, x, y, CAR_REST_Z, yaw);
            }
            break;
        }

        case BALL_CHASE: {
            // Ball at random position, all cars chasing it (high collision chance)
            float ballX = randf(seed, -FIELD_HALF_X * 0.8f, FIELD_HALF_X * 0.8f);
            float ballY = randf(seed, -FIELD_HALF_Y * 0.8f, FIELD_HALF_Y * 0.8f);

            for (int i = 0; i < nCar; i++) {
                // Cars within 300 units of ball
                float angle = randf(seed) * 2.0f * PI;
                float dist = randf(seed, 50, 300);
                float x = ballX + cosf(angle) * dist;
                float y = ballY + sinf(angle) * dist;
                float yaw = atan2f(ballY - y, ballX - x);  // Face ball
                setCar(state, simIdx, i, x, y, CAR_REST_Z, yaw);
            }
            break;
        }

        case SPREAD_ROTATION: {
            // Cars spread across field in rotation pattern (low collision chance)
            for (int i = 0; i < nCar; i++) {
                float x = randf(seed, -FIELD_HALF_X * 0.9f, FIELD_HALF_X * 0.9f);
                float y = randf(seed, -FIELD_HALF_Y * 0.9f, FIELD_HALF_Y * 0.9f);
                float yaw = randf(seed) * 2.0f * PI;
                setCar(state, simIdx, i, x, y, CAR_REST_Z, yaw);
            }
            break;
        }

        case CORNER_PLAY: {
            // Cars clustered in a corner (very high collision chance)
            float cornerX = (randf(seed) > 0.5f ? 1 : -1) * FIELD_HALF_X * 0.85f;
            float cornerY = (randf(seed) > 0.5f ? 1 : -1) * FIELD_HALF_Y * 0.85f;

            for (int i = 0; i < nCar; i++) {
                float x = cornerX + randf(seed, -200, 200);
                float y = cornerY + randf(seed, -200, 200);
                float yaw = randf(seed) * 2.0f * PI;
                setCar(state, simIdx, i, x, y, CAR_REST_Z, yaw);
            }
            break;
        }

        case GOAL_SCRAMBLE: {
            // Cars near goal line (medium-high collision chance)
            float goalY = (randf(seed) > 0.5f ? 1 : -1) * FIELD_HALF_Y;

            for (int i = 0; i < nCar; i++) {
                float x = randf(seed, -893, 893);  // Goal width
                float y = goalY + randf(seed, -400, 100) * (goalY > 0 ? -1 : 1);
                float yaw = (goalY > 0 ? -PI/2 : PI/2) + randf(seed, -0.5f, 0.5f);
                setCar(state, simIdx, i, x, y, CAR_REST_Z, yaw);
            }
            break;
        }

        case AERIAL_CONTEST: {
            // Cars in the air contesting (medium collision chance)
            float contestX = randf(seed, -1000, 1000);
            float contestY = randf(seed, -1000, 1000);
            float contestZ = randf(seed, 500, 1500);

            for (int i = 0; i < nCar; i++) {
                float x = contestX + randf(seed, -300, 300);
                float y = contestY + randf(seed, -300, 300);
                float z = contestZ + randf(seed, -200, 200);
                // Random 3D orientation (simplified)
                float yaw = randf(seed) * 2.0f * PI;
                float pitch = randf(seed, -0.5f, 0.5f);
                setCar(state, simIdx, i, x, y, z, yaw);
            }
            break;
        }

        case DEMO_CHASE: {
            // Pairs of cars close together (50% high collision chance)
            for (int i = 0; i < nCar; i += 2) {
                float x = randf(seed, -FIELD_HALF_X * 0.8f, FIELD_HALF_X * 0.8f);
                float y = randf(seed, -FIELD_HALF_Y * 0.8f, FIELD_HALF_Y * 0.8f);
                float yaw = randf(seed) * 2.0f * PI;

                setCar(state, simIdx, i, x, y, CAR_REST_Z, yaw);

                if (i + 1 < nCar) {
                    // Second car right behind first (likely collision)
                    float x2 = x - cosf(yaw) * randf(seed, 100, 200);
                    float y2 = y - sinf(yaw) * randf(seed, 100, 200);
                    setCar(state, simIdx, i + 1, x2, y2, CAR_REST_Z, yaw);
                }
            }
            break;
        }

        default:
            break;
    }
}

// Time the collision kernel
float timeCollisionKernel(GameState* d_state, int sims, int nCar, int iterations) {
    int nPairs = nCar * (nCar - 1) / 2;
    int nTotal = sims * nPairs;
    int blockSize = 256;
    int gridSize = (nTotal + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 10; i++) {
        collisionKernel<<<gridSize, blockSize>>>(d_state);
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        collisionKernel<<<gridSize, blockSize>>>(d_state);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (ms / iterations) * 1000.0f;  // Return microseconds
}

// Access device state pointer from host GameState
__global__ void getDeviceStatePtr(GameState* hostState, GameState** outPtr) {
    *outPtr = hostState;
}

int main() {
    const int SIMS = 2048;
    const int CARS_PER_TEAM = 3;  // 3v3 = 6 cars total (more pairs = 15)
    const int TICKS_PER_SCENARIO = 20;
    const int ITERATIONS_PER_TICK = 100;

    printf("=== Rocket League Collision Scenario Benchmark ===\n");
    printf("Simulations: %d, Cars: %d (%dv%d)\n", SIMS, CARS_PER_TEAM * 2, CARS_PER_TEAM, CARS_PER_TEAM);
    printf("Ticks per scenario: %d, Iterations per tick: %d\n\n",
           TICKS_PER_SCENARIO, ITERATIONS_PER_TICK);

    // We need direct access to device state
    // Manually create a minimal GameState for benchmarking
    int nCar = CARS_PER_TEAM * 2;
    int totalCars = SIMS * nCar;

    // Host-side struct to set up pointers
    struct {
        int sims;
        int nCar;
        int numB;
        int numO;
        int seed;
        struct { float4 *position, *velocity, *angularV, *rotation; } ball;
        struct { float4 *position, *velocity, *angularV, *rotation; float* boost; } cars;
        struct { bool* isActive; } pads;
    } hostState;

    hostState.sims = SIMS;
    hostState.nCar = nCar;
    hostState.numB = CARS_PER_TEAM;
    hostState.numO = CARS_PER_TEAM;
    hostState.seed = 42;

    // Allocate car data arrays
    cudaMalloc(&hostState.cars.position, totalCars * sizeof(float4));
    cudaMalloc(&hostState.cars.rotation, totalCars * sizeof(float4));
    cudaMalloc(&hostState.cars.velocity, totalCars * sizeof(float4));
    cudaMalloc(&hostState.cars.angularV, totalCars * sizeof(float4));
    cudaMalloc(&hostState.cars.boost, totalCars * sizeof(float));

    // Ball and pads not needed for collision benchmark, set to null
    hostState.ball.position = nullptr;
    hostState.ball.velocity = nullptr;
    hostState.ball.angularV = nullptr;
    hostState.ball.rotation = nullptr;
    hostState.pads.isActive = nullptr;

    // Copy to device
    GameState* d_state;
    cudaMalloc(&d_state, sizeof(GameState));
    cudaMemcpy(d_state, &hostState, sizeof(GameState), cudaMemcpyHostToDevice);

    int posGrid = (SIMS + 255) / 256;

    printf("%-20s %10s %10s %10s %10s\n",
           "Scenario", "Min (us)", "Max (us)", "Avg (us)", "Std Dev");
    printf("%-20s %10s %10s %10s %10s\n",
           "--------", "--------", "--------", "--------", "--------");

    float scenarioAvgs[NUM_SCENARIOS];
    float globalMin = 1e9f, globalMax = 0;

    // Run each scenario
    for (int s = 0; s < NUM_SCENARIOS; s++) {
        Scenario scenario = (Scenario)s;

        float times[TICKS_PER_SCENARIO];
        float sum = 0, min = 1e9f, max = 0;

        for (int tick = 0; tick < TICKS_PER_SCENARIO; tick++) {
            // Set positions for this tick
            setScenarioPositions<<<posGrid, 256>>>(d_state, scenario, tick, 12345 + s * 1000);
            cudaDeviceSynchronize();

            // Time the collision kernel
            float us = timeCollisionKernel(d_state, SIMS, nCar, ITERATIONS_PER_TICK);
            times[tick] = us;
            sum += us;
            if (us < min) min = us;
            if (us > max) max = us;
        }

        float avg = sum / TICKS_PER_SCENARIO;
        scenarioAvgs[s] = avg;
        if (avg < globalMin) globalMin = avg;
        if (avg > globalMax) globalMax = avg;

        // Compute std dev
        float variance = 0;
        for (int t = 0; t < TICKS_PER_SCENARIO; t++) {
            float diff = times[t] - avg;
            variance += diff * diff;
        }
        float stddev = sqrtf(variance / TICKS_PER_SCENARIO);

        printf("%-20s %10.2f %10.2f %10.2f %10.2f\n",
               SCENARIO_NAMES[s], min, max, avg, stddev);
    }

    printf("\n=== Analysis ===\n");
    int nPairs = nCar * (nCar - 1) / 2;
    printf("Total collision pairs per tick: %d sims x %d pairs = %d\n", SIMS, nPairs, SIMS * nPairs);
    printf("\nExpected collision density:\n");
    printf("  SPREAD_ROTATION: ~0%%  (cars far apart)\n");
    printf("  KICKOFF:         ~5%%  (some adjacent pairs)\n");
    printf("  GOAL_SCRAMBLE:   ~20%% (moderate clustering)\n");
    printf("  BALL_CHASE:      ~40%% (all chasing ball)\n");
    printf("  CORNER_PLAY:     ~60%% (tight corner)\n");
    printf("\nPerformance range: %.1f us (sparse) to %.1f us (dense)\n", globalMin, globalMax);
    printf("Dense/Sparse ratio: %.2fx\n", globalMax / globalMin);

    // Cleanup
    cudaFree(hostState.cars.position);
    cudaFree(hostState.cars.rotation);
    cudaFree(hostState.cars.velocity);
    cudaFree(hostState.cars.angularV);
    cudaFree(hostState.cars.boost);
    cudaFree(d_state);

    return 0;
}
