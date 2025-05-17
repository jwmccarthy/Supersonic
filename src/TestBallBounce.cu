#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <vector>

#include "GameState.cuh"
#include "GameStateDevice.cuh"
#include "CudaCommon.cuh"
#include "InitializerKernels.cuh"
#include "RLEnvironment.cuh"

// Physics constants for ball simulation
__device__ constexpr float GRAVITY = -650.0f;         // Gravity acceleration
__device__ constexpr float RESTITUTION = 0.6f;        // Bounciness factor
__device__ constexpr float GROUND_PLANE_Z = 0.0f;     // Ground level
__device__ constexpr float DRAG_COEFFICIENT = 0.03f;  // Drag coefficient

// Array of different gravity values for testing
__device__ float GRAVITY_VALUES[] = {
    -450.0f,   // Low gravity
    -650.0f,   // Default gravity (Rocket League)
    -850.0f,   // High gravity
    -1100.0f   // Very high gravity
};

// Ball physics update kernel
__global__ void updateBallPhysicsKernel(GameState* state, float dt) {
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->simCount) return;
    
    // Get current ball state
    float3 position = state->ballPosition[simIdx];
    float3 velocity = state->ballVelocity[simIdx];
    
    // Apply gravity to velocity
    velocity.z += GRAVITY * dt;
    
    // Apply simple drag
    velocity.x *= (1.0f - DRAG_COEFFICIENT * dt);
    velocity.y *= (1.0f - DRAG_COEFFICIENT * dt);
    velocity.z *= (1.0f - DRAG_COEFFICIENT * dt);
    
    // Update position based on velocity
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;
    
    // Check for ground collision
    if (position.z - BALL_REST_Z < GROUND_PLANE_Z) {
        // Place ball at ground level
        position.z = GROUND_PLANE_Z + BALL_REST_Z;
        
        // Bounce - reverse Z velocity and apply restitution
        if (velocity.z < 0) {
            velocity.z = -velocity.z * RESTITUTION;
            
            // Apply slight damping to horizontal velocity during bounce
            velocity.x *= 0.9f;
            velocity.y *= 0.9f;
        }
    }
    
    // Update ball state
    state->ballPosition[simIdx] = position;
    state->ballVelocity[simIdx] = velocity;
    
    // Update game time
    state->gameTime[simIdx] += dt;
}

int main() {
    try {
        std::cout << "== Ball Bounce Physics Test ==" << std::endl;
        
        // Initialize CUDA
        cudaFree(0);
        
        // Parameters
        const int simCount = 1024;  // Run multiple simulations in parallel
        const int blueCars = 0;    // No cars needed for this test
        const int orangeCars = 0;
        const uint64_t seed = 12345;
        const float initialHeight = 1000.0f; // Initial height (above rest Z)
        const float simulationTime = 5.0f;   // Simulate for 5 seconds
        
        std::cout << "Creating GameStateDevice with " << simCount << " simulations..." << std::endl;
        
        // First test GameStateDevice directly
        GameStateDevice state(simCount, blueCars, orangeCars, seed);
        
        std::cout << "Successfully created GameStateDevice" << std::endl;
        
        // Get the state pointer
        GameState* stateView = state.view();
        
        int blockSize = 64;
        int gridSize = (simCount + blockSize - 1) / blockSize;
        
        // Initialize RNG states
        seedKernel<<<gridSize, blockSize>>>(stateView, seed);
        cudaDeviceSynchronize();
        
        // Reset ball to initial position
        resetToKickoffKernel<<<gridSize, blockSize>>>(stateView);
        cudaDeviceSynchronize();
        
        // Set the balls to an initial height for better bounce visualization
        // Create a temporary buffer for all ball positions
        std::vector<float3> ballPositions(simCount);
        state.ballPosition.download(ballPositions.data());
        
        // Modify the heights and add some random x,y velocities to make them more varied
        for (int i = 0; i < simCount; i++) {
            ballPositions[i].z += initialHeight;
            
            // Vary the initial velocity for different simulations
            // but leave the first simulation with predictable initial velocity
            if (i > 0) {
                float xvel = ((i % 7) - 3) * 100.0f;
                float yvel = ((i % 5) - 2) * 100.0f;
                ballPositions[i].x += xvel * 0.1f; // Offset positions slightly
                ballPositions[i].y += yvel * 0.1f;
            }
        }
        
        // Upload back to device
        state.ballPosition.upload(ballPositions.data());
        
        // Set initial velocities
        std::vector<float3> ballVelocities(simCount, make_float3(0.0f, 0.0f, -100.0f, 0.0f));
        
        // Set various velocities for different simulations except the first one
        for (int i = 1; i < simCount; i++) {
            float xvel = ((i % 7) - 3) * 100.0f;
            float yvel = ((i % 5) - 2) * 100.0f;
            ballVelocities[i].x = xvel;
            ballVelocities[i].y = yvel;
        }
        
        state.ballVelocity.upload(ballVelocities.data());
        
        // Reset game time
        std::vector<float> gameTimes(simCount, 0.0f);
        state.gameTime.upload(gameTimes.data());
        
        std::cout << "\nStarting ball bounce simulation from height " 
                 << ballPositions[0].z << " for " << simulationTime << " seconds..." << std::endl;
        
        // Vector to store position data for output (only for the first simulation)
        std::vector<float3> positionHistory;
        
        // Main simulation loop
        const int stepsPerSecond = 120;  // 120 Hz physics
        const int totalSteps = static_cast<int>(simulationTime * stepsPerSecond);
        const float dt = 1.0f / stepsPerSecond;
        
        for (int step = 0; step <= totalSteps; step++) {
            // Run physics update for all simulations
            updateBallPhysicsKernel<<<gridSize, blockSize>>>(stateView, dt);
            
            // Check for errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Error in physics kernel: " 
                          << cudaGetErrorString(err) << std::endl;
                return 1;
            }
            
            // Synchronize to ensure physics is complete
            cudaDeviceSynchronize();
            
            // Download position every frame (only tracking the first simulation)
            if (step % 1 == 0) {
                // Download all ball positions
                std::vector<float3> allPositions(simCount);
                state.ballPosition.download(allPositions.data());
                
                // Only keep the first simulation's position
                positionHistory.push_back(allPositions[0]);
            }
        }
        
        // Output the position history for the first simulation
        std::cout << "\nBall position data (time, x, y, z):" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        
        for (size_t i = 0; i < positionHistory.size(); i++) {
            float time = i * 1 * dt; // Now sampling every frame
            std::cout << "[" << time << ", " << positionHistory[i].z << "]," << std::endl;
        }
        
        std::cout << "\nSimulation completed successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }
    
    // Clean up CUDA resources
    cudaDeviceReset();
    
    return 0;
}