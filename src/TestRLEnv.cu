#include <iostream>
#include <cuda_runtime.h>

#include "GameState.cuh"
#include "GameStateDevice.cuh"
#include "CudaCommon.cuh"
#include "InitializerKernels.cuh"
#include "RLEnvironment.cuh"

// Simple print kernel to verify memory
__global__ void printStateKernel(GameState* state) {
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= 5) return;

    // Just print the first ball position
    printf("%d: Car position (from device): (%f, %f, %f)\n",
            carIdx, 
            state->carPosition[threadIdx.x].x,
            state->carPosition[threadIdx.x].y,
            state->carPosition[threadIdx.x].z
    );

    // Just print the first ball position
    printf("%d: Car rotation F (from device): (%f, %f, %f)\n",
            carIdx, 
            state->carRotationF[threadIdx.x].x,
            state->carRotationF[threadIdx.x].y,
            state->carRotationF[threadIdx.x].z
    );

    // Just print the first ball position
    printf("%d: Car rotation R (from device): (%f, %f, %f)\n",
            carIdx, 
            state->carRotationR[threadIdx.x].x,
            state->carRotationR[threadIdx.x].y,
            state->carRotationR[threadIdx.x].z
    );

    // Just print the first ball position
    printf("%d: Car rotation U (from device): (%f, %f, %f)\n",
            carIdx, 
            state->carRotationU[threadIdx.x].x,
            state->carRotationU[threadIdx.x].y,
            state->carRotationU[threadIdx.x].z
    );
}

int main() {
    try {
        std::cout << "== RLEnvironment Initialization Test ==" << std::endl;
        
        // Initialize CUDA
        cudaFree(0);
        
        // Parameters
        const int simCount = 2;  // Keep this small for testing
        const int blueCars = 3;
        const int orangeCars = 3;
        const uint64_t seed = 12345;
        
        std::cout << "Creating GameStateDevice..." << std::endl;
        
        // First test GameStateDevice directly
        GameStateDevice state(simCount, blueCars, orangeCars, seed);
        
        std::cout << "Successfully created GameStateDevice" << std::endl;
        std::cout << "Getting state view..." << std::endl;
        
        // Get the state pointer
        GameState* stateView = state.view();
        
        int blockSize = 64;
        int gridSize = (simCount + blockSize - 1) / blockSize;
        
        // Initialize RNG states
        seedKernel<<<gridSize, blockSize>>>(stateView, seed);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Error after seedKernel: " 
                      << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        // Synchronize
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Error synchronizing after seedKernel: " 
                      << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        std::cout << "Successfully initialized RNG states" << std::endl;
        
        // Reset to kickoff
        std::cout << "\nResetting to kickoff positions..." << std::endl;
        
        resetToKickoffKernel<<<gridSize, blockSize>>>(stateView);
        
        // Check for errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Error after resetToKickoffKernel: " 
                      << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        // Synchronize
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Error synchronizing after resetToKickoffKernel: " 
                      << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        std::cout << "Successfully reset to kickoff positions" << std::endl;
        
        // Print the first ball position from the device
        std::cout << "\nVerifying car position from device:" << std::endl;
        printStateKernel<<<gridSize, blockSize>>>(stateView);
        cudaDeviceSynchronize();
        
        // Read ball positions from device
        std::cout << "\nVerifying ball positions from host:" << std::endl;
        
        // We'll check a few of the ball positions
        float4 ballPositions[4];
        state.ballPosition.download(ballPositions);
        
        for (int i = 0; i < simCount; i++) {
            std::cout << "  Sim " << i << ": ("
                      << ballPositions[i].x << ", "
                      << ballPositions[i].y << ", "
                      << ballPositions[i].z << ")" << std::endl;
        }
        
        // Now try the RLEnvironment wrapper
        std::cout << "\n== Testing RLEnvironment wrapper ==" << std::endl;
        try {
            RLEnvironment env(2, blueCars, orangeCars, seed + 1);
            std::cout << "Successfully created RLEnvironment" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error creating RLEnvironment: " << e.what() << std::endl;
            return 1;
        }
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
    }
    catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }
    
    // Clean up CUDA resources
    cudaDeviceReset();
    
    return 0;
}