#include <iostream>
#include <cuda_runtime.h>

#include "GameState.cuh"
#include "GameStateDevice.cuh"
#include "CudaCommon.cuh"
#include "InitializerKernels.cuh"
#include "RLEnvironment.cuh"

// Simple print kernel to verify memory
__global__ void printStateKernel(GameState* state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Just print the first ball position
        printf("Ball position (from device): (%f, %f, %f)\n", 
               state->ballPosition[0].x,
               state->ballPosition[0].y,
               state->ballPosition[0].z);
    }
}

int main() {
    try {
        std::cout << "== RLEnvironment Initialization Test ==" << std::endl;
        
        // Initialize CUDA
        cudaFree(0);
        
        // Parameters
        const int simCount = 4;  // Keep this small for testing
        const int blueCars = 1;
        const int orangeCars = 1;
        const uint64_t seed = 12345;
        
        std::cout << "Creating GameStateDevice..." << std::endl;
        
        // First test GameStateDevice directly
        GameStateDevice state(simCount, blueCars, orangeCars, seed);
        
        std::cout << "Successfully created GameStateDevice" << std::endl;
        std::cout << "Getting state view..." << std::endl;
        
        // Get the state pointer
        GameState* stateView = state.view();
        
        // Verify basic properties
        std::cout << "State properties: " << std::endl;
        std::cout << "  Sim count: " << stateView->simCount << std::endl;
        std::cout << "  Blue cars: " << stateView->numBlueCars << std::endl;
        std::cout << "  Orange cars: " << stateView->numOrangeCars << std::endl;
        
        // Manually launch the kernels
        std::cout << "\nInitializing random seeds..." << std::endl;
        
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
        std::cout << "\nVerifying ball position from device:" << std::endl;
        printStateKernel<<<1, 1>>>(stateView);
        cudaDeviceSynchronize();
        
        // Read ball positions from device
        std::cout << "\nVerifying ball positions from host:" << std::endl;
        
        // We'll check a few of the ball positions
        float4 ballPositions[4];
        cudaMemcpy(ballPositions, stateView->ballPosition, 
                   simCount * sizeof(float4), cudaMemcpyDeviceToHost);
        
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