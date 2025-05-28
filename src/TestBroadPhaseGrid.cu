#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "CudaCommon.cuh"
#include "CudaMath.cuh"
#include "BroadPhaseGrid.cuh"
#include "RLConstants.cuh"

// Simple kernel to test triangle iteration
__global__ void testTriangleIteration(const BroadPhaseGrid* grid, int* triangleCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return; // Only use one thread for this test

    // Define a test AABB that should intersect with some part of the mesh
    // We'll use a box at the center of the grid with some reasonable size
    float4 aabbMin = make_float4(3520.0f, 4544.0f, 0.0f, 0.0f);
    float4 aabbMax = make_float4(3520.0f, 4544.0f, 0.0f, 0.0f);
    
    // Local counter to keep track of triangles processed
    int localCount = 0;
    
    // Call forEachTriangle with our AABB and a lambda to count triangles
    grid->forEachTriangle(aabbMin, aabbMax, [&](const Triangle& tri) {        
        // Debug output for the first few triangles
        if (localCount++ <= 5) {
            printf("Triangle %d: v0(%f, %f, %f), v1(%f, %f, %f), v2(%f, %f, %f)\n",
                   localCount,
                   tri.v0.x, tri.v0.y, tri.v0.z,
                   tri.v1.x, tri.v1.y, tri.v1.z,
                   tri.v2.x, tri.v2.y, tri.v2.z);
        }
    });
    
    // Store the count in device memory
    *triangleCount = localCount;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.obj>\n";
        return 1;
    }
    std::string meshPath = argv[1];
    
    std::cout << "Testing BroadPhaseGrid with mesh: " << meshPath << std::endl;
    
    // Create a BroadPhaseGrid with reasonable parameters
    // Parameters: mesh path, cells in each dimension, arena extents
    std::cout << "Creating grid..." << std::endl;

    BroadPhaseGrid* grid = new BroadPhaseGrid(
        meshPath, NUM_CELLS_X, NUM_CELLS_Y, NUM_CELLS_Z, 
        ARENA_HALF_EXTENT_X, ARENA_HALF_EXTENT_Y, ARENA_FULL_EXTENT_Z
    );

    std::cout << "Grid created. Copying grid to device..." << std::endl;
    
    // Allocate device memory for the grid and counter
    BroadPhaseGrid* d_grid;
    int* d_triangleCount;
    int h_triangleCount = 0;
    
    CUDA_CHECK(cudaMalloc(&d_grid, sizeof(BroadPhaseGrid)));
    CUDA_CHECK(cudaMalloc(&d_triangleCount, sizeof(int)));
    
    // Copy grid to device
    CUDA_CHECK(cudaMemcpy(d_grid, grid, sizeof(BroadPhaseGrid), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_triangleCount, 0, sizeof(int)));
    
    std::cout << "Grid copied. Running test..." << std::endl;

    // Launch the test kernel with 1 thread (for simplicity)
    testTriangleIteration<<<1, 1>>>(d_grid, d_triangleCount);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(&h_triangleCount, d_triangleCount, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Output the results
    std::cout << "Spatial grid test complete!" << std::endl;
    std::cout << "Triangles found in test region: " << h_triangleCount << std::endl;
    
    // Clean up
    delete grid;
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_triangleCount));
    
    return 0;
}