#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include "CudaCommon.cuh"
#include "CudaMath.cuh"
#include "BroadPhaseGrid.cuh"
#include "RLConstants.cuh"

// Define Rocket League car dimensions (approximated)
const float CAR_HALF_LENGTH = 59.5f; // Half length of car (X)
const float CAR_HALF_WIDTH = 42.0f;  // Half width of car (Y)
const float CAR_HALF_HEIGHT = 17.0f; // Half height of car (Z)

// Structure to hold test box information
struct TestBox {
    float4 position;
    float4 halfExtents;
    int triangleCount;
};

// Kernel to test triangle iteration with multiple car-sized boxes
__global__ void testMultipleBoxes(const BroadPhaseGrid* grid, TestBox* boxes, int numBoxes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes) return;

    TestBox& box = boxes[idx];
    
    // Calculate AABB for this box
    float4 aabbMin = box.position - box.halfExtents;
    float4 aabbMax = box.position + box.halfExtents;
    
    // Local counter to keep track of triangles processed
    int localCount = 0;
    
    // Call forEachTriangle with our AABB and a lambda to count triangles
    grid->forEachTriangle(aabbMin, aabbMax, [&](const Triangle& tri) {
        localCount++;
        
        // Debug output for the first few triangles for the first few boxes
        if (idx < 3 && localCount <= 3) {
            printf("Box %d - Triangle %d: v0(%f, %f, %f), v1(%f, %f, %f), v2(%f, %f, %f)\n",
                   idx, localCount,
                   tri.v0.x, tri.v0.y, tri.v0.z,
                   tri.v1.x, tri.v1.y, tri.v1.z,
                   tri.v2.x, tri.v2.y, tri.v2.z);
        }
    });
    
    // Store the count in the box
    box.triangleCount = localCount;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.obj>\n";
        return 1;
    }
    std::string meshPath = argv[1];
    
    std::cout << "Testing BroadPhaseGrid with mesh: " << meshPath << std::endl;
    
    // Create a BroadPhaseGrid with Rocket League arena parameters
    std::cout << "Creating grid..." << std::endl;
    
    BroadPhaseGrid* grid = new BroadPhaseGrid(
        meshPath, NUM_CELLS_X, NUM_CELLS_Y, NUM_CELLS_Z, 
        ARENA_HALF_EXTENT_X, ARENA_HALF_EXTENT_Y, ARENA_FULL_EXTENT_Z
    );
    
    std::cout << "Grid created. Setting up test boxes..." << std::endl;
    
    // Generate random car positions
    const int NUM_TEST_CARS = 20;
    std::vector<TestBox> testBoxes(NUM_TEST_CARS);
    
    // Setup random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(-ARENA_HALF_EXTENT_X * 0.9f, ARENA_HALF_EXTENT_X * 0.9f);
    std::uniform_real_distribution<float> distY(-ARENA_HALF_EXTENT_Y * 0.9f, ARENA_HALF_EXTENT_Y * 0.9f);
    std::uniform_real_distribution<float> distZ(CAR_REST_Z, ARENA_FULL_EXTENT_Z * 0.5f);
    
    // Create car-sized boxes at random positions
    for (int i = 0; i < NUM_TEST_CARS; i++) {
        testBoxes[i].position = make_float4(distX(gen), distY(gen), distZ(gen), 0.0f);
        testBoxes[i].halfExtents = make_float4(CAR_HALF_LENGTH, CAR_HALF_WIDTH, CAR_HALF_HEIGHT, 0.0f);
        testBoxes[i].triangleCount = 0;
    }
    
    std::cout << "Copying data to device..." << std::endl;
    
    // Allocate device memory for the grid and test boxes
    BroadPhaseGrid* d_grid;
    TestBox* d_testBoxes;
    
    CUDA_CHECK(cudaMalloc(&d_grid, sizeof(BroadPhaseGrid)));
    CUDA_CHECK(cudaMalloc(&d_testBoxes, sizeof(TestBox) * NUM_TEST_CARS));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_grid, grid, sizeof(BroadPhaseGrid), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_testBoxes, testBoxes.data(), sizeof(TestBox) * NUM_TEST_CARS, cudaMemcpyHostToDevice));
    
    std::cout << "Running spatial query test..." << std::endl;
    
    // Launch the test kernel with enough threads for all boxes
    int threadsPerBlock = 256;
    int blocksNeeded = (NUM_TEST_CARS + threadsPerBlock - 1) / threadsPerBlock;
    
    testMultipleBoxes<<<blocksNeeded, threadsPerBlock>>>(d_grid, d_testBoxes, NUM_TEST_CARS);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(testBoxes.data(), d_testBoxes, sizeof(TestBox) * NUM_TEST_CARS, cudaMemcpyDeviceToHost));
    
    // Output the results
    std::cout << "Spatial grid test complete with " << NUM_TEST_CARS << " car-sized boxes" << std::endl;
    std::cout << std::endl;
    std::cout << "Box positions and triangle counts:" << std::endl;
    std::cout << std::left << std::setw(5) << "Box" 
              << std::setw(12) << "Position X" 
              << std::setw(12) << "Position Y" 
              << std::setw(12) << "Position Z" 
              << std::setw(15) << "Triangle Count" << std::endl;
    
    // Print a separator line
    std::cout << std::string(60, '-') << std::endl;
    
    int totalTriangles = 0;
    for (int i = 0; i < NUM_TEST_CARS; i++) {
        std::cout << std::left << std::setw(5) << i 
                  << std::fixed << std::setprecision(2) 
                  << std::setw(12) << testBoxes[i].position.x 
                  << std::setw(12) << testBoxes[i].position.y 
                  << std::setw(12) << testBoxes[i].position.z 
                  << std::setw(15) << testBoxes[i].triangleCount << std::endl;
        
        totalTriangles += testBoxes[i].triangleCount;
    }
    
    std::cout << std::endl;
    std::cout << "Total triangles found across all boxes: " << totalTriangles << std::endl;
    std::cout << "Average triangles per box: " << static_cast<float>(totalTriangles) / NUM_TEST_CARS << std::endl;
    
    // Find the box with the most triangles
    auto maxBox = std::max_element(testBoxes.begin(), testBoxes.end(), 
        [](const TestBox& a, const TestBox& b) { return a.triangleCount < b.triangleCount; });
    
    int maxIdx = std::distance(testBoxes.begin(), maxBox);
    
    std::cout << "Box " << maxIdx << " had the most triangles: " << maxBox->triangleCount << std::endl;
    std::cout << "Position: (" << maxBox->position.x << ", " << maxBox->position.y << ", " 
              << maxBox->position.z << ")" << std::endl;
    
    // Clean up
    delete grid;
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_testBoxes));
    
    return 0;
}