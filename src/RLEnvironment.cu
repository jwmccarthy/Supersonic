#include "InitializerKernels.cuh"
#include "RLEnvironment.cuh"

RLEnvironment::RLEnvironment(int sims, int blues, int oranges, uint64_t seed) :
    m_state(sims, blues, oranges, seed)
{
    // Set grid for initialization kernels
    int blockSize = 256;
    int gridSize = (m_state.view()->simCount + blockSize - 1) / blockSize;

    // Initialize random seed
    seedKernel<<<gridSize, blockSize>>>(m_state.view(), seed);
    cudaDeviceSynchronize();

    // Initialize ball and cars at kickoff locations
    resetToKickoffKernel<<<gridSize, blockSize>>>(m_state.view());
    cudaDeviceSynchronize();

    // Activate boost pads

    // Calculate length of output buffer tensor
    int outputSize = m_state.getPhysicsStateLength();

    // Initialize output buffer tensor
    auto tensorOpts = torch::TensorOptions()
        .device(torch::kCUDA)
        .dtype(torch::kFloat32);
    m_output = torch::empty({outputSize}, tensorOpts);
}