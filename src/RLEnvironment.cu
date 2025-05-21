#include "InitializerKernels.cuh"
#include "RLEnvironment.cuh"

RLEnvironment::RLEnvironment(int sims, int blues, int oranges, uint64_t seed) :
    m_state(sims, blues, oranges, seed),
    m_stream(c10::cuda::getCurrentCUDAStream())
{
    // Set grid for initialization kernels
    int blockSize = 256;
    int gridSize = (sims + blockSize - 1) / blockSize;

    // Get POD game state struct
    GameState* d_state = m_state.view();

    // Initialize random seed
    printf("Initializing random states...\n");
    seedKernel<<<gridSize, blockSize, 0, m_stream>>>(d_state, seed);
    CUDA_CHECK(cudaGetLastError());

    // Initialize ball and cars at kickoff locations
    printf("Initializing initial car & ball states...\n");
    resetToKickoffKernel<<<gridSize, blockSize, 0, m_stream>>>(d_state);
    CUDA_CHECK(cudaGetLastError());

    // Activate boost pads
    printf("Activating boost pads...\n");
    m_state.boostPadIsActive.setValue(true, m_stream);

    // Calculate length of output buffer tensor
    int outputSize = m_state.getPhysicsStateLength();

    // Initialize output buffer tensor
    printf("Initializing tensor output buffer...\n");
    auto tensorOpts = torch::TensorOptions()
        .device(torch::kCUDA)
        .dtype(torch::kFloat32);
    m_output = torch::empty({outputSize}, tensorOpts);
    m_output.record_stream(m_stream);

    printf("Successfully initialized RLEnvironment\n");
}

torch::Tensor step(torch::Tensor actions) {
    return torch::empty({0});
}

torch::Tensor reset() {
    return torch::empty({0});
}