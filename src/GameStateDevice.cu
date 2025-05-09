#include "GameStateDevice.cuh"

GameStateDevice::GameStateDevice(int sims, int blues, int oranges, uint64_t seed) :
    m_simCount(sims),
    m_carsPerSim(blues + oranges),
    m_randomSeed(seed)
{
    // Allocate memory for each DeviceArray
    #define ALLOCATE_FIELD(type, name, count) \
        name.allocate(count);
    GAMESTATE_FIELDS(ALLOCATE_FIELD, m_simCount, m_carsPerSim)
    #undef ALLOCATE_FIELD

    // Create POD view of state
    h_view.simCount = m_simCount;
    h_view.numBlueCars = blues;
    h_view.numOrangeCars = oranges;
    h_view.carsPerSim = m_carsPerSim;
    h_view.randomSeed = m_randomSeed;

    // Set pointers to device array memory
    #define SETPTR_FIELD(type, name, count) \
        h_view.name = name.data();
    GAMESTATE_FIELDS(SETPTR_FIELD,,)
    #undef SETPTR_FIELD

    // Allocate memory for view on device
    CUDA_CHECK(cudaMalloc(&d_view, sizeof(GameState)));

    // Copy the host staging struct to device
    CUDA_CHECK(cudaMemcpy(d_view, &h_view, sizeof(GameState), cudaMemcpyHostToDevice));
}

GameStateDevice::~GameStateDevice() {
    if (d_view) CUDA_CHECK(cudaFree(d_view));
}