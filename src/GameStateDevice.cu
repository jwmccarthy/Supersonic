#include "GameStateDevice.cuh"

GameStateDevice::GameStateDevice(int sims, int blues, int oranges, uint64_t seed) :
    m_simCount(sims),
    m_numBlueCars(blues),
    m_numOrangeCars(oranges),
    m_carsPerSim(blues + oranges),
    m_randomSeed(seed)
{
    // Allocate memory for each DeviceArray
    #define ALLOCATE_FIELD(type, name, count) \
        name.allocate(count);
    GAMESTATE_FIELDS(ALLOCATE_FIELD, m_simCount, m_carsPerSim)
    #undef ALLOCATE_FIELD

    // Create POD view of state
    m_view.simCount = m_simCount;
    m_view.numBlueCars = m_numBlueCars;
    m_view.numOrangeCars = m_numOrangeCars;
    m_view.carsPerSim = m_carsPerSim;
    m_view.randomSeed = m_randomSeed;

    // Set pointers to device array memory
    #define SETPTR_FIELD(type, name, count) \
    m_view.name = name.data();
    GAMESTATE_FIELDS(SETPTR_FIELD,,)
    #undef SETPTR_FIELD
}

GameState* GameStateDevice::view() {
    return &m_view;
}