#include "GameState.cuh"
#include "GameStateDevice.cuh"

GameStateDevice::GameStateDevice(int sims, int blues, int oranges) :
    m_simCount(sims),
    m_numBlueCars(blues),
    m_numOrangeCars(oranges),
    m_carsPerSim(blues + oranges)
{
    // Allocate memory for each DeviceArray
    #define ALLOCATE_FIELD(type, name, count) \
        name.allocate(count);
    GAMESTATE_FIELDS(ALLOCATE_FIELD, m_simCount, m_carsPerSim)
    #undef ALLOCATE_FIELD
}

GameState GameStateDevice::view() const {
    GameState s;
    s.simCount = m_simCount;
    s.numBlueCars = m_numBlueCars;
    s.numOrangeCars = m_numOrangeCars;
    s.carsPerSim = m_carsPerSim;

    // Set pointers to device array memory
    #define SETPTR_FIELD(type, name, count) \
        s.name = name.data();
    GAMESTATE_FIELDS(SETPTR_FIELD,,)
    #undef SETPTR_FIELD

    return s;
}