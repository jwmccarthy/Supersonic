#pragma once

#include <cstdint>
#include <type_traits>

#include "DeviceArray.cuh"
#include "GameState.cuh"
#include "CudaCommon.cuh"

class GameStateDevice {   
private:
    int m_simCount;
    int m_numBlueCars;
    int m_numOrangeCars;
    int m_carsPerSim;
    uint64_t m_randomSeed;

    GameState h_view;
    GameState* d_view = nullptr;

public:
    GameStateDevice(int sims, int blues, int oranges, uint64_t seed);
    ~GameStateDevice();

    // Declare DeviceArray for each field
    #define DECLARE_FIELD(type, name, count) \
    DeviceArray<std::remove_pointer_t<type>> name;
    GAMESTATE_FIELDS(DECLARE_FIELD,,)
    #undef DECLARE_FIELD

    GameState* view();

    inline int getPhysicsStateLength() const {
        int totalFloats = 0;

        // Count floats in physics state vectors
        // Position, (angular) velocity, rotation
        totalFloats += m_simCount * 3 * 4;                 // Ball state
        totalFloats += m_simCount * m_carsPerSim * 3 * 4;  // Car states

        return totalFloats;
    }
};