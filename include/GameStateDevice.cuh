#pragma once

#include <cstdint>
#include <type_traits>

#include "DeviceArray.cuh"
#include "GameState.cuh"
#include "CudaCommon.cuh"

class GameStateDevice {   
private:
    int      m_simCount;
    int      m_carsPerSim;
    uint64_t m_randomSeed;

    GameState  h_view;
    GameState* d_view = nullptr;

public:
    GameStateDevice(int sims, int blues, int oranges, uint64_t seed);
    ~GameStateDevice();

    // Prevent copy operations
    GameStateDevice(const GameStateDevice&) = delete;
    GameStateDevice& operator=(const GameStateDevice&) = delete;
    
    // Allow move operations
    GameStateDevice(GameStateDevice&& other) noexcept;
    GameStateDevice& operator=(GameStateDevice&& other) noexcept;

    // Declare DeviceArray for each field
    #define DECLARE_FIELD(type, name, count) \
        DeviceArray<std::remove_pointer_t<type>> name;
    GAMESTATE_FIELDS(DECLARE_FIELD,,)
    #undef DECLARE_FIELD

    GameState* view() const { return d_view; };

    inline int getPhysicsStateLength() const {
        int totalFloats = 0;

        // Count floats in physics state vectors
        // Position, (angular) velocity, rotation
        totalFloats += m_simCount * 3 * 6;                 // Ball state
        totalFloats += m_simCount * m_carsPerSim * 3 * 6;  // Car states

        return totalFloats * m_simCount;
    }
};