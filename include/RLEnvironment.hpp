#pragma once

#include <cuda_runtime.h>

#include "GameState.hpp"

class RLEnvironment
{
private:
    GameState  m_state;
    GameState* d_state;
    ArenaMesh  m_arena;
    ArenaMesh* d_arena;
    float*     d_output;

public:
    int sims;
    int numB;
    int numO;
    int seed;

    RLEnvironment(int sims, int numB, int numO, int seed);

    // Gymnasium API
    float* step();
    float* reset();
};