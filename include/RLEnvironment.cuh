#pragma once

#include <cuda_runtime.h>

#include "GameState.cuh"
#include "ArenaMesh.cuh"

class RLEnvironment
{
private:
    GameState  m_state;
    GameState* d_state;

    ArenaMesh  m_arena;
    ArenaMesh* d_arena;

    Workspace  m_space;
    Workspace* d_space;

    void*      d_cubTemp;
    size_t     cubTempBytes;
    float*     d_output;

public:
    int sims;
    int cars;
    int numB;
    int numO;
    int seed;

    RLEnvironment(int sims, int numB, int numO, int seed);

    // Gymnasium API
    float* step();
    float* reset();
};