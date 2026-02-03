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

    float*     d_output;

public:
    // Debug stats
    long long  debugTotalPairs = 0;
    long long  debugSatHits = 0;
    int sims;
    int numB;
    int numO;
    int cars;
    int seed;

    RLEnvironment(int sims, int numB, int numO, int seed);

    // Gymnasium API
    float* step();
    float* reset();

    // Debug
    void printSatStats();
    Workspace* getWorkspace() { return d_space; }
};