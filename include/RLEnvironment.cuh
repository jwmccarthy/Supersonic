#pragma once

#include <cuda_runtime.h>

#include "GameState.cuh"
#include "ArenaMesh.cuh"

class RLEnvironment
{
private:
    // Game physics state
    GameState  m_state;
    GameState* d_state;

    // Static arena mesh
    ArenaMesh  m_arena;
    ArenaMesh* d_arena;

    // Intermediate outputs
    Workspace  m_space;
    Workspace* d_space;

    // Buffers
    void*  d_cubBuf;
    size_t cubBytes;
    float* d_output;

    // Temp vars
    int m_broadTris;   // Total tris from broad phase (before AABB filter)
    int m_narrowTris;  // Total tris after AABB filter
    int m_maxBroad;    // Max allocation size for AABB filter arrays
    int m_nHit;        // Debug

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