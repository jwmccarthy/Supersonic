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

    // CUDA graph for step()
    cudaGraph_t     m_stepGraph;
    cudaGraphExec_t m_stepGraphExec;
    cudaStream_t    m_stream;

public:
    int sims;
    int numB;
    int numO;
    int cars;
    int seed;

    RLEnvironment(int sims, int numB, int numO, int seed);
    ~RLEnvironment();

    // Gymnasium API
    float* step();
    float* reset();
};