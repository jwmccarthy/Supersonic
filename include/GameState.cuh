#pragma once

#include <cuda_runtime.h>

#include "Reflection.hpp"

struct RigidBody
{
    float4* position;
    float4* velocity;
    float4* angularV;
    float4* rotation;
};

struct REFLECT Ball : RigidBody
{
};

struct REFLECT Cars : RigidBody
{
    int* numTris;
    float* boost;
};

struct REFLECT Pads
{
    bool* isActive;
};

struct REFLECT Workspace
{
    int*  numPairs;  // Number of car-triangle pairs for narrow phase
    int*  blockCnt;  // For grid sync in tail launch
    int2* pairs;     // Array of (carIdx, triIdx) pairs
};

struct GameState
{
    int sims;  // # of simulations
    int numB;  // # of blue cars
    int numO;  // # of orange cars
    int nCar;  // # of cars total
    int seed;  // For pseudo-RNG

    Ball ball{};
    Cars cars{};
    Pads pads{};

    GameState(int sims, int numB, int numO, int seed);
    ~GameState();
};
