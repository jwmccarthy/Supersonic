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

struct GameState
{
    int sims;  // # of simulations
    int nCar;  // # of cars total
    int numB;  // # of blue cars
    int numO;  // # of orange cars
    int seed;  // For pseudo-RNG

    Ball ball{};
    Cars cars{};
    Pads pads{};

    GameState(int sims, int numB, int numO, int seed);
    ~GameState();
};
