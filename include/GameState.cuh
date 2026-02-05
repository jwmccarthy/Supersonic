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
    int*  numHit;   // Debug: total AABB hits
    int*  numTri;   // Per-car triangle counts (from overlapping groups)
    int*  triOff;   // Prefix sum of triCounts for thread mapping
    int4* groupIdx; // Per-car group index (w unused, for alignment)
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
