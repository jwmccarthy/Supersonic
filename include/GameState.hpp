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
    float* boost;
};

struct REFLECT Pads
{
    bool* isActive;
};

// Collision pair data (SoA layout for SAT results)
struct REFLECT Cols
{
    // SATContext fields
    float4* vecAB;      // Vector from A to B in A's local space
    float4* axB0;       // B's X axis in A's local space
    float4* axB1;       // B's Y axis in A's local space
    float4* axB2;       // B's Z axis in A's local space

    // SATResult fields
    float*  depth;      // Penetration depth
    float4* bestAx;     // Best separating axis
    int*    axisIdx;    // Which axis was best (0-14)
    bool*   overlap;    // Did boxes overlap?

    // Car indices
    int*    carA;       // Index of car A within simulation
    int*    carB;       // Index of car B within simulation
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
    Cols cols{};

    GameState(int sims, int numB, int numO, int seed);
    ~GameState();
};