#pragma once

#include <cuda_runtime.h>

struct CarSpawn 
{
    float x, y, yaw;
};

__device__ constexpr float TICK = 1 / 120.0f;

__device__ constexpr float PI = 3.1415926535897932384626433832795029;
__device__ constexpr float PI_2 = PI / 2;
__device__ constexpr float PI_4 = PI / 4;

// Car-car collision constants
__device__ constexpr float CAR_CAR_FRICTION = 0.09f;
__device__ constexpr float CAR_CAR_RESTITUTION = 0.1f;
__device__ constexpr float BUMP_COOLDOWN = 0.25f;
__device__ constexpr float BUMP_MIN_DIST = 64.5f;
__device__ constexpr float SUPERSONIC_SPEED = 2200.0f;

// Body rest heights
__device__ constexpr float BALL_REST_Z = 93.15f;
__device__ constexpr float CAR_REST_Z  = 17.01f;

// Boost pad constants
__device__ constexpr int NUM_BOOST_PADS = 34;

// Car spawn locations
__device__ constexpr CarSpawn KICKOFF_LOCATIONS[5] = {
    {  2048, -2440, PI_4 * 1 },  // Right corner
    {  2048, -2560, PI_4 * 3 },  // Left corner
    {  -256, -3840, PI_4 * 2 },  // Back right
    {   256, -3840, PI_4 * 2 },  // Back Left
    {     0, -4608, PI_4 * 2 }   // Back center
};

__device__ constexpr CarSpawn RESPAWN_LOCATIONS[4] = {
    { -2304, -4608, PI_2 * 1 },  // Right inside
    { -2688, -4608, PI_2 * 1 },  // Right outside
    {  2304, -4608, PI_2 * 1 },  // Left inside
    {  2688, -4608, PI_2 * 1 }   // Left outside
};

extern __device__ __constant__ int KICKOFF_PERMUTATIONS[120][4];

// Dynamic rigid body masses
__device__ constexpr float CAR_MASS  = 180.0f;
__device__ constexpr float BALL_MASS = CAR_MASS / 6.0f;

// Inverse masses for convenience
__device__ constexpr float INV_CAR_MASS  = 1 / CAR_MASS;
__device__ constexpr float INV_BALL_MASS = 1 / BALL_MASS;

// Car dimensions/face locations in local space
__device__ constexpr float4 CAR_EXTENTS = {  118, 84.2,  36.2, 0};
__device__ constexpr float4 CAR_HALF_EX = {   59, 42.1,  18.1, 0};
__device__ constexpr float4 CAR_OFFSETS = {13.87,    0, 20.75, 0};

// Car half extents array
__device__ constexpr float CAR_HALF_EX_ARR[3] = { CAR_HALF_EX.x, CAR_HALF_EX.y, CAR_HALF_EX.z };

// World axis helpers
__device__ constexpr float4 WORLD_X = {1, 0, 0, 0};
__device__ constexpr float4 WORLD_Y = {0, 1, 0, 0};
__device__ constexpr float4 WORLD_Z = {0, 0, 1, 0};

// World axis array
__device__ constexpr float4 WORLD_AXES[3] = { WORLD_X, WORLD_Y, WORLD_Z };

// Bounding sphere for broad-phase
__device__ constexpr float CAR_BOUNDING_RADIUS = 74.5f;
__device__ constexpr float CAR_BOUNDING_RADIUS_SQ = CAR_BOUNDING_RADIUS * CAR_BOUNDING_RADIUS;
__device__ constexpr float CAR_PAIR_MAX_DIST_SQ = 4.0f * CAR_BOUNDING_RADIUS_SQ;

// SAT face axis bias
__device__ constexpr float SAT_FUDGE = 1.05f;