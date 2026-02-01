#pragma once

#include <cuda_runtime.h>

struct CarSpawn
{
    float x, y, z, yaw;
};

__host__ __device__ constexpr float TICK = 1 / 120.0f;

__host__ __device__ constexpr float PI = 3.1415926535897932384626433832795029;
__host__ __device__ constexpr float PI_2 = PI / 2;
__host__ __device__ constexpr float PI_4 = PI / 4;

// Car-car collision constants
__host__ __device__ constexpr float CAR_CAR_FRICTION = 0.09f;
__host__ __device__ constexpr float CAR_CAR_RESTITUTION = 0.1f;
__host__ __device__ constexpr float BUMP_COOLDOWN = 0.25f;
__host__ __device__ constexpr float BUMP_MIN_DIST = 64.5f;
__host__ __device__ constexpr float SUPERSONIC_SPEED = 2200.0f;

// Body rest heights
__host__ __device__ constexpr float BALL_REST_Z = 93.15f;
__host__ __device__ constexpr float CAR_REST_Z  = 17.0f;

// Boost pad constants
__host__ __device__ constexpr int NUM_BOOST_PADS = 34;

// Car spawn locations
__host__ __device__ constexpr CarSpawn KICKOFF_LOCATIONS[5] = {
    { -2048, -2560, CAR_REST_Z, PI_4 * 1 },  // Right corner
    {  2048, -2560, CAR_REST_Z, PI_4 * 3 },  // Left corner
    {  -256, -3840, CAR_REST_Z, PI_4 * 2 },  // Back right
    {   256, -3840, CAR_REST_Z, PI_4 * 2 },  // Back Left
    {     0, -4608, CAR_REST_Z, PI_4 * 2 }   // Back center
};

__host__ __device__ constexpr CarSpawn RESPAWN_LOCATIONS[4] = {
    { -2304, -4608, CAR_REST_Z, PI_2 * 1 },  // Right inside
    { -2688, -4608, CAR_REST_Z, PI_2 * 1 },  // Right outside
    {  2304, -4608, CAR_REST_Z, PI_2 * 1 },  // Left inside
    {  2688, -4608, CAR_REST_Z, PI_2 * 1 }   // Left outside
};

extern __device__ __constant__ int KICKOFF_PERMUTATIONS[120][4];

// Dynamic rigid body masses
__host__ __device__ constexpr float CAR_MASS  = 180.0f;
__host__ __device__ constexpr float BALL_MASS = CAR_MASS / 6.0f;

// Inverse masses for convenience
__host__ __device__ constexpr float INV_CAR_MASS  = 1 / CAR_MASS;
__host__ __device__ constexpr float INV_BALL_MASS = 1 / BALL_MASS;

// Car inertia tensor (diagonal, Octane hitbox)
// Calculated via Bullet's btBoxShape::calculateLocalInertia formula:
// Ixx = m/12 * (ly² + lz²), Iyy = m/12 * (lx² + lz²), Izz = m/12 * (lx² + ly²)
__host__ __device__ constexpr float CAR_INERTIA_X = 135169.80f;
__host__ __device__ constexpr float CAR_INERTIA_Y = 240247.05f;
__host__ __device__ constexpr float CAR_INERTIA_Z = 330580.95f;

// Inverse inertia for convenience
__host__ __device__ constexpr float INV_CAR_INERTIA_X = 1 / CAR_INERTIA_X;
__host__ __device__ constexpr float INV_CAR_INERTIA_Y = 1 / CAR_INERTIA_Y;
__host__ __device__ constexpr float INV_CAR_INERTIA_Z = 1 / CAR_INERTIA_Z;

// Car dimensions/face locations in local space (Octane hitbox from RocketSim)
__host__ __device__ constexpr float4 CAR_EXTENTS = { 120.507f, 86.6994f,  38.6591f, 0.0f };
__host__ __device__ constexpr float4 CAR_HALF_EX = { 60.2535f, 43.3497f, 19.32955f, 0.0f };
__host__ __device__ constexpr float4 CAR_OFFSETS = { 13.8757f,     0.0f,   20.755f, 0.0f };

// World axis helpers
__host__ __device__ constexpr float4 WORLD_X = { 1, 0, 0, 0 };
__host__ __device__ constexpr float4 WORLD_Y = { 0, 1, 0, 0 };
__host__ __device__ constexpr float4 WORLD_Z = { 0, 0, 1, 0 };

// Arena grid & extents
__host__ __device__ constexpr int3   GRID_DIMS = { 48, 48, 16 };
__host__ __device__ constexpr float4 ARENA_MIN = { -6000.f, -4108.f,  -14.f, 0.f };
__host__ __device__ constexpr float4 ARENA_MAX = {  6000.f,  4108.f, 2076.f, 0.f };
__host__ __device__ constexpr float4 CELL_SIZE = {
    (ARENA_MAX.x - ARENA_MIN.x) / (float)GRID_DIMS.x,
    (ARENA_MAX.y - ARENA_MIN.y) / (float)GRID_DIMS.y,
    (ARENA_MAX.z - ARENA_MIN.z) / (float)GRID_DIMS.z,
    0.f
};