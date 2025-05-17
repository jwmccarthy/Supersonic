#pragma once

#include <cuda_runtime.h>

#include "CudaCommon.cuh"

__device__ constexpr float PI = 3.1415926535897932384626433832795029;
__device__ constexpr float PI_2 = PI / 2;
__device__ constexpr float PI_4 = PI / 4;

__device__ constexpr float BALL_REST_Z = 93.15f;
__device__ constexpr float CAR_REST_Z  = 17.01f;

__device__ constexpr float SPAWN_BOOST_AMOUNT = 33.0f;

// === Car spawn location information ===

__device__ constexpr int NUM_KICKOFF_LOCATIONS = 5;
__device__ constexpr int NUM_RESPAWN_LOCATIONS = 4;

struct CarSpawn {
    float x, y, yaw;
};

__device__ constexpr CarSpawn KICKOFF_LOCATIONS[NUM_KICKOFF_LOCATIONS] = {
    //    x,     y,      yaw
    { -2048, -2560, PI_4 * 1 },  // Right corner
    {  2048, -2560, PI_4 * 3 },  // Left corner
    {  -256, -3840, PI_4 * 2 },  // Back right
    {   256, -3840, PI_4 * 2 },  // Back Left
    {     0, -4608, PI_4 * 2 }   // Back center
};

__device__ constexpr CarSpawn RESPAWN_LOCATIONS[NUM_RESPAWN_LOCATIONS] = {
    { -2304, -4608, PI_2 * 1 },  // Right inside
    { -2688, -4608, PI_2 * 1 },  // Right outside
    {  2304, -4608, PI_2 * 1 },  // Left inside
    {  2688, -4608, PI_2 * 1 }   // Left outside
};

// === Boost pad location information ===

constexpr int NUM_SMALL_BOOSTS = 28;
constexpr int NUM_LARGE_BOOSTS = 6;
constexpr int TOTAL_NUM_BOOSTS = NUM_SMALL_BOOSTS + NUM_LARGE_BOOSTS;

__device__ constexpr float3 SMALL_BOOST_LOCATIONS[NUM_SMALL_BOOSTS] = {
    {     0, -4240, 70, 0 },
    { -1792, -4184, 70, 0 },
    {  1792, -4184, 70, 0 },
    {  -940, -3308, 70, 0 },
    {   940, -3308, 70, 0 },
    {     0, -2816, 70, 0 },
    { -3584, -2484, 70, 0 },
    {  3584, -2484, 70, 0 },
    { -1788, -2300, 70, 0 },
    {  1788, -2300, 70, 0 },
    { -2048, -1036, 70, 0 },
    {     0, -1024, 70, 0 },
    {  2048, -1036, 70, 0 },
    { -1024,     0, 70, 0 },
    {  1024,     0, 70, 0 },
    { -2048,  1036, 70, 0 },
    {     0,  1024, 70, 0 },
    {  2048,  1036, 70, 0 },
    { -1788,  2300, 70, 0 },
    {  1788,  2300, 70, 0 },
    { -3584,  2484, 70, 0 },
    {  3584,  2484, 70, 0 },
    {     0,  2816, 70, 0 },
    {  -940,  3308, 70, 0 },
    {   940,  3308, 70, 0 },
    { -1792,  4184, 70, 0 },
    {  1792,  4184, 70, 0 },
    {     0,  4240, 70, 0 }
};

__device__ constexpr float3 LARGE_BOOST_LOCATIONS[NUM_LARGE_BOOSTS] = {
    { -3584,     0, 73, 0 },
    {  3584,     0, 73, 0 },
    { -3072,  4096, 73, 0 },
    {  3072,  4096, 73, 0 },
    { -3072, -4096, 73, 0 },
    {  3072, -4096, 73, 0 }
};

// === Car suspension information ===

constexpr int NUM_WHEELS = 4;