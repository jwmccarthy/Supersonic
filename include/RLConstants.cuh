#pragma once

#include <cuda_runtime.h>

#include "CudaCommon.cuh"

__device__ constexpr float PI = 3.1415926535897932384626433832795029;
__device__ constexpr float PI_2 = PI / 2;
__device__ constexpr float PI_4 = PI / 4;

__device__ constexpr float BALL_REST_Z = 92.75f;
__device__ constexpr float CAR_REST_Z  = 17.01f;

__device__ constexpr float SPAWN_BOOST_AMOUNT = 33.0f;

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
    { -2304, -4608, PI_2 },  // Right inside
    { -2688, -4608, PI_2 },  // Right outside
    {  2304, -4608, PI_2 },  // Left inside
    {  2688, -4608, PI_2 }   // Left outside
};

constexpr int NUM_WHEELS = 4;
constexpr int NUM_BOOST_PADS = 34;