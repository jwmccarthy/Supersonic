#pragma once

#include <cuda_runtime.h>

#include "CudaCommon.cuh"

CUDA_HD constexpr float PI = 3.1415926535897932384626433832795029;
CUDA_HD constexpr float PI_2 = PI / 2;
CUDA_HD constexpr float PI_4 = PI / 4;

CUDA_HD constexpr float BALL_REST_Z = 92.75f;
CUDA_HD constexpr float CAR_REST_Z  = 17.01f;

CUDA_HD constexpr float SPAWN_BOOST_AMOUNT = 33.0f;

CUDA_HD constexpr int NUM_KICKOFF_LOCATIONS = 5;
CUDA_HD constexpr int NUM_RESPAWN_LOCATIONS = 4;

struct CarSpawn {
    float x, y, yaw;
};

CUDA_HD constexpr CarSpawn KICKOFF_LOCATIONS[NUM_KICKOFF_LOCATIONS] = {
    //    x,     y,      yaw
    { -2048, -2560, PI_4 * 1 },  // Right corner
    {  2048, -2560, PI_4 * 3 },  // Left corner
    {  -256, -3840, PI_4 * 2 },  // Back right
    {   256, -3840, PI_4 * 2 },  // Back Left
    {     0, -4608, PI_4 * 2 }   // Back center
};

CUDA_HD constexpr CarSpawn RESPAWN_LOCATIONS[NUM_RESPAWN_LOCATIONS] = {
    { -2304, -4608, PI_2 },  // Right inside
    { -2688, -4608, PI_2 },  // Right outside
    {  2304, -4608, PI_2 },  // Left inside
    {  2688, -4608, PI_2 }   // Left outside
};

constexpr int NUM_WHEELS = 4;
constexpr int NUM_BOOST_PADS = 34;