#pragma once

struct CarControls
{
    // Driving controls
    float* gas;    // [-1, 1]
    float* steer;  // [-1, 1]

    // Car orientation
    float* pitch;  // [-1, 1]
    float* yaw;    // [-1, 1]
    float* roll;   // [-1, 1]

    // Digital inputs
    bool* jump;    // {0, 1}
    bool* boost;   // {0, 1}
    bool* brake;   // {0, 1}
};