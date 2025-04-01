#pragma once

#include "MathTypes.h"
#include <torch/extension.h>

struct Environment {
    float position;
    float velocity;
    float gravity;
    float ground_level;
    float restitution;
    int step_count;
    CudaVec test_vector;
    
#ifdef __CUDACC__
    __device__ void reset() {
        position = 10.0f;
        velocity = 0.0f;
        gravity = -0.5f;
        ground_level = 0.0f;
        restitution = 0.8f;
        step_count = 0;
        test_vector *= 0;
    }

    __device__ void step(float action) {
        CudaVec temp = test_vector.To2D();
        velocity += gravity + action + temp[0];
        position += velocity;

        if (position <= ground_level) {
            position = ground_level;
            velocity *= -restitution;
        }

        step_count++;
    }

    __device__ float get_state() const {
        return position;
    }
#endif
};

void run_step(Environment* d_envs, torch::Tensor actions, torch::Tensor states);

void run_reset(Environment* d_envs, torch::Tensor states);