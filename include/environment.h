#pragma once
#include <torch/extension.h>

struct Environment {
    float state;
    int step_count;
    
#ifdef __CUDACC__
    __device__ void reset() {
        state = 0.0f;
        step_count = 0;
    }
    __device__ void step(float action) {
        state += action;
        step_count++;
    }
    __device__ float get_state() const {
        return state;
    }
#endif
};

void run_step(Environment* d_envs, torch::Tensor actions, torch::Tensor states);

void run_reset(Environment* d_envs, torch::Tensor states);