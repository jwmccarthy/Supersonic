#include "MathTypes.h"
#include "Environment.h"

__global__ void step_kernel(Environment* envs, float* actions, float* states, int num_envs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_envs) {
        // These __device__ functions are available because nvcc is compiling this file.
        envs[idx].step(actions[idx]);
        states[idx] = envs[idx].get_state();
    }
}

__global__ void reset_kernel(Environment* envs, float* states, int num_envs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_envs) {
        envs[idx].reset();
        states[idx] = envs[idx].get_state();
    }
}

void run_step(Environment* d_envs, torch::Tensor actions, torch::Tensor states) {
    int num_envs = actions.size(0);
    float* d_actions = actions.data_ptr<float>();
    float* d_states = states.data_ptr<float>();

    int threadsPerBlock = 256;
    int numBlocks = (num_envs + threadsPerBlock - 1) / threadsPerBlock;
    step_kernel<<<numBlocks, threadsPerBlock>>>(d_envs, d_actions, d_states, num_envs);
    cudaDeviceSynchronize();
}

void run_reset(Environment* d_envs, torch::Tensor states) {
    int num_envs = states.size(0);
    float* d_states = states.data_ptr<float>();

    int threadsPerBlock = 256;
    int numBlocks = (num_envs + threadsPerBlock - 1) / threadsPerBlock;
    reset_kernel<<<numBlocks, threadsPerBlock>>>(d_envs, d_states, num_envs);
    cudaDeviceSynchronize();
}