#include <torch/extension.h>

#include "InitializerKernels.cuh"
#include "RLEnvironment.cuh"

RLEnvironment::RLEnvironment(int sims, int blues, int oranges) :
    m_state(sims, blues, oranges)
{
    m_wasResetByUser = false;  // Require reset before first step

    // Set initial ball position

    // Set random initial car positions

    // Set car state

    // Activate boost pads

    // Calculate length of output buffer tensor
    int outputSize = m_state.getPhysicsStateLength();

    // Initialize output buffer tensor
    auto tensorOpts = torch::TensorOptions()
        .device(torch::kCUDA)
        .dtype(torch::kFloat32);
    m_output = torch::empty({outputSize}, tensorOpts);
}