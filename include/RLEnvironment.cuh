#pragma once

#include <cstdint>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include "GameState.cuh"
#include "GameStateDevice.cuh"

class RLEnvironment {
private:
    GameStateDevice m_state;
    torch::Tensor m_output;

public:
    RLEnvironment(int sims, int blues, int oranges, uint64_t seed);

    torch::Tensor& step(torch::Tensor actions);
    torch::Tensor& reset();
};