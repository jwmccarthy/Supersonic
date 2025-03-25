#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>  // Needed for cudaMalloc, cudaMemset, cudaFree
#include "environment.h"

namespace py = pybind11;

class EnvironmentManager {
public:
    Environment* d_envs;
    int num_envs;
    torch::Tensor states;

    EnvironmentManager(int n) : num_envs(n) {
        // Allocate memory on the GPU.
        cudaMalloc(&d_envs, num_envs * sizeof(Environment));
        cudaMemset(d_envs, 0, num_envs * sizeof(Environment));

        auto tensor_opts = torch::TensorOptions()
            .device(torch::kCUDA)
            .dtype(torch::kFloat32);
        states = torch::empty({num_envs}, tensor_opts);
    }

    ~EnvironmentManager() {
        cudaFree(d_envs);
    }

    void step(torch::Tensor actions) {
        run_step(d_envs, actions, states);
    }

    torch::Tensor get_states() {
        return states;
    }
};

PYBIND11_MODULE(supersonic, m) {
    py::class_<EnvironmentManager>(m, "EnvironmentManager")
        .def(py::init<int>())
        .def("step", &EnvironmentManager::step)
        .def("get_states", &EnvironmentManager::get_states);
}
