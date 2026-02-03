#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#include "RLEnvironment.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        return 1; \
    } \
} while(0)

int main()
{
    using clock  = std::chrono::steady_clock;
    using second = std::chrono::duration<double>;

    const int sims = 1024;
    const int nCar = 4;
    const int seed = 111;
    const int iter = 10000;

    std::cout << "Creating environment..." << std::flush;
    RLEnvironment env{sims, nCar, nCar, seed};
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << " done" << std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Reset..." << std::flush;
    env.reset();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << " done" << std::endl;

    std::cout << "Warmup..." << std::flush;

    for (int i = 0; i < 1000; i++)
        env.step();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << " done" << std::endl;

    std::cout << "Running " << iter << " iterations..." << std::flush;
    auto t0 = clock::now();
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iter; i++)
    {
        env.step();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    auto t1 = clock::now();
    std::cout << " done" << std::endl;

    float gpuMs = 0;
    cudaEventElapsedTime(&gpuMs, start, stop);

    second wallTime = t1 - t0;
    float avgUs = (gpuMs * 1000.0f) / iter;

    std::cout << "Iterations:    " << iter << "\n";
    std::cout << "Wall time:     " << wallTime.count() << " s\n";
    std::cout << "GPU time:      " << gpuMs << " ms\n";
    std::cout << "Avg per step:  " << avgUs << " us\n";
    std::cout << "Steps/sec:     " << (iter / wallTime.count()) << "\n";
    std::cout << "Sim-steps/sec: " << (iter * sims / wallTime.count() / 1e6) << " M\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
