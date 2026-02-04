#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#include "RLEnvironment.cuh"

int main()
{
    using clock  = std::chrono::steady_clock;
    using second = std::chrono::duration<double>;

    const int sims = 1024;
    const int nCar = 4;
    const int seed = 111;
    const int iter = 10000;

    RLEnvironment env{sims, nCar, nCar, seed};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    env.reset();

    for (int i = 0; i < 1000; i++)
        env.step();
    cudaDeviceSynchronize();

    auto t0 = clock::now();
    cudaEventRecord(start);

    for (int i = 0; i < iter; i++)
    {
        env.step();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    auto t1 = clock::now();

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
