#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#include "RLEnvironment.cuh"

int main()
{
    using clock  = std::chrono::steady_clock;
    using second = std::chrono::duration<double>;

    const long sims = 1024;
    const int nCar  = 4;
    const int seed  = 123;

    RLEnvironment env{sims, nCar, nCar, seed};

    // CUDA events for kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    env.reset();

    // Warmup
    for (int i = 0; i < 100; i++)
        env.step();
    cudaDeviceSynchronize();

    // Timed run
    const int iterations = 100000;

    auto t0 = clock::now();
    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++)
    {
        env.step();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    auto t1 = clock::now();

    // Results
    float gpuMs = 0;
    cudaEventElapsedTime(&gpuMs, start, stop);

    second wallTime = t1 - t0;
    float avgUs = (gpuMs * 1000.0f) / iterations;

    std::cout << "Iterations:    " << iterations << "\n";
    std::cout << "Wall time:     " << wallTime.count() << " s\n";
    std::cout << "GPU time:      " << gpuMs << " ms\n";
    std::cout << "Avg per step:  " << avgUs << " us\n";
    std::cout << "Steps/sec:     " << (iterations / wallTime.count()) << "\n";
    std::cout << "Sim-steps/sec: " << (iterations * sims / wallTime.count() / 1e6) << " M\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
