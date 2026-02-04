#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#include "RLEnvironment.cuh"

int main(int argc, char** argv)
{
    using clock  = std::chrono::steady_clock;
    using second = std::chrono::duration<double>;

    int sims = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int nCar = (argc > 2) ? std::atoi(argv[2]) : 4;
    int seed = (argc > 3) ? std::atoi(argv[3]) : 111;
    int iter = (argc > 4) ? std::atoi(argv[4]) : 10000;

    std::cout << "sims=" << sims << " nCar=" << nCar << " seed=" << seed << " iter=" << iter << "\n";

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
