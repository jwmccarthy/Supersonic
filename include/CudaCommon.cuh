#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_H  __host__
#define CUDA_D  __device__
#define CUDA_HD __host__ __device__

// General error checking for CUDA memory operations
void check(cudaError_t err, const char* const func, const char* const file, 
           const int line) 
{
    if (err == cudaSuccess) return;

    std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)

// Helpers for CUDA type initialization
__device__ inline float4 zero4() { return make_float4(0, 0, 0, 0); }
