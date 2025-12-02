#pragma once

#include <cuda_runtime.h>
#include <iostream>

// General error checking for CUDA memory operations
inline void check(cudaError_t err, const char *const func,
                  const char *const file, const int line)
{
    if (err == cudaSuccess)
        return;

    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)