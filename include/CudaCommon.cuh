#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)

// General error checking for CUDA memory operations
inline void check(cudaError_t err, const char *const func,
                  const char *const file, const int line)
{
    if (err == cudaSuccess) return;
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
}

template<typename T>
inline void cudaMallocCpy(T*& d_ptr, const T* h_data, size_t n = 1)
{
    CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_ptr, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
}   