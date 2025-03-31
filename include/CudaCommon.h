#pragma once

#ifdef __CUDACC__
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__
    #define CUDA_BOTH __host__ __device__
#else
    #define CUDA_HOST
    #define CUDA_DEVICE
    #define CUDA_BOTH
#endif