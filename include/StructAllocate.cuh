#pragma once

#include <cuda_runtime.h>

#include "CudaCommon.cuh"
#include "Reflection.hpp"

template <class S>
void cudaMallocSOA(S& soa, int n)
{
    for (const auto& field : reflection::fields<S>)
    {
        if (field.isPointer)
        {
            CUDA_CHECK(cudaMalloc(&field.ref(soa), n * field.refSize));
        }
    }
}

template <class S>
void cudaFreeSOA(S& soa)
{
    for (const auto& field : reflection::fields<S>)
    {
        if (field.isPointer)
        {
            CUDA_CHECK(cudaFree(field.ref(soa)));
        }
    }
}