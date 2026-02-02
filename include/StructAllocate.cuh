#pragma once

#include <cassert>
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
void cudaMallocSOA(S& soa, std::initializer_list<int> sizes)
{
    auto it = sizes.begin();

    for (const auto& field : reflection::fields<S>)
    {
        if (field.isPointer)
        {
            assert(it != sizes.end() && "Not enough sizes provided");
            CUDA_CHECK(cudaMalloc(&field.ref(soa), (*it++) * field.refSize));
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