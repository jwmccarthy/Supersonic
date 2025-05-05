#pragma once

#include <cuda_runtime.h>

#include "CudaCommon.cuh"

template<typename T>
class DeviceArray {
public:
    DeviceArray() : m_ptr(nullptr), m_size(0), m_length(0) {}
    ~DeviceArray() { free(); }

    void allocate(size_t n) {
        free();
        m_length = n; 
        m_size = n * sizeof(T);

        // Allocate device memory and set to 0
        CUDA_CHECK(cudaMalloc(&m_ptr, m_size));
        CUDA_CHECK(cudaMemset(m_ptr, 0, m_size));
    }

    void upload(const T* src) {
        if (!m_ptr || !src) return;
        CUDA_CHECK(cudaMemcpy(m_ptr, src, m_size, cudaMemcpyHostToDevice));
    }

    void download(T* dst) {
        if (!m_ptr || !dst) return;
        CUDA_CHECK(cudaMemcpy(dst, m_ptr, m_size, cudaMemcpyDeviceToHost));
    }

    T*     data()         { return m_ptr; }
    T*     data()   const { return m_ptr; }
    size_t length() const { return m_length; }
    size_t size()   const { return m_size; }

private:
    T*     m_ptr;
    size_t m_length;
    size_t m_size;

    void free() {
        if (m_ptr) CUDA_CHECK(cudaFree(m_ptr));
        m_ptr = nullptr;
        m_length = 0;
        m_size = 0;
    }
};