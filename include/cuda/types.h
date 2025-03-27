#pragma once

#include <cuda_runtime.h>


struct alignas(16) CudaVec {
    float x, y, z, _w;

    __host__ __device__ CudaVec() : x(0), y(0), z(0), _w(0) {}
    __host__ __device__ CudaVec(float x, float y, float z, float _w = 0) : x(x), y(y), z(z), _w(_w) {}

    // Basic operations
    __host__ __device__ CudaVec operator+(const CudaVec& other) const;
    __host__ __device__ CudaVec operator-(const CudaVec& other) const;
    __host__ __device__ CudaVec operator*(const CudaVec& other) const;
    __host__ __device__ CudaVec operator/(const CudaVec& other) const;

    // In-place basic operations
    __host__ __device__ CudaVec operator+=(const CudaVec& other);
    __host__ __device__ CudaVec operator-=(const CudaVec& other);
    __host__ __device__ CudaVec operator*=(const CudaVec& other);
    __host__ __device__ CudaVec operator/=(const CudaVec& other);

    // Scalar operations
    __host__ __device__ CudaVec operator*(float scalar) const;
    __host__ __device__ CudaVec operator/(float scalar) const;

    // In-place scalar operations
    __host__ __device__ CudaVec operator*=(float scalar);
    __host__ __device__ CudaVec operator/=(float scalar);

    // Comparison operations
    __host__ __device__ bool operator==(const CudaVec& other) const;
    __host__ __device__ bool operator!=(const CudaVec& other) const;
    __host__ __device__ bool operator<(const CudaVec& other) const;
    __host__ __device__ bool operator>(const CudaVec& other) const;

    // Negation
    __host__ __device__ CudaVec operator-() const;

    // Indexing
    __host__ __device__ float& operator[](int index);
    __host__ __device__ float operator[](int index) const;

    // Helper functions
    bool IsZero() const;
    CudaVec To2D() const;

    // Vector properties
    float LengthSq() const;
    float Length() const;
    float LengthSq2D() const;
    float Length2D() const;
    CudaVec Normalized() const;

    // Vector operations
    float Dot(const CudaVec& other) const;
    float DistSq(const CudaVec& other) const;
    float Dist(const CudaVec& other) const;
    float DistSq2D(const CudaVec& other) const;
    float Dist2D(const CudaVec& other) const;
    CudaVec Cross(const CudaVec& other) const;
};