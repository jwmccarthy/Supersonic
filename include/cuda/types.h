#pragma once

#include <cuda_runtime.h>


struct alignas(16) CudaVec {
    float x, y, z, _w;

    // Constructors
    __host__ __device__ CudaVec() : x(0), y(0), z(0), _w(0) {}
    __host__ __device__ CudaVec(float x, float y, float z, float _w = 0) 
        : x(x), y(y), z(z), _w(_w) {}

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
    __host__ __device__ bool IsZero() const;
    __host__ __device__ CudaVec To2D() const;

    // Vector properties
    __host__ __device__ float LengthSq() const;
    __host__ __device__ float Length() const;
    __host__ __device__ float LengthSq2D() const;
    __host__ __device__ float Length2D() const;
    __host__ __device__ CudaVec Normalized() const;

    // Vector operations
    __host__ __device__ float Dot(const CudaVec& other) const;
    __host__ __device__ float DistSq(const CudaVec& other) const;
    __host__ __device__ float Dist(const CudaVec& other) const;
    __host__ __device__ float DistSq2D(const CudaVec& other) const;
    __host__ __device__ float Dist2D(const CudaVec& other) const;
    __host__ __device__ CudaVec Cross(const CudaVec& other) const;
};


struct alignas(16) CudaRotMat {
    CudaVec f, r, u;

    __host__ __device__ CudaRotMat()
    __host__ __device__ CudaRotMat(CudaVec f, CudaVec r, CudaVec u) 
        : f(f), r(r), u(u) {}

    __host__ __device__ static CudaRotMat GetIdentity();
    __host__ __device__ static CudaRotMat LookAt();

    // Basic operations
    __host__ __device__ CudaRotMat operator+(const CudaRotMat& other) const;
    __host__ __device__ CudaRotMat operator-(const CudaRotMat& other) const;
    __host__ __device__ CudaRotMat operator*(const CudaRotMat& other) const;
    __host__ __device__ CudaRotMat operator/(const CudaRotMat& other) const;

    // In-place basic operations
    __host__ __device__ CudaRotMat operator+=(const CudaRotMat& other);
    __host__ __device__ CudaRotMat operator-=(const CudaRotMat& other);
    __host__ __device__ CudaRotMat operator*=(const CudaRotMat& other);
    __host__ __device__ CudaRotMat operator/=(const CudaRotMat& other);

    // Scalar operations
    __host__ __device__ CudaRotMat operator*(float scalar) const;
    __host__ __device__ CudaRotMat operator/(float scalar) const;

    // In-place scalar operations
    __host__ __device__ CudaRotMat operator*=(float scalar);
    __host__ __device__ CudaRotMat operator/=(float scalar);

    // Comparison operations
    __host__ __device__ bool operator==(const CudaRotMat& other) const;
    __host__ __device__ bool operator!=(const CudaRotMat& other) const;

    // Indexing
    __host__ __device__ CudaVec& operator[](int index);
    __host__ __device__ CudaVec operator[](int index) const;

    // Vector/Matrix operations
    __host__ __device__ CudaVec Dot(const CudaVec& vec) const;
    __host__ __device__ CudaRotMat Dot(const CudaRotMat& other) const;
    __host__ __device__ CudaRotMat Transpose() const;
};


struct CudaAngle {
    float yaw, pitch, roll;

    __host__ __device__ CudaAngle(float yaw = 0, float pitch = 0, float roll = 0) 
        : yaw(yaw), pitch(pitch), roll(roll) {}
    
    // Rotation matrix operations
    __host__ __device__ static CudaAngle FromRotMat(CudaRotMat& mat);
    __host__ __device__ CudaRotMat ToRotMat() const;

    // Vector operations
    __host__ __device__ static CudaAngle FromVec(const CudaVec& vec);
    __host__ __device__ CudaVec GetForwardVec() const;

    // Angle operations
    __host__ __device__ void NormalizeFix();  // Constrain angles
    __host__ __device__ CudaAngle GetDeltaTo(const CudaAngle& other) const;

    // Basic operations
    __host__ __device__ CudaAngle operator+(const CudaAngle& other) const;
    __host__ __device__ CudaAngle operator-(const CudaAngle& other) const;

    // Comparison operations
    __host__ __device__ bool operator==(const CudaAngle& other) const;
    __host__ __device__ bool operator!=(const CudaAngle& other) const;
};