#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include "CudaCommon.h"

struct alignas(16) CudaVec {
    float x, y, z, _w;

    // Constructors
    CUDA_BOTH CudaVec() : x(0), y(0), z(0), _w(0) {}
    CUDA_BOTH CudaVec(float x, float y, float z, float _w = 0) 
        : x(x), y(y), z(z), _w(_w) {}

    // Basic operations
    CUDA_BOTH CudaVec operator+(const CudaVec& other) const;
    CUDA_BOTH CudaVec operator-(const CudaVec& other) const;
    CUDA_BOTH CudaVec operator*(const CudaVec& other) const;
    CUDA_BOTH CudaVec operator/(const CudaVec& other) const;

    // In-place basic operations
    CUDA_BOTH CudaVec& operator+=(const CudaVec& other);
    CUDA_BOTH CudaVec& operator-=(const CudaVec& other);
    CUDA_BOTH CudaVec& operator*=(const CudaVec& other);
    CUDA_BOTH CudaVec& operator/=(const CudaVec& other);

    // Scalar operations
    CUDA_BOTH CudaVec operator*(float scalar) const;
    CUDA_BOTH CudaVec operator/(float scalar) const;

    // In-place scalar operations
    CUDA_BOTH CudaVec& operator*=(float scalar);
    CUDA_BOTH CudaVec& operator/=(float scalar);

    // Comparison operations
    CUDA_BOTH bool operator==(const CudaVec& other) const;
    CUDA_BOTH bool operator!=(const CudaVec& other) const;
    CUDA_BOTH bool operator<(const CudaVec& other) const;
    CUDA_BOTH bool operator>(const CudaVec& other) const;

    // Negation
    CUDA_BOTH CudaVec operator-() const;

    // Indexing
    CUDA_BOTH float& operator[](int index);
    CUDA_BOTH float operator[](int index) const;

    // Helper functions
    CUDA_BOTH bool IsZero() const;
    CUDA_BOTH CudaVec To2D() const;

    // Vector properties
    CUDA_BOTH float LengthSq() const;
    CUDA_BOTH float Length() const;
    CUDA_BOTH float LengthSq2D() const;
    CUDA_BOTH float Length2D() const;
    CUDA_BOTH CudaVec Normalized() const;

    // Vector functions
    CUDA_BOTH float Dot(const CudaVec& other) const;
    CUDA_BOTH float DistSq(const CudaVec& other) const;
    CUDA_BOTH float Dist(const CudaVec& other) const;
    CUDA_BOTH float DistSq2D(const CudaVec& other) const;
    CUDA_BOTH float Dist2D(const CudaVec& other) const;
    CUDA_BOTH CudaVec Cross(const CudaVec& other) const;
};

struct alignas(16) CudaRotMat {
    CudaVec f, r, u;

    CUDA_BOTH CudaRotMat();
    CUDA_BOTH CudaRotMat(CudaVec f, CudaVec r, CudaVec u) 
        : f(f), r(r), u(u) {}

    CUDA_BOTH static CudaRotMat GetIdentity();
    CUDA_BOTH static CudaRotMat LookAt(CudaVec _f, CudaVec _u);
    CUDA_BOTH static CudaRotMat EulerYPR(float y, float p, float r);

    // Basic operations
    CUDA_BOTH CudaRotMat operator+(const CudaRotMat& other) const;
    CUDA_BOTH CudaRotMat operator-(const CudaRotMat& other) const;

    // In-place basic operations
    CUDA_BOTH CudaRotMat& operator+=(const CudaRotMat& other);
    CUDA_BOTH CudaRotMat& operator-=(const CudaRotMat& other);

    // Scalar operations
    CUDA_BOTH CudaRotMat operator*(float scalar) const;
    CUDA_BOTH CudaRotMat operator/(float scalar) const;

    // In-place scalar operations
    CUDA_BOTH CudaRotMat& operator*=(float scalar);
    CUDA_BOTH CudaRotMat& operator/=(float scalar);

    // Comparison operations
    CUDA_BOTH bool operator==(const CudaRotMat& other) const;
    CUDA_BOTH bool operator!=(const CudaRotMat& other) const;

    // Indexing
    CUDA_BOTH CudaVec& operator[](int index);
    CUDA_BOTH CudaVec operator[](int index) const;

    // Vector/Matrix operations
    CUDA_BOTH CudaVec Dot(const CudaVec& vec) const;
    CUDA_BOTH CudaRotMat Dot(const CudaRotMat& other) const;
    CUDA_BOTH CudaRotMat Transpose() const;
};

struct alignas(16) CudaAngle {
    float yaw, pitch, roll;

    CUDA_BOTH CudaAngle(float yaw = 0, float pitch = 0, float roll = 0) 
        : yaw(yaw), pitch(pitch), roll(roll) {}
    
    // Rotation matrix operations
    CUDA_BOTH static CudaAngle FromRotMat(CudaRotMat& mat);
    CUDA_BOTH CudaRotMat ToRotMat() const;

    // Vector operations
    CUDA_BOTH static CudaAngle FromVec(const CudaVec& vec);
    CUDA_BOTH CudaVec GetForwardVec() const;

    // Angle operations
    CUDA_BOTH void NormalizeFix();  // Constrain angles
    CUDA_BOTH CudaAngle GetDeltaTo(const CudaAngle& other) const;

    // Basic operations
    CUDA_BOTH CudaAngle operator+(const CudaAngle& other) const;
    CUDA_BOTH CudaAngle operator-(const CudaAngle& other) const;

    // Comparison operations
    CUDA_BOTH bool operator==(const CudaAngle& other) const;
    CUDA_BOTH bool operator!=(const CudaAngle& other) const;
};

struct alignas(16) CudaTransform {
    CudaRotMat rotation;
    CudaVec translation;

    CUDA_BOTH CudaTransform() {}
    CUDA_BOTH CudaTransform(const CudaRotMat& r, const CudaVec& t = CudaVec(0, 0, 0))
        : rotation(r), translation(t) {}

    CUDA_BOTH static CudaTransform GetIdentity();

    // Transform compositions
    CUDA_BOTH CudaTransform operator*(const CudaTransform& other) const;
    CUDA_BOTH CudaTransform& operator*=(const CudaTransform& other);

    // Comparison operators
    CUDA_BOTH bool operator==(const CudaTransform& other) const;
    CUDA_BOTH bool operator!=(const CudaTransform& other) const;

    // Transforms
    CUDA_BOTH CudaTransform Inverse() const;
    CUDA_BOTH CudaVec InverseTransform(const CudaVec& vec) const;
};