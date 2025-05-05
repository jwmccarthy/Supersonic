// Vec3.h
#pragma once

#include <cuda_runtime.h>

#include "CudaCommon.cuh"

struct CUDA_HD __align__(16) Vec3 {
    float4 v;

    // Constructors
    CUDA_HD Vec3()                           : v{0, 0, 0, 0} {}
    CUDA_HD Vec3(float x, float y, float z)  : v{x, y, z, 0} {}
    CUDA_HD Vec3(const float3& f)            : v{f.x, f.y, f.z, 0} {}

    // Component accessors
    CUDA_HD float x() const { return v.x; }
    CUDA_HD float y() const { return v.y; }
    CUDA_HD float z() const { return v.z; }

    // Vector ops
    CUDA_HD Vec3 operator+(const Vec3& b) const {
        return Vec3{v.x + b.v.x, v.y + b.v.y, v.z + b.v.z};
    }

    CUDA_HD Vec3 operator-(const Vec3& b) const {
        return Vec3{v.x - b.v.x, v.y - b.v.y, v.z - b.v.z};
    }

    CUDA_HD Vec3& operator+=(const Vec3& b) {
        v.x += b.v.x;  
        v.y += b.v.y;  
        v.z += b.v.z;
        return *this;
    }

    CUDA_HD Vec3& operator-=(const Vec3& b) {
        v.x -= b.v.x;  
        v.y -= b.v.y;  
        v.z -= b.v.z;
        return *this;
    }

    // Scalar ops
    CUDA_HD Vec3 operator*(float s) const {
        return Vec3{v.x * s, v.y * s, v.z * s};
    }

    CUDA_HD Vec3& operator*=(float s) {
        v.x *= s;  
        v.y *= s;  
        v.z *= s;
        return *this;
    }

    // Vector functions
    CUDA_HD float dot(const Vec3& b) const {
        return v.x * b.v.x + v.y * b.v.y + v.z * b.v.z;
    }

    CUDA_HD Vec3 cross(const Vec3& b) const {
        return Vec3{
            v.y * b.v.z - v.z * b.v.y,
            v.z * b.v.x - v.x * b.v.z,
            v.x * b.v.y - v.y * b.v.x
        };
    }

    CUDA_HD float lengthSq() const {
        return dot(*this);
    }

    CUDA_HD float length() const {
        return sqrtf(lengthSq());
    }

    CUDA_HD Vec3 normalize() const {
        float len = length();
        if (len > __FLT_EPSILON__) {
            return (*this) * (1.0f / len);
        }
        return Vec3{1.0f, 0.0f, 0.0f};
    }

    CUDA_HD Vec3 absolute() const {
        return Vec3{ fabsf(v.x), fabsf(v.y), fabsf(v.z) };
    }

    CUDA_HD Vec3 min(const Vec3& b) const {
        return Vec3{ 
            fminf(v.x, b.v.x),
            fminf(v.y, b.v.y),
            fminf(v.z, b.v.z) 
        };
    }

    CUDA_HD Vec3 max(const Vec3& b) const {
        return Vec3{ 
            fmaxf(v.x, b.v.x),
            fmaxf(v.y, b.v.y),
            fmaxf(v.z, b.v.z) 
        };
    }

    // Comparisons
    CUDA_HD bool operator==(const Vec3& b) const {
        return v.x == b.v.x && v.y == b.v.y && v.z == b.v.z;
    }

    CUDA_HD bool operator!=(const Vec3& b) const {
        return !(*this == b);
    }
};
