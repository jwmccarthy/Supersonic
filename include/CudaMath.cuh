#pragma once

#include "CudaCommon.cuh"

template <typename T>
__host__ __device__ inline T clamp(T val, int low, int high) {
    return max(min(val, high), low);
}

// int4 arithmetic
__host__ __device__ inline int4 operator+(const int4 a, const int4 b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ inline int4 operator-(const int4 a, const int4 b) {
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ inline int4 operator*(const int4 a, const int4 b) {
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ inline int4 operator/(const int4 a, const int4 b) {
    return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

// float4 arithmetic
__host__ __device__ inline float4 operator+(const float4 a, const float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ inline float4 operator-(const float4 a, const float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ inline float4 operator*(const float4 a, const float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ inline float4 operator/(const float4 a, const float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

// int4 min/max
__host__ __device__ inline int4 min(const int4 a, const int4 b) {
    return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

__host__ __device__ inline int4 max(const int4 a, const int4 b) {
    return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

// float4 min/max
__host__ __device__ inline float4 fminf(const float4 a, const float4 b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

__host__ __device__ inline float4 fmaxf(const float4 a, const float4 b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}