#pragma once

#include <cmath>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define rsqrtf(x) (1.0f / sqrtf(x))
#endif

__host__ __device__ __forceinline__ float sign(float x)
{
    return (x >= 0.0f) ? 1.0f : -1.0f;
}

// Scalar min/max that work on both host and device
template<typename T>
__host__ __device__ __forceinline__ T min(T a, T b)
{
    return a < b ? a : b;
}

template<typename T>
__host__ __device__ __forceinline__ T max(T a, T b)
{
    return a > b ? a : b;
}

namespace vec3
{
    // ===== Templated functions (float3, int3) =====

    template<typename T>
    __host__ __device__ __forceinline__ T add(T a, T b)
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T sub(T a, T b)
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T div(T a, T b)
    {
        return { a.x / b.x, a.y / b.y, a.z / b.z };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T min(T a, T b)
    {
        return { ::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z) };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T max(T a, T b)
    {
        return { ::max(a.x, b.x), ::max(a.y, b.y), ::max(a.z, b.z) };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T clamp(T v, T lo, T hi)
    {
        return vec3::max(lo, vec3::min(v, hi));
    }

    // ===== float3 specific =====

    __host__ __device__ __forceinline__ float3 mult(float3 v, float s)
    {
        return { v.x * s, v.y * s, v.z * s };
    }

    __host__ __device__ __forceinline__ float dot(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ __forceinline__ float3 cross(float3 a, float3 b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }

    __host__ __device__ __forceinline__ float3 norm(float3 v)
    {
        float lenSq = dot(v, v);
        if (lenSq <= 1e-6f) return { 0, 0, 0 };
        return mult(v, rsqrtf(lenSq));
    }

    // ===== int3 specific =====

    __host__ __device__ __forceinline__ int3 mult(int3 v, int s)
    {
        return { v.x * s, v.y * s, v.z * s };
    }

    __host__ __device__ __forceinline__ int dot(int3 a, int3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ __forceinline__ int prod(int3 v)
    {
        return v.x * v.y * v.z;
    }

    // ===== float4 (xyz only, w=0) =====

    __host__ __device__ __forceinline__ float4 add(float4 a, float4 b)
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z, 0 };
    }

    __host__ __device__ __forceinline__ float4 sub(float4 a, float4 b)
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z, 0 };
    }

    __host__ __device__ __forceinline__ float4 mult(float4 v, float s)
    {
        return { v.x * s, v.y * s, v.z * s, 0 };
    }

    __host__ __device__ __forceinline__ float4 div(float4 a, float4 b)
    {
        return { a.x / b.x, a.y / b.y, a.z / b.z, 0 };
    }

    __host__ __device__ __forceinline__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ __forceinline__ float4 cross(float4 a, float4 b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
            0
        };
    }

    __host__ __device__ __forceinline__ float4 norm(float4 v)
    {
        float lenSq = dot(v, v);
        if (lenSq <= 1e-6f) return { 0, 0, 0, 0 };
        return mult(v, rsqrtf(lenSq));
    }

    __host__ __device__ __forceinline__ float4 min(float4 a, float4 b)
    {
        return { fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), 0 };
    }

    __host__ __device__ __forceinline__ float4 max(float4 a, float4 b)
    {
        return { fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), 0 };
    }

    __host__ __device__ __forceinline__ bool gt(float4 a, float4 b)
    {
        return a.x > b.x && a.y > b.y && a.z > b.z;
    }

    __host__ __device__ __forceinline__ bool gte(float4 a, float4 b)
    {
        return a.x >= b.x && a.y >= b.y && a.z >= b.z;
    }

    __host__ __device__ __forceinline__ bool lt(float4 a, float4 b)
    {
        return a.x < b.x && a.y < b.y && a.z < b.z;
    }

    __host__ __device__ __forceinline__ bool lte(float4 a, float4 b)
    {
        return a.x <= b.x && a.y <= b.y && a.z <= b.z;
    }
}

namespace vec4
{
    // ===== Templated functions (float4, int4) =====

    template<typename T>
    __host__ __device__ __forceinline__ T add(T a, T b)
    {
        return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T sub(T a, T b)
    {
        return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T div(T a, T b)
    {
        return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T min(T a, T b)
    {
        return { ::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z), ::min(a.w, b.w) };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T max(T a, T b)
    {
        return { ::max(a.x, b.x), ::max(a.y, b.y), ::max(a.z, b.z), ::max(a.w, b.w) };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T clamp(T v, T lo, T hi)
    {
        return vec4::max(lo, vec4::min(v, hi));
    }

    // ===== float4 specific =====

    __host__ __device__ __forceinline__ float4 mult(float4 v, float s)
    {
        return { v.x * s, v.y * s, v.z * s, v.w * s };
    }

    __host__ __device__ __forceinline__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // ===== int4 specific =====

    __host__ __device__ __forceinline__ int4 mult(int4 v, int s)
    {
        return { v.x * s, v.y * s, v.z * s, v.w * s };
    }

    __host__ __device__ __forceinline__ int dot(int4 a, int4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
}

namespace quat
{
    // Normalize to unit quaternion
    __host__ __device__ __forceinline__ float4 norm(float4 q)
    {
        float d = vec4::dot(q, q);
        if (d <= 0.0f) return { 0, 0, 0, 1 };
        return vec4::mult(q, 1.0f / sqrtf(d));
    }

    // Get conjugate of given quaternion
    __host__ __device__ __forceinline__ float4 conj(float4 q)
    {
        return { -q.x, -q.y, -q.z, q.w };
    }

    // Compose two quaternions
    __host__ __device__ __forceinline__ float4 comp(float4 a, float4 b)
    {
        return {
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
            a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
        };
    }

    // Rotate vector from local to world space
    __host__ __device__ __forceinline__ float4 toWorld(float4 v, float4 q)
    {
        // t = 2 * cross(q.xyz, v)
        float4 t = vec3::mult(vec3::cross(q, v), 2.0f);

        // v' = v + q.w * t + cross(q.xyz, t)
        float4 u = vec3::cross(q, t);

        return {
            v.x + q.w * t.x + u.x,
            v.y + q.w * t.y + u.y,
            v.z + q.w * t.z + u.z,
            v.w
        };
    }

    // Rotate vector from world to local space
    __host__ __device__ __forceinline__ float4 toLocal(float4 v, float4 q)
    {
        return toWorld(v, conj(q));
    }
}
