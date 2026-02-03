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

    template<typename T>
    __host__ __device__ __forceinline__ bool eq(T a, T b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }

    template<typename T>
    __host__ __device__ __forceinline__ bool gt(T a, T b)
    {
        return a.x > b.x && a.y > b.y && a.z > b.z;
    }

    template<typename T>
    __host__ __device__ __forceinline__ bool gte(T a, T b)
    {
        return a.x >= b.x && a.y >= b.y && a.z >= b.z;
    }

    template<typename T>
    __host__ __device__ __forceinline__ bool lt(T a, T b)
    {
        return a.x < b.x && a.y < b.y && a.z < b.z;
    }

    template<typename T>
    __host__ __device__ __forceinline__ bool lte(T a, T b)
    {
        return a.x <= b.x && a.y <= b.y && a.z <= b.z;
    }

    template<typename T, typename S>
    __host__ __device__ __forceinline__ T mult(T v, S s)
    {
        return { v.x * s, v.y * s, v.z * s };
    }

    template<typename T>
    __host__ __device__ __forceinline__ auto dot(T a, T b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    template<typename T>
    __host__ __device__ __forceinline__ T cross(T a, T b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }

    template<typename T>
    __host__ __device__ __forceinline__ T norm(T v)
    {
        auto lenSq = dot(v, v);
        if (lenSq <= 1e-6f) return {};
        return mult(v, rsqrtf(lenSq));
    }

    __host__ __device__ __forceinline__ int prod(int3 v)
    {
        return v.x * v.y * v.z;
    }
}

namespace vec4
{
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

    template<typename T, typename S>
    __host__ __device__ __forceinline__ T mult(T v, S s)
    {
        return { v.x * s, v.y * s, v.z * s, v.w * s };
    }

    template<typename T>
    __host__ __device__ __forceinline__ auto dot(T a, T b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
}

namespace quat
{
    __host__ __device__ __forceinline__ float4 norm(float4 q)
    {
        float d = vec4::dot(q, q);
        if (d <= 0.0f) return { 0, 0, 0, 1 };
        return vec4::mult(q, 1.0f / sqrtf(d));
    }

    __host__ __device__ __forceinline__ float4 conj(float4 q)
    {
        return { -q.x, -q.y, -q.z, q.w };
    }

    __host__ __device__ __forceinline__ float4 comp(float4 a, float4 b)
    {
        return {
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
            a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
        };
    }

    __host__ __device__ __forceinline__ float4 toWorld(float4 v, float4 q)
    {
        float4 t = vec3::mult(vec3::cross(q, v), 2.0f);
        float4 u = vec3::cross(q, t);
        return {
            v.x + q.w * t.x + u.x,
            v.y + q.w * t.y + u.y,
            v.z + q.w * t.z + u.z,
            v.w
        };
    }

    __host__ __device__ __forceinline__ float4 toLocal(float4 v, float4 q)
    {
        return toWorld(v, conj(q));
    }
}
