#pragma once

#include <cmath>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define rsqrtf(x) (1.0f / sqrtf(x))
#endif

__device__ __forceinline__ float sign(float x)
{
    return (x >= 0.0f) ? 1.0f : -1.0f;
}

namespace vec3
{
    __device__ __forceinline__ float4 add(float4 a, float4 b)
    {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0);
    }

    __device__ __forceinline__ float4 sub(float4 a, float4 b)
    {
        return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0);
    }

    __device__ __forceinline__ float4 mult(float4 v, float s)
    {
        return make_float4(v.x * s, v.y * s, v.z * s, 0);
    }

    __device__ __forceinline__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __device__ __forceinline__ float4 cross(float4 a, float4 b)
    {
        return make_float4(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x, 0
        );
    }

    // Normalize xyz components
    __device__ __forceinline__ float4 norm(float4 v)
    {
        float lenSq = vec3::dot(v, v);
        if (lenSq <= 1e-6f) return make_float4(0, 0, 0, 0);
        return vec3::mult(v, rsqrtf(lenSq));
    }

    // Normalize with length² output
    __device__ __forceinline__ float4 norm(float4 v, float& lenSq)
    {
        lenSq = vec3::dot(v, v);
        if (lenSq <= 1e-6f) return make_float4(0, 0, 0, 0);
        return vec3::mult(v, rsqrtf(lenSq));
    }
}

namespace vec4
{
    __device__ __forceinline__ float4 add(float4 a, float4 b)
    {
        return 
        make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    __device__ __forceinline__ float4 sub(float4 a, float4 b)
    {
        return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    }

    __device__ __forceinline__ float4 mult(float4 v, float s)
    {
        return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
    }

    __device__ __forceinline__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
}

namespace quat
{
    // Normalize to unit quaternion
    __device__ __forceinline__ float4 norm(float4 q)
    {
        float d = vec4::dot(q, q);
        if (d <= 0.0f) return make_float4(0, 0, 0, 1);
        return vec4::mult(q, 1.0f / sqrtf(d));
    }

    // Get conjugate of given quaternion
    __device__ __forceinline__ float4 conj(float4 q)
    {
        return make_float4(-q.x, -q.y, -q.z, q.w);
    }

    // Compose two quaternions
    __device__ __forceinline__ float4 comp(float4 a, float4 b)
    {
        return make_float4(
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
            a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
        );
    }

    // Transform a vector via a quaternion (inv=true: world -> local)
    __device__ __forceinline__ float4 mult(float4 v, float4 q, bool inv=false)
    {
        // If inverse, use conjugate
        if (inv) q = quat::conj(q);
        
        // Standard rotation formula: t = 2 * cross(q.xyz, v)
        float4 t = vec3::mult(vec3::cross(q, v), 2.0f);
        
        // v' = v + q.w * t + cross(q.xyz, t)
        float4 u = vec3::cross(q, t);
        
        return make_float4(
            v.x + q.w * t.x + u.x,
            v.y + q.w * t.y + u.y,
            v.z + q.w * t.z + u.z, v.w
        );
    }
}