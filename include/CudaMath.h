#pragma once

#include "CudaCommon.h"

namespace CudaMath {
    inline constexpr float PI = 3.1415926535897932384626433832795029;
    inline constexpr float TWO_PI = 2 * PI;
    inline constexpr float HALF_PI = PI * 0.5f;

    template<typename T>
    CUDA_BOTH T Clamp(const T& val, const T& min, const T& max) {
        return (val < min) ? min : (val > max) ? max : val;
    }

    CUDA_BOTH inline float Fabs(float x) {
        return fabsf(x);
    }

    CUDA_BOTH inline float Atan2f(float x, float y) {
        return atan2f(x, y);
    }

    CUDA_BOTH inline float Asinf(float x) {
        return asinf(Clamp(x, -1.0f, 1.0f));
    }
}