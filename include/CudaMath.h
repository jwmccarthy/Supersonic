#pragma once

#include "CudaCommon.h"

namespace CudaMath {
    inline constexpr float PI = 3.1415926535897932384626433832795029;
    inline constexpr float TWO_PI = 2 * PI;
    inline constexpr float HALF_PI = PI * 0.5f;

    CUDA_BOTH inline float Clamp(const float val, const float min, const float max) {
        return (val < min) ? min : (val > max) ? max : val;
    }

    CUDA_BOTH inline float Fabsf(const float x) {
        return fabsf(x);
    }

    CUDA_BOTH inline float Fmodf(const float x, const float y) {
        return fmodf(x, y);
    }

    CUDA_BOTH inline float Sinf(const float x) {
        return sinf(x);
    }

    CUDA_BOTH inline float Cosf(const float x) {
        return cosf(x);
    }

    CUDA_BOTH inline float Tanf(const float x) {
        return tanf(x);
    }

    CUDA_BOTH inline float Acosf(const float x) {
        return acosf(Clamp(x, -1.0f, 1.0f));
    }
    
    CUDA_BOTH inline float Asinf(const float x) {
        return asinf(Clamp(x, -1.0f, 1.0f));
    }

    CUDA_BOTH inline float Atan2f(const float x, const float y) {
        return atan2f(x, y);
    }

    CUDA_BOTH float WrapNormalizeFloat(const float val, const float minmax);
}