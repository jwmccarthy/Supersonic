#include "CudaMath.h"

CUDA_BOTH float CudaMath::WrapNormalizeFloat(const float val, const float minmax) {
    float range = 2.0f * Fabsf(minmax);
    float result = Fmodf(val, range);
    
    if (result > minmax) {
        result -= range;
    } else if (result < -minmax) {
        result += range;
    }
    
    return result;
}