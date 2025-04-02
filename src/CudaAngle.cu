#include <assert.h>
#include "MathTypes.h"

CUDA_BOTH CudaAngle CudaAngle::FromRotMat(CudaRotMat& mat) {
    float
        y = atan2f(mat[0][1], mat[0][0]),
        p = asinf(-mat[0][2]),
        r = atan2f(mat[1][2], mat[2][2]);

    return CudaAngle();
}