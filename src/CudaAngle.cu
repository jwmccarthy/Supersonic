#include <assert.h>
#include "MathTypes.h"
#include "CudaMath.h"

using namespace CudaMath;

CUDA_BOTH CudaAngle CudaAngle::FromRotMat(CudaRotMat& mat) {
    float
        y = Atan2f(mat[0][1], mat[0][0]),
        p = Asinf(-mat[0][2]),
        r = Atan2f(mat[1][2], mat[2][2]);

    // prevent gimbal lock
    if (Fabs(p) == HALF_PI) {
        y += (y > 0) ? -PI : PI;
        r += (r > 0) ? -PI : PI;
    }

    return CudaAngle(y, -p, -r);
}

CUDA_BOTH CudaRotMat CudaAngle::ToRotMat() const {
    return CudaRotMat();
}