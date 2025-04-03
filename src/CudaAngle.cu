#include <assert.h>
#include "MathTypes.h"
#include "CudaMath.h"

using namespace CudaMath;

CUDA_BOTH CudaAngle CudaAngle::FromRotMat(CudaRotMat& mat) {
    float
        y = Atan2f(mat[0].y, mat[0].x),
        p = Asinf(-mat[0].z),
        r = Atan2f(mat[1].z, mat[2].z);

    // prevent gimbal lock
    if (Fabs(p) == HALF_PI) {
        y += (y > 0) ? -PI : PI;
        r += (r > 0) ? -PI : PI;
    }

    return CudaAngle(y, -p, -r);
}

CUDA_BOTH CudaRotMat CudaAngle::ToRotMat() const {
    float
        cosRoll = Cosf(-roll), cosPitch = Cosf(-pitch), cosYaw = Cosf(yaw),
        sinRoll = Sinf(-roll), sinPitch = Sinf(-pitch), sinYaw = Sinf(yaw);
    
    float
        cosRollCosYaw = cosRoll * cosYaw,
        cosRollSinYaw = cosRoll * sinYaw,
        sinRollCosYaw = sinRoll * cosYaw,
        sinRollSinYaw = sinRoll * sinYaw;
    
    return CudaRotMat(
        // Forward vector
        CudaVec(cosPitch * cosYaw, cosPitch * sinYaw, -sinPitch),

        // Right vector
        CudaVec(sinPitch * sinRollCosYaw - cosRollSinYaw,
                sinPitch * sinRollSinYaw + cosRollCosYaw,
                cosPitch * sinRoll),

        // Up Vector
        CudaVec(sinPitch * cosRollCosYaw + sinRollSinYaw,
                sinPitch * cosRollSinYaw - sinRollCosYaw,
                cosPitch * cosRoll)
    );
}

CUDA_BOTH CudaAngle CudaAngle::FromVec(const CudaVec& f) {
    float yaw = 0.0f;
    float pitch = 0.0f;

    if (abs(f.y) > __FLT_EPSILON__ || abs(f.x) > __FLT_EPSILON__) {
        yaw = Atan2f(f.y, f.x);
        float dist2D = f.Length2D();
        pitch = -Atan2f(-f.z, dist2D);
    } else {
        // Vector has ~0 horizontal length
        if (f.z > __FLT_EPSILON__) {
            pitch = HALF_PI;
        } else if (f.z < -__FLT_EPSILON__) {
            pitch = -HALF_PI;
        }
    }

    return CudaAngle(yaw, pitch, 0);
}

CUDA_BOTH CudaVec CudaAngle::GetForwardVec() const {
    float
        cosPitch = Cosf(-pitch), cosYaw = Cosf(yaw),
        sinPitch = Sinf(-pitch), sinYaw = Sinf(yaw);

    return CudaVec(
        cosPitch * cosYaw, 
        sinPitch * sinYaw,
        -sinPitch
    );
}