#include "MathTypes.h"

CUDA_BOTH CudaTransform CudaTransform::GetIdentity() {
    return CudaTransform(CudaRotMat::GetIdentity(), CudaVec());
}

// Transform multiplication/composition
CUDA_BOTH CudaTransform CudaTransform::operator*(const CudaTransform& other) const {
    return CudaTransform(
        rotation.Dot(other.rotation),
        rotation.Dot(other.translation) + translation
    );
}

CUDA_BOTH CudaTransform& CudaTransform::operator*=(const CudaTransform& other) {
    *this = *this * other;
}

// Comparison operators
CUDA_BOTH bool CudaTransform::operator==(const CudaTransform& other) const {
    return (rotation == other.rotation) && (translation == other.translation);
}

CUDA_BOTH bool CudaTransform::operator!=(const CudaTransform& other) const {
    return !(*this == other);
}

// Transforms
CUDA_BOTH CudaTransform CudaTransform::Inverse() const {
    CudaRotMat inv = rotation.Transpose();
    return CudaTransform(inv, inv.Dot(-translation));
}

CUDA_BOTH CudaVec CudaTransform::InverseTransform(const CudaVec& vec) const {
    CudaVec v = vec - translation;
    return rotation.Transpose().Dot(v);
}