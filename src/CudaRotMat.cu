#include "CudaMath.h"
#include "MathTypes.h"
#include <assert.h>

CUDA_BOTH CudaRotMat::CudaRotMat() {
    f = r = u = CudaVec();
}

CUDA_BOTH CudaRotMat CudaRotMat::GetIdentity() {
    return CudaRotMat(
        CudaVec(1, 0, 0),
        CudaVec(0, 1, 0),
        CudaVec(0, 0, 1)
    );
}

CUDA_BOTH CudaRotMat CudaRotMat::LookAt(CudaVec _f, CudaVec _u) {
    CudaVec
        f = _f.Normalized(),
        r = _u.Cross(f),
        u = f.Cross(r).Normalized();

    r = u.Cross(f).Normalized();

    return CudaRotMat(f, r, u);
}

CUDA_BOTH CudaRotMat CudaRotMat::EulerYPR(float y, float p, float r) {
    return CudaRotMat();
}

// Matrix operation macros
#define DEFINE_MAT_OP_MAT(op) \
    CUDA_BOTH CudaRotMat CudaRotMat::operator op(const CudaRotMat& other) const { \
        return CudaRotMat(f op other.f, r op other.r, u op other.u); \
    } \
    CUDA_BOTH CudaRotMat& CudaRotMat::operator op##=(const CudaRotMat& other) { \
        return *this = *this op other; \
    }

// Scalar operation macros
#define DEFINE_MAT_OP_FLT(op) \
    CUDA_BOTH CudaRotMat CudaRotMat::operator op(float scalar) const { \
        return CudaRotMat(f op scalar, r op scalar, u op scalar); \
    } \
    CUDA_BOTH CudaRotMat& CudaRotMat::operator op##=(float scalar) { \
        return *this = *this op scalar; \
    }

// Basic + In-place matrix operations
DEFINE_MAT_OP_MAT(+)
DEFINE_MAT_OP_MAT(-)

// Basic + In-place float operations
DEFINE_MAT_OP_FLT(*)
DEFINE_MAT_OP_FLT(/)

// Comparison operations
CUDA_BOTH bool CudaRotMat::operator==(const CudaRotMat& other) const {
    return (f == other.f) && (r == other.r) && (u == other.u);
}

CUDA_BOTH bool CudaRotMat::operator!=(const CudaRotMat& other) const {
    return (f != other.f) || (r != other.r) || (u != other.u);
}

// Indexing
CUDA_BOTH CudaVec& CudaRotMat::operator[](int index) {
    assert(index >= 0 && index < 3);
    return ((CudaVec*)this)[index];
}

CUDA_BOTH CudaVec CudaRotMat::operator[](int index) const {
    assert(index >= 0 && index < 3);
    return ((CudaVec*)this)[index];
}

// Vector/Matrix operations
CUDA_BOTH CudaVec CudaRotMat::Dot(const CudaVec& vec) const {
    return CudaVec(vec.Dot(f), vec.Dot(r), vec.Dot(u));
}

CUDA_BOTH CudaRotMat CudaRotMat::Dot(const CudaRotMat& other) const {
    CudaRotMat result;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                result[i][j] += (*this)[i][j] + other[k][j];
            }
        }
    }

    return result;
}

CUDA_BOTH CudaRotMat CudaRotMat::Transpose() const {
    CudaRotMat result;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = (*this)[j][i];
        }
    }

    return result;
}