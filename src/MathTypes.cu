#include "MathTypes.h"
#include <assert.h>

// CUDA vector class implementation

// Vector operation macros
#define DEFINE_VEC_OP_VEC(op) \
    CUDA_BOTH CudaVec CudaVec::operator op(const CudaVec& other) const { \
        return CudaVec(x op other.x, y op other.y, z op other.z); \
    } \
    CUDA_BOTH CudaVec& CudaVec::operator op##=(const CudaVec& other) { \
        return *this = *this op other; \
    }

// Scalar operation macros
#define DEFINE_VEC_OP_FLT(op) \
    CUDA_BOTH CudaVec CudaVec::operator op(float scalar) const { \
        return CudaVec(x op scalar, y op scalar, z op scalar); \
    } \
    CUDA_BOTH CudaVec& CudaVec::operator op##=(float scalar) { \
        return *this = *this op scalar; \
    }

// Basic + In-place vector operations
DEFINE_VEC_OP_VEC(+)
DEFINE_VEC_OP_VEC(-)
DEFINE_VEC_OP_VEC(*)
DEFINE_VEC_OP_VEC(/)

// Basic + In-place float operations
DEFINE_VEC_OP_FLT(*)
DEFINE_VEC_OP_FLT(/)

// Comparison operations
CUDA_BOTH bool CudaVec::operator==(const CudaVec& other) const {
    return (x == other.x) && (y == other.y) && (z == other.z);
}

CUDA_BOTH bool CudaVec::operator!=(const CudaVec& other) const {
    return !(*this == other);
}

CUDA_BOTH bool CudaVec::operator<(const CudaVec& other) const {
    return (x < other.x) && (y < other.y) && (z < other.z);
}

CUDA_BOTH bool CudaVec::operator>(const CudaVec& other) const {
    return (x > other.x) && (y > other.y) && (z > other.z);
}

// Negation
CUDA_BOTH CudaVec CudaVec::operator-() const {
    return CudaVec(-x, -y, -z);
}

// Indexing
CUDA_BOTH float& CudaVec::operator[](int index) {
    assert(index >= 0 && index < 3);
    return ((float*)this)[index];
}

CUDA_BOTH float CudaVec::operator[](int index) const {
    assert(index >= 0 && index < 3);
    return ((float*)this)[index];
}

// Helper functions
CUDA_BOTH bool CudaVec::IsZero() const {
    return (x == 0.0f) && (y == 0.0f) && (z == 0.0f);
}

CUDA_BOTH CudaVec CudaVec::To2D() const {
    return CudaVec(x, y, 0.0f);
}

// Vector properties
CUDA_BOTH float CudaVec::LengthSq() const {
    return x * x + y * y + z * z;
}

CUDA_BOTH float CudaVec::Length() const {
    float lengthSq = LengthSq();
    if (lengthSq > 0) {
        return sqrtf(lengthSq);
    }
    return 0.0f;
}

CUDA_BOTH float CudaVec::LengthSq2D() const {
    return x * x + y * y;
}

CUDA_BOTH float CudaVec::Length2D() const {
    float lengthSq = LengthSq2D();
    if (lengthSq > 0) {
        return sqrtf(lengthSq);
    }
    return 0.0f;
}

CUDA_BOTH CudaVec CudaVec::Normalized() const {
    float length = Length();
    if (length > __FLT_EPSILON__ * __FLT_EPSILON__) {
        return *this / length;
    }
    return CudaVec();
}

// Vector functions
CUDA_BOTH float CudaVec::Dot(const CudaVec& other) const {
    return x * other.x + y * other.y + z * other.z;
}

CUDA_BOTH float CudaVec::DistSq(const CudaVec& other) const {
    return (*this - other).LengthSq();
}

CUDA_BOTH float CudaVec::Dist(const CudaVec& other) const {
    return sqrtf(DistSq(other));
}

CUDA_BOTH float CudaVec::DistSq2D(const CudaVec& other) const {
    float dx = x - other.x;
    float dy = y - other.y;
    return dx * dx + dy * dy;
}

CUDA_BOTH float CudaVec::Dist2D(const CudaVec& other) const {
    return sqrtf(DistSq2D(other));
}

CUDA_BOTH CudaVec CudaVec::Cross(const CudaVec& other) const {
    return CudaVec(
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x
    );
}

// CUDA rotation matrix class implementation

CUDA_BOTH CudaRotMat::CudaRotMat() {
    f, r, u = CudaVec();
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
        u = f.Cross(r).Normalized(),
        r = u.Cross(f).Normalized();

    return CudaRotMat(f, r, u);
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