#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "CudaMath.hpp"

// ============================================================================
// Test result structure for GPU -> CPU transfer
// ============================================================================

struct MathTestResult
{
    float4 vec;
    float  scalar;
};

// ============================================================================
// CUDA test kernels
// ============================================================================

__global__ void runQuatRotationTest(float4 v, float4 q, MathTestResult* result)
{
    result->vec = quat::mult(v, q);
}

__global__ void runQuatComposeTest(float4 a, float4 b, MathTestResult* result)
{
    result->vec = quat::comp(a, b);
}

__global__ void runQuatConjTest(float4 q, MathTestResult* result)
{
    result->vec = quat::conj(q);
}

__global__ void runQuatNormTest(float4 q, MathTestResult* result)
{
    result->vec = quat::norm(q);
    result->scalar = sqrtf(vec4::dot(result->vec, result->vec));
}

__global__ void runCrossProductTest(float4 a, float4 b, MathTestResult* result)
{
    result->vec = vec3::cross(a, b);
}

__global__ void runDotProductTest(float4 a, float4 b, MathTestResult* result)
{
    result->scalar = vec3::dot(a, b);
}

__global__ void runVec4DotProductTest(float4 a, float4 b, MathTestResult* result)
{
    result->scalar = vec4::dot(a, b);
}

__global__ void runNormalizationTest(float4 v, MathTestResult* result)
{
    result->vec = vec3::norm(v);
    result->scalar = sqrtf(vec3::dot(result->vec, result->vec));
}

__global__ void runVec3AddTest(float4 a, float4 b, MathTestResult* result)
{
    result->vec = vec3::add(a, b);
}

__global__ void runVec3SubTest(float4 a, float4 b, MathTestResult* result)
{
    result->vec = vec3::sub(a, b);
}

__global__ void runVec3MultTest(float4 v, float s, MathTestResult* result)
{
    result->vec = vec3::mult(v, s);
}

// ============================================================================
// Helper functions to run GPU tests
// ============================================================================

MathTestResult runQuatRotation(float4 v, float4 q)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runQuatRotationTest<<<1, 1>>>(v, q, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runQuatCompose(float4 a, float4 b)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runQuatComposeTest<<<1, 1>>>(a, b, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runQuatConj(float4 q)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runQuatConjTest<<<1, 1>>>(q, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runQuatNorm(float4 q)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runQuatNormTest<<<1, 1>>>(q, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runCrossProduct(float4 a, float4 b)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runCrossProductTest<<<1, 1>>>(a, b, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runDotProduct(float4 a, float4 b)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runDotProductTest<<<1, 1>>>(a, b, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runVec4DotProduct(float4 a, float4 b)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runVec4DotProductTest<<<1, 1>>>(a, b, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runNormalization(float4 v)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runNormalizationTest<<<1, 1>>>(v, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runVec3Add(float4 a, float4 b)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runVec3AddTest<<<1, 1>>>(a, b, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runVec3Sub(float4 a, float4 b)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runVec3SubTest<<<1, 1>>>(a, b, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

MathTestResult runVec3Mult(float4 v, float s)
{
    MathTestResult* d_result;
    MathTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(MathTestResult));
    runVec3MultTest<<<1, 1>>>(v, s, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(MathTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

// ============================================================================
// Vec3 Tests
// ============================================================================

class Vec3Test : public ::testing::Test {};

TEST_F(Vec3Test, Add)
{
    float4 a = make_float4(1, 2, 3, 0);
    float4 b = make_float4(4, 5, 6, 0);
    
    auto result = runVec3Add(a, b);
    
    EXPECT_NEAR(result.vec.x, 5.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 7.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 9.0f, 0.001f);
}

TEST_F(Vec3Test, Sub)
{
    float4 a = make_float4(5, 7, 9, 0);
    float4 b = make_float4(1, 2, 3, 0);
    
    auto result = runVec3Sub(a, b);
    
    EXPECT_NEAR(result.vec.x, 4.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 5.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 6.0f, 0.001f);
}

TEST_F(Vec3Test, ScalarMult)
{
    float4 v = make_float4(1, 2, 3, 0);
    
    auto result = runVec3Mult(v, 2.0f);
    
    EXPECT_NEAR(result.vec.x, 2.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 4.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 6.0f, 0.001f);
}

TEST_F(Vec3Test, DotProduct)
{
    float4 a = make_float4(1, 2, 3, 0);
    float4 b = make_float4(4, 5, 6, 0);
    
    auto result = runDotProduct(a, b);
    
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_NEAR(result.scalar, 32.0f, 0.001f);
}

TEST_F(Vec3Test, DotProductOrthogonal)
{
    float4 x = make_float4(1, 0, 0, 0);
    float4 y = make_float4(0, 1, 0, 0);
    
    auto result = runDotProduct(x, y);
    
    EXPECT_NEAR(result.scalar, 0.0f, 0.001f);
}

TEST_F(Vec3Test, CrossProductXY)
{
    float4 x = make_float4(1, 0, 0, 0);
    float4 y = make_float4(0, 1, 0, 0);
    
    auto result = runCrossProduct(x, y);
    
    // X cross Y = Z
    EXPECT_NEAR(result.vec.x, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 1.0f, 0.001f);
}

TEST_F(Vec3Test, CrossProductYZ)
{
    float4 y = make_float4(0, 1, 0, 0);
    float4 z = make_float4(0, 0, 1, 0);
    
    auto result = runCrossProduct(y, z);
    
    // Y cross Z = X
    EXPECT_NEAR(result.vec.x, 1.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.001f);
}

TEST_F(Vec3Test, CrossProductZX)
{
    float4 z = make_float4(0, 0, 1, 0);
    float4 x = make_float4(1, 0, 0, 0);
    
    auto result = runCrossProduct(z, x);
    
    // Z cross X = Y
    EXPECT_NEAR(result.vec.x, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 1.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.001f);
}

TEST_F(Vec3Test, CrossProductAnticommutative)
{
    float4 a = make_float4(1, 2, 3, 0);
    float4 b = make_float4(4, 5, 6, 0);
    
    auto ab = runCrossProduct(a, b);
    auto ba = runCrossProduct(b, a);
    
    // a x b = -(b x a)
    EXPECT_NEAR(ab.vec.x, -ba.vec.x, 0.001f);
    EXPECT_NEAR(ab.vec.y, -ba.vec.y, 0.001f);
    EXPECT_NEAR(ab.vec.z, -ba.vec.z, 0.001f);
}

TEST_F(Vec3Test, Normalization)
{
    float4 v = make_float4(3, 4, 0, 0);
    
    auto result = runNormalization(v);
    
    EXPECT_NEAR(result.scalar, 1.0f, 0.001f) << "Normalized vector should have length 1";
    EXPECT_NEAR(result.vec.x, 0.6f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.8f, 0.001f);
}

TEST_F(Vec3Test, NormalizationZeroVector)
{
    float4 v = make_float4(0, 0, 0, 0);
    
    auto result = runNormalization(v);
    
    // Zero vector should return zero
    EXPECT_NEAR(result.vec.x, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.001f);
}

TEST_F(Vec3Test, NormalizationUnitVector)
{
    float4 v = make_float4(1, 0, 0, 0);
    
    auto result = runNormalization(v);
    
    EXPECT_NEAR(result.scalar, 1.0f, 0.001f);
    EXPECT_NEAR(result.vec.x, 1.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.001f);
}

// ============================================================================
// Vec4 Tests
// ============================================================================

class Vec4Test : public ::testing::Test {};

TEST_F(Vec4Test, DotProduct)
{
    float4 a = make_float4(1, 2, 3, 4);
    float4 b = make_float4(5, 6, 7, 8);
    
    auto result = runVec4DotProduct(a, b);
    
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_NEAR(result.scalar, 70.0f, 0.001f);
}

// ============================================================================
// Quaternion Tests
// ============================================================================

class QuaternionTest : public ::testing::Test {};

TEST_F(QuaternionTest, IdentityRotation)
{
    float4 v = make_float4(1, 0, 0, 0);
    float4 q = make_float4(0, 0, 0, 1);  // Identity quaternion
    
    auto result = runQuatRotation(v, q);
    
    EXPECT_NEAR(result.vec.x, 1.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.001f);
}

TEST_F(QuaternionTest, Rotation90DegreeZ)
{
    float4 v = make_float4(1, 0, 0, 0);
    float s = sinf(3.14159f / 4.0f);  // 45 deg for quat = 90 deg rotation
    float c = cosf(3.14159f / 4.0f);
    float4 q = make_float4(0, 0, s, c);
    
    auto result = runQuatRotation(v, q);
    
    // (1,0,0) rotated 90 degrees around Z should be (0,1,0)
    EXPECT_NEAR(result.vec.x, 0.0f, 0.01f);
    EXPECT_NEAR(result.vec.y, 1.0f, 0.01f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.01f);
}

TEST_F(QuaternionTest, Rotation90DegreeX)
{
    float4 v = make_float4(0, 1, 0, 0);
    float s = sinf(3.14159f / 4.0f);
    float c = cosf(3.14159f / 4.0f);
    float4 q = make_float4(s, 0, 0, c);  // 90 deg around X
    
    auto result = runQuatRotation(v, q);
    
    // (0,1,0) rotated 90 degrees around X should be (0,0,1)
    EXPECT_NEAR(result.vec.x, 0.0f, 0.01f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.01f);
    EXPECT_NEAR(result.vec.z, 1.0f, 0.01f);
}

TEST_F(QuaternionTest, Rotation180Degree)
{
    float4 v = make_float4(1, 0, 0, 0);
    // 180 deg around Z: sin(90) = 1, cos(90) = 0
    float4 q = make_float4(0, 0, 1, 0);
    
    auto result = runQuatRotation(v, q);
    
    // (1,0,0) rotated 180 degrees around Z should be (-1,0,0)
    EXPECT_NEAR(result.vec.x, -1.0f, 0.01f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.01f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.01f);
}

TEST_F(QuaternionTest, Conjugate)
{
    float4 q = make_float4(1, 2, 3, 4);
    
    auto result = runQuatConj(q);
    
    EXPECT_NEAR(result.vec.x, -1.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, -2.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, -3.0f, 0.001f);
    EXPECT_NEAR(result.vec.w, 4.0f, 0.001f);
}

TEST_F(QuaternionTest, Normalize)
{
    float4 q = make_float4(1, 2, 3, 4);
    
    auto result = runQuatNorm(q);
    
    EXPECT_NEAR(result.scalar, 1.0f, 0.001f) << "Normalized quaternion should have length 1";
}

TEST_F(QuaternionTest, ComposeWithIdentity)
{
    float4 q = make_float4(0.5f, 0.5f, 0.5f, 0.5f);
    float4 identity = make_float4(0, 0, 0, 1);
    
    auto result = runQuatCompose(q, identity);
    
    EXPECT_NEAR(result.vec.x, q.x, 0.001f);
    EXPECT_NEAR(result.vec.y, q.y, 0.001f);
    EXPECT_NEAR(result.vec.z, q.z, 0.001f);
    EXPECT_NEAR(result.vec.w, q.w, 0.001f);
}

TEST_F(QuaternionTest, ComposeIdentityLeft)
{
    float4 q = make_float4(0.5f, 0.5f, 0.5f, 0.5f);
    float4 identity = make_float4(0, 0, 0, 1);
    
    auto result = runQuatCompose(identity, q);
    
    EXPECT_NEAR(result.vec.x, q.x, 0.001f);
    EXPECT_NEAR(result.vec.y, q.y, 0.001f);
    EXPECT_NEAR(result.vec.z, q.z, 0.001f);
    EXPECT_NEAR(result.vec.w, q.w, 0.001f);
}

