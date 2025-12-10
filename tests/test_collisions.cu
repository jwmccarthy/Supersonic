#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsSAT.hpp"
#include "CollisionsEdge.hpp"

// ============================================================================
// Test result structures for GPU -> CPU transfer
// ============================================================================

struct SATTestResult
{
    float  maxSep;
    int    axisIdx;
    bool   overlap;
};

struct EdgeAxesResult
{
    int ai, a1, a2;
    int bi, b1, b2;
};

struct MathTestResult
{
    float4 vec;
    float  scalar;
};

// ============================================================================
// CUDA test kernels
// ============================================================================

__global__ void runSATTest(
    float4 posA, float4 rotA, 
    float4 posB, float4 rotB, 
    SATTestResult* result
)
{
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult res = carCarSATTest(ctx);
    
    result->maxSep = res.maxSep;
    result->axisIdx = res.axisIdx;
    result->overlap = res.overlap;
}

__global__ void runEdgeAxesTest(int axisIdx, EdgeAxesResult* result)
{
    EdgeAxes ax = getEdgeAxes(axisIdx);
    result->ai = ax.ai;
    result->a1 = ax.a1;
    result->a2 = ax.a2;
    result->bi = ax.bi;
    result->b1 = ax.b1;
    result->b2 = ax.b2;
}

__global__ void runQuatRotationTest(float4 v, float4 q, MathTestResult* result)
{
    result->vec = quat::mult(v, q);
}

__global__ void runCrossProductTest(float4 a, float4 b, MathTestResult* result)
{
    result->vec = vec3::cross(a, b);
}

__global__ void runDotProductTest(float4 a, float4 b, MathTestResult* result)
{
    result->scalar = vec3::dot(a, b);
}

__global__ void runNormalizationTest(float4 v, MathTestResult* result)
{
    result->vec = vec3::norm(v);
    result->scalar = sqrtf(vec3::dot(result->vec, result->vec));
}

// ============================================================================
// Helper to run SAT test and get result
// ============================================================================

SATTestResult runSAT(float4 posA, float4 rotA, float4 posB, float4 rotB)
{
    SATTestResult* d_result;
    SATTestResult h_result;
    
    cudaMalloc(&d_result, sizeof(SATTestResult));
    runSATTest<<<1, 1>>>(posA, rotA, posB, rotB, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(SATTestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

EdgeAxesResult runEdgeAxes(int axisIdx)
{
    EdgeAxesResult* d_result;
    EdgeAxesResult h_result;
    
    cudaMalloc(&d_result, sizeof(EdgeAxesResult));
    runEdgeAxesTest<<<1, 1>>>(axisIdx, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(EdgeAxesResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

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

// ============================================================================
// SAT Collision Tests
// ============================================================================

class SATCollisionTest : public ::testing::Test
{
protected:
    float4 identityQuat = make_float4(0, 0, 0, 1);
    float4 origin = make_float4(0, 0, 0, 0);
};

TEST_F(SATCollisionTest, IdenticalPositionsOverlap)
{
    auto result = runSAT(origin, identityQuat, origin, identityQuat);
    
    EXPECT_TRUE(result.overlap) << "Identical positions should overlap";
    EXPECT_LT(result.maxSep, 0) << "Separation should be negative (penetration)";
}

TEST_F(SATCollisionTest, FarApartNoOverlap)
{
    float4 farPos = make_float4(500, 0, 0, 0);
    auto result = runSAT(origin, identityQuat, farPos, identityQuat);
    
    EXPECT_FALSE(result.overlap) << "Cars 500 units apart should not overlap";
    EXPECT_GT(result.maxSep, 0) << "Separation should be positive";
}

TEST_F(SATCollisionTest, BarelyTouchingOverlaps)
{
    // Car half-extent along X is 59, so 2*59=118 is the touching distance
    float4 touchingPos = make_float4(115, 0, 0, 0);
    auto result = runSAT(origin, identityQuat, touchingPos, identityQuat);
    
    EXPECT_TRUE(result.overlap) << "Cars at 115 units should overlap (threshold ~118)";
}

TEST_F(SATCollisionTest, BarelySeparatedNoOverlap)
{
    // Just past the touching threshold
    float4 separatedPos = make_float4(120, 0, 0, 0);
    auto result = runSAT(origin, identityQuat, separatedPos, identityQuat);
    
    EXPECT_FALSE(result.overlap) << "Cars at 120 units should not overlap";
}

TEST_F(SATCollisionTest, RotatedCarOverlap)
{
    float4 pos = make_float4(100, 0, 0, 0);
    // 90 degree rotation around Z
    float s = sinf(3.14159f / 4.0f);
    float c = cosf(3.14159f / 4.0f);
    float4 rotB = make_float4(0, 0, s, c);
    
    auto result = runSAT(origin, identityQuat, pos, rotB);
    
    EXPECT_TRUE(result.overlap) << "Rotated car at 100 units should overlap";
}

TEST_F(SATCollisionTest, YOffsetOverlap)
{
    // Car half-extent along Y is 42.1, so 2*42.1=84.2 is threshold
    float4 yOffset = make_float4(0, 80, 0, 0);
    auto result = runSAT(origin, identityQuat, yOffset, identityQuat);
    
    EXPECT_TRUE(result.overlap) << "Cars offset 80 units along Y should overlap";
}

TEST_F(SATCollisionTest, ZOffsetOverlap)
{
    // Car half-extent along Z is 18.1, so 2*18.1=36.2 is threshold
    float4 zOffset = make_float4(0, 0, 35, 0);
    auto result = runSAT(origin, identityQuat, zOffset, identityQuat);
    
    EXPECT_TRUE(result.overlap) << "Cars offset 35 units along Z should overlap";
}

TEST_F(SATCollisionTest, DiagonalSeparation)
{
    float4 diagonal = make_float4(200, 200, 0, 0);
    auto result = runSAT(origin, identityQuat, diagonal, identityQuat);
    
    EXPECT_FALSE(result.overlap) << "Diagonal separation should not overlap";
}

TEST_F(SATCollisionTest, EdgeEdgeScenarioValidAxis)
{
    float4 pos = make_float4(100, 100, 0, 0);
    // 45 degree rotation
    float s = sinf(3.14159f / 8.0f);
    float c = cosf(3.14159f / 8.0f);
    float4 rot = make_float4(0, 0, s, c);
    
    auto result = runSAT(origin, identityQuat, pos, rot);
    
    EXPECT_GE(result.axisIdx, 0) << "Axis index should be non-negative";
    EXPECT_LT(result.axisIdx, 15) << "Axis index should be < 15 (6 face + 9 edge)";
}

TEST_F(SATCollisionTest, FaceAlignedCollisionReturnsFaceAxis)
{
    // Overlapping distance along X
    float4 pos = make_float4(110, 0, 0, 0);
    auto result = runSAT(origin, identityQuat, pos, identityQuat);
    
    EXPECT_TRUE(result.overlap) << "Should overlap at 110 units";
    EXPECT_GE(result.axisIdx, 0) << "Axis should be >= 0";
    EXPECT_LT(result.axisIdx, 6) << "Face-aligned collision should use face axis (0-5)";
}

// ============================================================================
// Edge Axes Tests
// ============================================================================

class EdgeAxesTest : public ::testing::Test {};

TEST_F(EdgeAxesTest, FirstEdgeAxis)
{
    auto result = runEdgeAxes(6);  // First edge-edge axis
    
    EXPECT_EQ(result.ai, 0);
    EXPECT_EQ(result.bi, 0);
    EXPECT_EQ(result.a1, 1);
    EXPECT_EQ(result.a2, 2);
    EXPECT_EQ(result.b1, 1);
    EXPECT_EQ(result.b2, 2);
}

TEST_F(EdgeAxesTest, LastEdgeAxis)
{
    auto result = runEdgeAxes(14);  // Last edge-edge axis
    
    EXPECT_EQ(result.ai, 2);
    EXPECT_EQ(result.bi, 2);
}

TEST_F(EdgeAxesTest, MiddleEdgeAxis)
{
    auto result = runEdgeAxes(9);  // axis (1,0)
    
    EXPECT_EQ(result.ai, 1);
    EXPECT_EQ(result.bi, 0);
}

// ============================================================================
// Math Utility Tests
// ============================================================================

class MathUtilsTest : public ::testing::Test {};

TEST_F(MathUtilsTest, QuaternionIdentityRotation)
{
    float4 v = make_float4(1, 0, 0, 0);
    float4 q = make_float4(0, 0, 0, 1);  // Identity quaternion
    
    auto result = runQuatRotation(v, q);
    
    EXPECT_NEAR(result.vec.x, 1.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.001f);
}

TEST_F(MathUtilsTest, Quaternion90DegreeZRotation)
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

TEST_F(MathUtilsTest, CrossProductXY)
{
    float4 x = make_float4(1, 0, 0, 0);
    float4 y = make_float4(0, 1, 0, 0);
    
    auto result = runCrossProduct(x, y);
    
    // X cross Y = Z
    EXPECT_NEAR(result.vec.x, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 1.0f, 0.001f);
}

TEST_F(MathUtilsTest, CrossProductYZ)
{
    float4 y = make_float4(0, 1, 0, 0);
    float4 z = make_float4(0, 0, 1, 0);
    
    auto result = runCrossProduct(y, z);
    
    // Y cross Z = X
    EXPECT_NEAR(result.vec.x, 1.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.001f);
}

TEST_F(MathUtilsTest, DotProduct)
{
    float4 a = make_float4(1, 2, 3, 0);
    float4 b = make_float4(4, 5, 6, 0);
    
    auto result = runDotProduct(a, b);
    
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_NEAR(result.scalar, 32.0f, 0.001f);
}

TEST_F(MathUtilsTest, Normalization)
{
    float4 v = make_float4(3, 4, 0, 0);
    
    auto result = runNormalization(v);
    
    EXPECT_NEAR(result.scalar, 1.0f, 0.001f) << "Normalized vector should have length 1";
    EXPECT_NEAR(result.vec.x, 0.6f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.8f, 0.001f);
}

TEST_F(MathUtilsTest, NormalizationZeroVector)
{
    float4 v = make_float4(0, 0, 0, 0);
    
    auto result = runNormalization(v);
    
    // Zero vector should return zero
    EXPECT_NEAR(result.vec.x, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.y, 0.0f, 0.001f);
    EXPECT_NEAR(result.vec.z, 0.0f, 0.001f);
}

