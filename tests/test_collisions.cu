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

// ============================================================================
// Helper functions to run GPU tests
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
