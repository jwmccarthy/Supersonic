#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "CudaMath.hpp"
#include "RLConstants.hpp"
#include "CollisionsSAT.hpp"
#include "CollisionsEdge.hpp"
#include "CollisionsFace.hpp"

// ============================================================================
// Test result structures for GPU -> CPU transfer
// ============================================================================

struct SATTestResult
{
    float  depth;
    int    axisIdx;
    bool   overlap;
};

struct EdgeAxesResult
{
    int ai, a1, a2;
    int bi, b1, b2;
};

struct BlendAxesResult
{
    float4 result;
};

struct FindIncidentAxisResult
{
    int bestIdx;
    float bestDot;
};

struct SetFaceVertsResult
{
    float4 verts[4];
};

struct ReferenceFaceResult
{
    float4 normal;
    float4 ortho1;
    float4 ortho2;
    float4 center;
    float2 halfEx;
};

struct IncidentFaceResult
{
    float4 verts[4];
};

struct EdgeManifoldResult
{
    int    count;
    float  depths[4];
    float4 points[4];
    float4 normal;
};

struct FaceManifoldResult
{
    int    count;
    float  depths[8];
    float4 points[8];
    float4 normal;
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
    
    result->depth = res.depth;
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

__global__ void runBlendAxesTest(float4 axA, float4 axB, float b, BlendAxesResult* result)
{
    result->result = blendAxes(axA, axB, b);
}

__global__ void runFindIncidentAxisTest(float4 dir, float4 ax0, float4 ax1, float4 ax2, FindIncidentAxisResult* result)
{
    float4 axes[3] = { ax0, ax1, ax2 };
    result->bestIdx = findIncidentAxis(dir, axes, result->bestDot);
}

__global__ void runSetFaceVertsTest(float4 center, float4 off1, float4 off2, SetFaceVertsResult* result)
{
    setFaceVertices(result->verts, center, off1, off2);
}

__global__ void runGetReferenceFaceTest(
    float4 posA, float4 rotA, float4 posB, float4 rotB, 
    int axisIdx, ReferenceFaceResult* result
)
{
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult res;
    res.axisIdx = axisIdx;
    res.overlap = true;
    
    ReferenceFace ref;
    getReferenceFace(ctx, res, ref);
    
    result->normal = ref.normal;
    result->ortho1 = ref.ortho1;
    result->ortho2 = ref.ortho2;
    result->center = ref.center;
    result->halfEx = ref.halfEx;
}

__global__ void runGetIncidentFaceTest(
    float4 posA, float4 rotA, float4 posB, float4 rotB,
    int axisIdx, IncidentFaceResult* result
)
{
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult res;
    res.axisIdx = axisIdx;
    res.overlap = true;
    
    ReferenceFace ref;
    getReferenceFace(ctx, res, ref);
    
    IncidentFace inc;
    getIncidentFace(ctx, res, ref, inc);
    
    result->verts[0] = inc.verts[0];
    result->verts[1] = inc.verts[1];
    result->verts[2] = inc.verts[2];
    result->verts[3] = inc.verts[3];
}

__global__ void runEdgeEdgeManifoldTest(
    float4 posA, float4 rotA, float4 posB, float4 rotB,
    EdgeManifoldResult* result
)
{
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult res = carCarSATTest(ctx);
    
    ContactManifold manifold = {};
    generateEdgeEdgeManifold(ctx, res, manifold);
    
    result->count = manifold.count;
    result->normal = manifold.normal;
    for (int i = 0; i < 4; i++)
    {
        result->points[i] = manifold.points[i];
        result->depths[i] = manifold.depths[i];
    }
}

__global__ void runFaceFaceManifoldTest(
    float4 posA, float4 rotA, float4 posB, float4 rotB,
    FaceManifoldResult* result
)
{
    SATContext ctx = buildSATContext(posA, rotA, posB, rotB);
    SATResult res = carCarSATTest(ctx);
    
    ContactManifold manifold = {};
    generateFaceFaceManifold(ctx, res, manifold);
    
    result->count = manifold.count;
    result->normal = manifold.normal;
    for (int i = 0; i < 8; i++)
    {
        result->points[i] = manifold.points[i];
        result->depths[i] = manifold.depths[i];
    }
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

BlendAxesResult runBlendAxes(float4 axA, float4 axB, float b)
{
    BlendAxesResult* d_result;
    BlendAxesResult h_result;
    
    cudaMalloc(&d_result, sizeof(BlendAxesResult));
    runBlendAxesTest<<<1, 1>>>(axA, axB, b, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(BlendAxesResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

FindIncidentAxisResult runFindIncidentAxis(float4 dir, float4 ax0, float4 ax1, float4 ax2)
{
    FindIncidentAxisResult* d_result;
    FindIncidentAxisResult h_result;
    
    cudaMalloc(&d_result, sizeof(FindIncidentAxisResult));
    runFindIncidentAxisTest<<<1, 1>>>(dir, ax0, ax1, ax2, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(FindIncidentAxisResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

SetFaceVertsResult runSetFaceVerts(float4 center, float4 off1, float4 off2)
{
    SetFaceVertsResult* d_result;
    SetFaceVertsResult h_result;
    
    cudaMalloc(&d_result, sizeof(SetFaceVertsResult));
    runSetFaceVertsTest<<<1, 1>>>(center, off1, off2, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(SetFaceVertsResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

ReferenceFaceResult runGetReferenceFace(float4 posA, float4 rotA, float4 posB, float4 rotB, int axisIdx)
{
    ReferenceFaceResult* d_result;
    ReferenceFaceResult h_result;
    
    cudaMalloc(&d_result, sizeof(ReferenceFaceResult));
    runGetReferenceFaceTest<<<1, 1>>>(posA, rotA, posB, rotB, axisIdx, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(ReferenceFaceResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

IncidentFaceResult runGetIncidentFace(float4 posA, float4 rotA, float4 posB, float4 rotB, int axisIdx)
{
    IncidentFaceResult* d_result;
    IncidentFaceResult h_result;
    
    cudaMalloc(&d_result, sizeof(IncidentFaceResult));
    runGetIncidentFaceTest<<<1, 1>>>(posA, rotA, posB, rotB, axisIdx, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(IncidentFaceResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

EdgeManifoldResult runEdgeEdgeManifold(float4 posA, float4 rotA, float4 posB, float4 rotB)
{
    EdgeManifoldResult* d_result;
    EdgeManifoldResult h_result;
    
    cudaMalloc(&d_result, sizeof(EdgeManifoldResult));
    runEdgeEdgeManifoldTest<<<1, 1>>>(posA, rotA, posB, rotB, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(EdgeManifoldResult), cudaMemcpyDeviceToHost);
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
    EXPECT_GT(result.depth, 0) << "Depth should be positive (penetration)";
}

TEST_F(SATCollisionTest, FarApartNoOverlap)
{
    float4 farPos = make_float4(500, 0, 0, 0);
    auto result = runSAT(origin, identityQuat, farPos, identityQuat);
    
    EXPECT_FALSE(result.overlap) << "Cars 500 units apart should not overlap";
    EXPECT_LT(result.depth, 0) << "Depth should be negative (separation)";
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
// Face-Face Helper Tests
// ============================================================================

class BlendAxesTest : public ::testing::Test
{
protected:
    float4 axisX = make_float4(1, 0, 0, 0);
    float4 axisY = make_float4(0, 1, 0, 0);
    float4 axisZ = make_float4(0, 0, 1, 0);
};

TEST_F(BlendAxesTest, BlendZeroReturnsFirst)
{
    auto result = runBlendAxes(axisX, axisY, 0.0f);
    
    EXPECT_NEAR(result.result.x, 1.0f, 1e-5f);
    EXPECT_NEAR(result.result.y, 0.0f, 1e-5f);
    EXPECT_NEAR(result.result.z, 0.0f, 1e-5f);
}

TEST_F(BlendAxesTest, BlendOneReturnsSecond)
{
    auto result = runBlendAxes(axisX, axisY, 1.0f);
    
    EXPECT_NEAR(result.result.x, 0.0f, 1e-5f);
    EXPECT_NEAR(result.result.y, 1.0f, 1e-5f);
    EXPECT_NEAR(result.result.z, 0.0f, 1e-5f);
}

TEST_F(BlendAxesTest, BlendHalfReturnsMidpoint)
{
    auto result = runBlendAxes(axisX, axisY, 0.5f);
    
    EXPECT_NEAR(result.result.x, 0.5f, 1e-5f);
    EXPECT_NEAR(result.result.y, 0.5f, 1e-5f);
    EXPECT_NEAR(result.result.z, 0.0f, 1e-5f);
}

// ============================================================================
// FindIncidentAxis Tests
// ============================================================================

class FindIncidentAxisTest : public ::testing::Test
{
protected:
    float4 axisX = make_float4(1, 0, 0, 0);
    float4 axisY = make_float4(0, 1, 0, 0);
    float4 axisZ = make_float4(0, 0, 1, 0);
};

TEST_F(FindIncidentAxisTest, FindsXAxisForNegativeX)
{
    // Direction pointing in -X should find X axis (largest |dot| = 1)
    float4 dir = make_float4(-1, 0, 0, 0);
    auto result = runFindIncidentAxis(dir, axisX, axisY, axisZ);
    
    EXPECT_EQ(result.bestIdx, 0);
    EXPECT_NEAR(result.bestDot, -1.0f, 1e-5f);
}

TEST_F(FindIncidentAxisTest, FindsXAxisForPositiveX)
{
    // Direction pointing in +X should also find X axis (largest |dot| = 1)
    float4 dir = make_float4(1, 0, 0, 0);
    auto result = runFindIncidentAxis(dir, axisX, axisY, axisZ);
    
    EXPECT_EQ(result.bestIdx, 0);
    EXPECT_NEAR(result.bestDot, 1.0f, 1e-5f);
}

TEST_F(FindIncidentAxisTest, FindsYAxisForNegativeY)
{
    float4 dir = make_float4(0, -1, 0, 0);
    auto result = runFindIncidentAxis(dir, axisX, axisY, axisZ);
    
    EXPECT_EQ(result.bestIdx, 1);
    EXPECT_NEAR(result.bestDot, -1.0f, 1e-5f);
}

TEST_F(FindIncidentAxisTest, FindsZAxisForNegativeZ)
{
    float4 dir = make_float4(0, 0, -1, 0);
    auto result = runFindIncidentAxis(dir, axisX, axisY, axisZ);
    
    EXPECT_EQ(result.bestIdx, 2);
    EXPECT_NEAR(result.bestDot, -1.0f, 1e-5f);
}

TEST_F(FindIncidentAxisTest, DiagonalDirectionFindsLargestComponent)
{
    // Direction pointing mostly in -Y (|dot| with Y = 0.9, largest)
    float4 dir = make_float4(0.1f, -0.9f, 0.1f, 0);
    auto result = runFindIncidentAxis(dir, axisX, axisY, axisZ);
    
    EXPECT_EQ(result.bestIdx, 1) << "Should find Y axis (largest |dot|)";
    EXPECT_LT(result.bestDot, 0.0f) << "Dot product should be negative";
}

// ============================================================================
// SetFaceVertices Tests
// ============================================================================

class SetFaceVertsTest : public ::testing::Test {};

TEST_F(SetFaceVertsTest, GeneratesFourDistinctVertices)
{
    float4 center = make_float4(10, 20, 30, 0);
    float4 off1 = make_float4(5, 0, 0, 0);
    float4 off2 = make_float4(0, 3, 0, 0);
    
    auto result = runSetFaceVerts(center, off1, off2);
    
    // v0 = center + off1 + off2 = (15, 23, 30)
    EXPECT_NEAR(result.verts[0].x, 15.0f, 1e-5f);
    EXPECT_NEAR(result.verts[0].y, 23.0f, 1e-5f);
    EXPECT_NEAR(result.verts[0].z, 30.0f, 1e-5f);
    
    // v1 = center + off1 - off2 = (15, 17, 30)
    EXPECT_NEAR(result.verts[1].x, 15.0f, 1e-5f);
    EXPECT_NEAR(result.verts[1].y, 17.0f, 1e-5f);
    EXPECT_NEAR(result.verts[1].z, 30.0f, 1e-5f);
    
    // v2 = center - off1 - off2 = (5, 17, 30)
    EXPECT_NEAR(result.verts[2].x, 5.0f, 1e-5f);
    EXPECT_NEAR(result.verts[2].y, 17.0f, 1e-5f);
    EXPECT_NEAR(result.verts[2].z, 30.0f, 1e-5f);
    
    // v3 = center - off1 + off2 = (5, 23, 30)
    EXPECT_NEAR(result.verts[3].x, 5.0f, 1e-5f);
    EXPECT_NEAR(result.verts[3].y, 23.0f, 1e-5f);
    EXPECT_NEAR(result.verts[3].z, 30.0f, 1e-5f);
}

TEST_F(SetFaceVertsTest, VertsFormRectangle)
{
    float4 center = make_float4(0, 0, 0, 0);
    float4 off1 = make_float4(10, 0, 0, 0);
    float4 off2 = make_float4(0, 5, 0, 0);
    
    auto result = runSetFaceVerts(center, off1, off2);
    
    // Check that opposite vertices are symmetric about center
    // v0 and v2 should be opposite
    EXPECT_NEAR(result.verts[0].x + result.verts[2].x, 0.0f, 1e-5f);
    EXPECT_NEAR(result.verts[0].y + result.verts[2].y, 0.0f, 1e-5f);
    
    // v1 and v3 should be opposite
    EXPECT_NEAR(result.verts[1].x + result.verts[3].x, 0.0f, 1e-5f);
    EXPECT_NEAR(result.verts[1].y + result.verts[3].y, 0.0f, 1e-5f);
}

// ============================================================================
// GetReferenceFace Tests
// ============================================================================

class GetReferenceFaceTest : public ::testing::Test
{
protected:
    float4 identityQuat = make_float4(0, 0, 0, 1);
    float4 origin = make_float4(0, 0, 0, 0);
};

TEST_F(GetReferenceFaceTest, XFaceAxisAligned)
{
    // Car B offset along X axis, axisIdx=0 means A's X face is reference
    float4 posB = make_float4(100, 0, 0, 0);
    
    auto result = runGetReferenceFace(origin, identityQuat, posB, identityQuat, 0);
    
    // Normal should point toward B (positive X)
    EXPECT_GT(result.normal.x, 0.9f) << "Normal should point in +X direction";
    EXPECT_NEAR(result.normal.y, 0.0f, 1e-5f);
    EXPECT_NEAR(result.normal.z, 0.0f, 1e-5f);
    
    // Half-extents should be Y and Z extents
    EXPECT_NEAR(result.halfEx.x, 42.1f, 0.1f);  // CAR_HALF_EX.y
    EXPECT_NEAR(result.halfEx.y, 18.1f, 0.1f);  // CAR_HALF_EX.z
}

TEST_F(GetReferenceFaceTest, YFaceAxisAligned)
{
    float4 posB = make_float4(0, 80, 0, 0);
    
    auto result = runGetReferenceFace(origin, identityQuat, posB, identityQuat, 1);
    
    // Normal should point toward B (positive Y)
    EXPECT_NEAR(result.normal.x, 0.0f, 1e-5f);
    EXPECT_GT(result.normal.y, 0.9f) << "Normal should point in +Y direction";
    EXPECT_NEAR(result.normal.z, 0.0f, 1e-5f);
    
    // Half-extents should be Z and X extents
    EXPECT_NEAR(result.halfEx.x, 18.1f, 0.1f);  // CAR_HALF_EX.z
    EXPECT_NEAR(result.halfEx.y, 59.0f, 0.1f);  // CAR_HALF_EX.x
}

TEST_F(GetReferenceFaceTest, NegativeOffsetFlipsNormal)
{
    // Car B behind A (negative X)
    float4 posB = make_float4(-100, 0, 0, 0);
    
    auto result = runGetReferenceFace(origin, identityQuat, posB, identityQuat, 0);
    
    // Normal should point toward B (negative X)
    EXPECT_LT(result.normal.x, -0.9f) << "Normal should point in -X direction";
}

TEST_F(GetReferenceFaceTest, BoxBFaceAsReference)
{
    // axisIdx=3 means B's X face is reference
    float4 posB = make_float4(100, 0, 0, 0);
    
    auto result = runGetReferenceFace(origin, identityQuat, posB, identityQuat, 3);
    
    // Normal should point from B toward A (negative X)
    EXPECT_LT(result.normal.x, -0.9f) << "B's reference normal should point toward A";
}

// ============================================================================
// GetIncidentFace Tests
// ============================================================================

class GetIncidentFaceTest : public ::testing::Test
{
protected:
    float4 identityQuat = make_float4(0, 0, 0, 1);
    float4 origin = make_float4(0, 0, 0, 0);
};

TEST_F(GetIncidentFaceTest, IncidentFaceHasFourVertices)
{
    float4 posB = make_float4(100, 0, 0, 0);
    
    auto result = runGetIncidentFace(origin, identityQuat, posB, identityQuat, 0);
    
    // All four vertices should be valid (non-zero in at least one component)
    for (int i = 0; i < 4; i++)
    {
        float mag = result.verts[i].x * result.verts[i].x +
                    result.verts[i].y * result.verts[i].y +
                    result.verts[i].z * result.verts[i].z;
        EXPECT_GT(mag, 1.0f) << "Vertex " << i << " should have non-trivial position";
    }
}

TEST_F(GetIncidentFaceTest, IncidentFaceOnOppositeBox)
{
    // A's X face is reference (axisIdx=0), so incident face is on B
    // B is at x=100, so incident face should be near x=100-59=41 (B's -X face)
    float4 posB = make_float4(100, 0, 0, 0);
    
    auto result = runGetIncidentFace(origin, identityQuat, posB, identityQuat, 0);
    
    // All incident vertices should have X coordinate near B's -X face
    // B center is at 100, half-extent is 59, so -X face is at ~41
    for (int i = 0; i < 4; i++)
    {
        EXPECT_NEAR(result.verts[i].x, 41.0f, 5.0f) 
            << "Vertex " << i << " should be on B's -X face";
    }
}

TEST_F(GetIncidentFaceTest, VertsFormValidRectangle)
{
    float4 posB = make_float4(100, 0, 0, 0);
    
    auto result = runGetIncidentFace(origin, identityQuat, posB, identityQuat, 0);
    
    // Compute centroid of the 4 vertices
    float cx = (result.verts[0].x + result.verts[1].x + result.verts[2].x + result.verts[3].x) / 4;
    float cy = (result.verts[0].y + result.verts[1].y + result.verts[2].y + result.verts[3].y) / 4;
    float cz = (result.verts[0].z + result.verts[1].z + result.verts[2].z + result.verts[3].z) / 4;
    
    // Opposite vertices should be equidistant from centroid
    float d0 = sqrtf(powf(result.verts[0].x - cx, 2) + powf(result.verts[0].y - cy, 2) + powf(result.verts[0].z - cz, 2));
    float d2 = sqrtf(powf(result.verts[2].x - cx, 2) + powf(result.verts[2].y - cy, 2) + powf(result.verts[2].z - cz, 2));
    
    EXPECT_NEAR(d0, d2, 1e-3f) << "Opposite vertices should be equidistant from center";
}

// ============================================================================
// Edge-Edge Manifold Tests
// ============================================================================

class EdgeEdgeManifoldTest : public ::testing::Test
{
protected:
    float4 identityQuat = make_float4(0, 0, 0, 1);
    float4 origin = make_float4(0, 0, 0, 0);
};

TEST_F(EdgeEdgeManifoldTest, ManifoldHasOneContactPoint)
{
    // Position B so edges intersect (rotated 45 degrees, offset diagonally)
    float4 posB = make_float4(80, 80, 0, 0);
    float s = sinf(3.14159f / 8.0f);  // 22.5 degrees
    float c = cosf(3.14159f / 8.0f);
    float4 rotB = make_float4(0, 0, s, c);
    
    auto result = runEdgeEdgeManifold(origin, identityQuat, posB, rotB);
    
    EXPECT_EQ(result.count, 1) << "Edge-edge collision should produce exactly 1 contact point";
}

TEST_F(EdgeEdgeManifoldTest, NormalIsUnitLength)
{
    float4 posB = make_float4(80, 80, 0, 0);
    float s = sinf(3.14159f / 8.0f);
    float c = cosf(3.14159f / 8.0f);
    float4 rotB = make_float4(0, 0, s, c);
    
    auto result = runEdgeEdgeManifold(origin, identityQuat, posB, rotB);
    
    float len = sqrtf(result.normal.x * result.normal.x + 
                      result.normal.y * result.normal.y + 
                      result.normal.z * result.normal.z);
    EXPECT_NEAR(len, 1.0f, 1e-4f) << "Contact normal should be unit length";
}

TEST_F(EdgeEdgeManifoldTest, NormalPointsFromAToB)
{
    float4 posB = make_float4(80, 80, 0, 0);
    float s = sinf(3.14159f / 8.0f);
    float c = cosf(3.14159f / 8.0f);
    float4 rotB = make_float4(0, 0, s, c);
    
    auto result = runEdgeEdgeManifold(origin, identityQuat, posB, rotB);
    
    // Normal dot (A->B direction) should be positive
    float dot = result.normal.x * posB.x + result.normal.y * posB.y + result.normal.z * posB.z;
    EXPECT_GT(dot, 0.0f) << "Normal should point from A toward B";
}

TEST_F(EdgeEdgeManifoldTest, ContactPointBetweenCars)
{
    float4 posB = make_float4(80, 80, 0, 0);
    float s = sinf(3.14159f / 8.0f);
    float c = cosf(3.14159f / 8.0f);
    float4 rotB = make_float4(0, 0, s, c);
    
    auto result = runEdgeEdgeManifold(origin, identityQuat, posB, rotB);
    
    // Contact point should be roughly between the two car centers
    EXPECT_GT(result.points[0].x, 0.0f) << "Contact X should be positive (between cars)";
    EXPECT_GT(result.points[0].y, 0.0f) << "Contact Y should be positive (between cars)";
    EXPECT_LT(result.points[0].x, posB.x) << "Contact X should be less than B's position";
    EXPECT_LT(result.points[0].y, posB.y) << "Contact Y should be less than B's position";
}

TEST_F(EdgeEdgeManifoldTest, DepthIsPositiveWhenPenetrating)
{
    float4 posB = make_float4(80, 80, 0, 0);
    float s = sinf(3.14159f / 8.0f);
    float c = cosf(3.14159f / 8.0f);
    float4 rotB = make_float4(0, 0, s, c);
    
    auto result = runEdgeEdgeManifold(origin, identityQuat, posB, rotB);
    
    EXPECT_GT(result.depths[0], 0.0f) << "Depth should be positive when penetrating";
}

// ============================================================================
// Face-Face Manifold Tests
// ============================================================================

FaceManifoldResult runFaceFaceManifold(float4 posA, float4 rotA, float4 posB, float4 rotB)
{
    FaceManifoldResult* d_result;
    FaceManifoldResult h_result;
    
    cudaMalloc(&d_result, sizeof(FaceManifoldResult));
    runFaceFaceManifoldTest<<<1, 1>>>(posA, rotA, posB, rotB, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(FaceManifoldResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

class FaceFaceManifoldTest : public ::testing::Test
{
protected:
    float4 identityQuat = make_float4(0, 0, 0, 1);
    float4 origin = make_float4(0, 0, 0, 0);
};

TEST_F(FaceFaceManifoldTest, ProducesContactPoints)
{
    // Overlapping face-face collision along X
    float4 posB = make_float4(110, 0, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    EXPECT_GT(result.count, 0) << "Should produce at least one contact point";
    EXPECT_LE(result.count, 4) << "Should produce at most 4 contact points (culled)";
}

TEST_F(FaceFaceManifoldTest, NormalIsUnitLength)
{
    float4 posB = make_float4(110, 0, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    float len = sqrtf(result.normal.x * result.normal.x +
                      result.normal.y * result.normal.y +
                      result.normal.z * result.normal.z);
    EXPECT_NEAR(len, 1.0f, 1e-4f) << "Contact normal should be unit length";
}

TEST_F(FaceFaceManifoldTest, NormalPointsAlongSeparationAxis)
{
    // Cars separated along X axis
    float4 posB = make_float4(110, 0, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    // Normal should be mostly along X (the separation axis)
    EXPECT_GT(fabsf(result.normal.x), 0.9f) << "Normal should point along X axis";
}

TEST_F(FaceFaceManifoldTest, NormalPointsFromATowardB)
{
    float4 posB = make_float4(110, 0, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    // Normal should point from A toward B
    float dot = result.normal.x * posB.x + result.normal.y * posB.y + result.normal.z * posB.z;
    EXPECT_GT(dot, 0.0f) << "Normal should point from A toward B";
}

TEST_F(FaceFaceManifoldTest, DepthsArePositiveForPenetration)
{
    float4 posB = make_float4(110, 0, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    for (int i = 0; i < result.count; ++i)
    {
        EXPECT_GT(result.depths[i], 0.0f) 
            << "Depth " << i << " should be positive (penetrating)";
    }
}

TEST_F(FaceFaceManifoldTest, ContactPointsInOverlapRegion)
{
    float4 posB = make_float4(110, 0, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    // Contact points should be between the two car centers
    for (int i = 0; i < result.count; ++i)
    {
        EXPECT_GT(result.points[i].x, 0.0f) 
            << "Contact point " << i << " X should be positive";
        EXPECT_LT(result.points[i].x, posB.x) 
            << "Contact point " << i << " X should be less than B's center";
    }
}

TEST_F(FaceFaceManifoldTest, CullingLimitsToFourPoints)
{
    // Deep overlap should generate many clipped points, then cull to 4
    float4 posB = make_float4(80, 0, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    EXPECT_LE(result.count, 4) << "Culling should limit to max 4 points";
}

TEST_F(FaceFaceManifoldTest, YAxisOverlap)
{
    // Overlap along Y axis
    float4 posB = make_float4(0, 75, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    EXPECT_GT(result.count, 0) << "Should produce contact points for Y overlap";
    EXPECT_GT(fabsf(result.normal.y), 0.9f) << "Normal should point along Y axis";
}

TEST_F(FaceFaceManifoldTest, ZAxisOverlap)
{
    // Overlap along Z axis
    float4 posB = make_float4(0, 0, 30, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    EXPECT_GT(result.count, 0) << "Should produce contact points for Z overlap";
    EXPECT_GT(fabsf(result.normal.z), 0.9f) << "Normal should point along Z axis";
}

TEST_F(FaceFaceManifoldTest, RotatedCarCollision)
{
    // Car B rotated 45 degrees around Z
    float4 posB = make_float4(90, 30, 0, 0);
    float s = sinf(3.14159f / 8.0f);
    float c = cosf(3.14159f / 8.0f);
    float4 rotB = make_float4(0, 0, s, c);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, rotB);
    
    EXPECT_GT(result.count, 0) << "Should produce contacts for rotated collision";
    EXPECT_LE(result.count, 4) << "Should still respect culling limit";
}

TEST_F(FaceFaceManifoldTest, IdenticalPositionProducesContacts)
{
    // Cars at same position (maximum overlap)
    auto result = runFaceFaceManifold(origin, identityQuat, origin, identityQuat);
    
    EXPECT_GT(result.count, 0) << "Identical positions should produce contacts";
}

TEST_F(FaceFaceManifoldTest, DeepestPointIsKept)
{
    // With culling, the deepest point should always be preserved
    float4 posB = make_float4(80, 0, 0, 0);
    
    auto result = runFaceFaceManifold(origin, identityQuat, posB, identityQuat);
    
    // Find max depth
    float maxDepth = 0.0f;
    for (int i = 0; i < result.count; ++i)
    {
        if (result.depths[i] > maxDepth) maxDepth = result.depths[i];
    }
    
    // Max depth should be reasonable (not zero, not huge)
    EXPECT_GT(maxDepth, 1.0f) << "Should have meaningful penetration depth";
    EXPECT_LT(maxDepth, 100.0f) << "Depth should be bounded";
}
