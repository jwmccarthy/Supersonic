#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "CudaMath.cuh"
#include "CudaCommon.hpp"

// Helper to compare floats with tolerance
constexpr float EPSILON = 1e-5f;

bool approxEqual(float a, float b, float eps = EPSILON)
{
    return fabsf(a - b) < eps;
}

bool approxEqual(float4 a, float4 b, float eps = EPSILON)
{
    return approxEqual(a.x, b.x, eps) &&
           approxEqual(a.y, b.y, eps) &&
           approxEqual(a.z, b.z, eps) &&
           approxEqual(a.w, b.w, eps);
}

// ============================================================================
// Sign function tests
// ============================================================================

__global__ void testSignKernel(float* input, float* output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = sign(input[idx]);
    }
}

TEST(SignTest, PositiveValues)
{
    float h_input[] = {1.0f, 0.5f, 100.0f, 0.001f};
    float h_output[4];
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 4 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 4 * sizeof(float), cudaMemcpyHostToDevice));

    testSignKernel<<<1, 4>>>(d_input, d_output, 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; i++)
    {
        EXPECT_FLOAT_EQ(h_output[i], 1.0f);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST(SignTest, NegativeValues)
{
    float h_input[] = {-1.0f, -0.5f, -100.0f, -0.001f};
    float h_output[4];
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 4 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 4 * sizeof(float), cudaMemcpyHostToDevice));

    testSignKernel<<<1, 4>>>(d_input, d_output, 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; i++)
    {
        EXPECT_FLOAT_EQ(h_output[i], -1.0f);
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

TEST(SignTest, Zero)
{
    float h_input = 0.0f;
    float h_output;
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice));

    testSignKernel<<<1, 1>>>(d_input, d_output, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_output, 1.0f);

    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================================
// Vec3 tests
// ============================================================================

__global__ void testVec3AddKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec3::add(a[0], b[0]);
}

__global__ void testVec3SubKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec3::sub(a[0], b[0]);
}

__global__ void testVec3MultKernel(float4* v, float s, float4* result)
{
    result[0] = vec3::mult(v[0], s);
}

__global__ void testVec3DotKernel(float4* a, float4* b, float* result)
{
    result[0] = vec3::dot(a[0], b[0]);
}

__global__ void testVec3CrossKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec3::cross(a[0], b[0]);
}

__global__ void testVec3NormKernel(float4* v, float4* result)
{
    result[0] = vec3::norm(v[0]);
}

TEST(Vec3Test, Add)
{
    float4 h_a = make_float4(1.0f, 2.0f, 3.0f, 0.0f);
    float4 h_b = make_float4(4.0f, 5.0f, 6.0f, 0.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3AddKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 5.0f);
    EXPECT_FLOAT_EQ(h_result.y, 7.0f);
    EXPECT_FLOAT_EQ(h_result.z, 9.0f);
    EXPECT_FLOAT_EQ(h_result.w, 0.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, Sub)
{
    float4 h_a = make_float4(5.0f, 7.0f, 9.0f, 0.0f);
    float4 h_b = make_float4(1.0f, 2.0f, 3.0f, 0.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3SubKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 4.0f);
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);
    EXPECT_FLOAT_EQ(h_result.z, 6.0f);
    EXPECT_FLOAT_EQ(h_result.w, 0.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, Mult)
{
    float4 h_v = make_float4(1.0f, 2.0f, 3.0f, 0.0f);
    float h_s = 2.5f;
    float4 h_result;
    float4 *d_v, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3MultKernel<<<1, 1>>>(d_v, h_s, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 2.5f);
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);
    EXPECT_FLOAT_EQ(h_result.z, 7.5f);
    EXPECT_FLOAT_EQ(h_result.w, 0.0f);

    cudaFree(d_v);
    cudaFree(d_result);
}

TEST(Vec3Test, Dot)
{
    float4 h_a = make_float4(1.0f, 2.0f, 3.0f, 0.0f);
    float4 h_b = make_float4(4.0f, 5.0f, 6.0f, 0.0f);
    float h_result;
    float4 *d_a, *d_b;
    float *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3DotKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(h_result, 32.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, DotOrthogonal)
{
    float4 h_a = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 h_b = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    float h_result;
    float4 *d_a, *d_b;
    float *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3DotKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result, 0.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, Cross)
{
    // i x j = k
    float4 h_a = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 h_b = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3CrossKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 0.0f);
    EXPECT_FLOAT_EQ(h_result.y, 0.0f);
    EXPECT_FLOAT_EQ(h_result.z, 1.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, CrossAnticommutative)
{
    // j x i = -k
    float4 h_a = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    float4 h_b = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3CrossKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 0.0f);
    EXPECT_FLOAT_EQ(h_result.y, 0.0f);
    EXPECT_FLOAT_EQ(h_result.z, -1.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, Norm)
{
    float4 h_v = make_float4(3.0f, 4.0f, 0.0f, 0.0f);
    float4 h_result;
    float4 *d_v, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3NormKernel<<<1, 1>>>(d_v, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    // Length is 5, so normalized is (0.6, 0.8, 0)
    EXPECT_NEAR(h_result.x, 0.6f, EPSILON);
    EXPECT_NEAR(h_result.y, 0.8f, EPSILON);
    EXPECT_NEAR(h_result.z, 0.0f, EPSILON);

    // Verify unit length
    float len = sqrtf(h_result.x * h_result.x +
                      h_result.y * h_result.y +
                      h_result.z * h_result.z);
    EXPECT_NEAR(len, 1.0f, EPSILON);

    cudaFree(d_v);
    cudaFree(d_result);
}

TEST(Vec3Test, NormZeroVector)
{
    float4 h_v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 h_result;
    float4 *d_v, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3NormKernel<<<1, 1>>>(d_v, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    // Zero vector should remain zero
    EXPECT_FLOAT_EQ(h_result.x, 0.0f);
    EXPECT_FLOAT_EQ(h_result.y, 0.0f);
    EXPECT_FLOAT_EQ(h_result.z, 0.0f);

    cudaFree(d_v);
    cudaFree(d_result);
}

// ============================================================================
// Vec4 tests
// ============================================================================

__global__ void testVec4AddKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec4::add(a[0], b[0]);
}

__global__ void testVec4SubKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec4::sub(a[0], b[0]);
}

__global__ void testVec4MultKernel(float4* v, float s, float4* result)
{
    result[0] = vec4::mult(v[0], s);
}

__global__ void testVec4DotKernel(float4* a, float4* b, float* result)
{
    result[0] = vec4::dot(a[0], b[0]);
}

TEST(Vec4Test, Add)
{
    float4 h_a = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 h_b = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec4AddKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 6.0f);
    EXPECT_FLOAT_EQ(h_result.y, 8.0f);
    EXPECT_FLOAT_EQ(h_result.z, 10.0f);
    EXPECT_FLOAT_EQ(h_result.w, 12.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Test, Sub)
{
    float4 h_a = make_float4(10.0f, 20.0f, 30.0f, 40.0f);
    float4 h_b = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec4SubKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 9.0f);
    EXPECT_FLOAT_EQ(h_result.y, 18.0f);
    EXPECT_FLOAT_EQ(h_result.z, 27.0f);
    EXPECT_FLOAT_EQ(h_result.w, 36.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Test, Mult)
{
    float4 h_v = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float h_s = 3.0f;
    float4 h_result;
    float4 *d_v, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));

    testVec4MultKernel<<<1, 1>>>(d_v, h_s, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 3.0f);
    EXPECT_FLOAT_EQ(h_result.y, 6.0f);
    EXPECT_FLOAT_EQ(h_result.z, 9.0f);
    EXPECT_FLOAT_EQ(h_result.w, 12.0f);

    cudaFree(d_v);
    cudaFree(d_result);
}

TEST(Vec4Test, Dot)
{
    float4 h_a = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 h_b = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
    float h_result;
    float4 *d_a, *d_b;
    float *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec4DotKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_FLOAT_EQ(h_result, 70.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

// ============================================================================
// Quaternion tests
// ============================================================================

__global__ void testQuatNormKernel(float4* q, float4* result)
{
    result[0] = quat::norm(q[0]);
}

__global__ void testQuatConjKernel(float4* q, float4* result)
{
    result[0] = quat::conj(q[0]);
}

__global__ void testQuatCompKernel(float4* a, float4* b, float4* result)
{
    result[0] = quat::comp(a[0], b[0]);
}

__global__ void testQuatMultKernel(float4* v, float4* q, float4* result, bool inv)
{
    result[0] = quat::mult(v[0], q[0], inv);
}

TEST(QuatTest, Norm)
{
    float4 h_q = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 h_result;
    float4 *d_q, *d_result;

    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_q, &h_q, sizeof(float4), cudaMemcpyHostToDevice));

    testQuatNormKernel<<<1, 1>>>(d_q, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    // Verify unit length
    float len = sqrtf(h_result.x * h_result.x +
                      h_result.y * h_result.y +
                      h_result.z * h_result.z +
                      h_result.w * h_result.w);
    EXPECT_NEAR(len, 1.0f, EPSILON);

    cudaFree(d_q);
    cudaFree(d_result);
}

TEST(QuatTest, NormIdentity)
{
    // Identity quaternion should remain unchanged
    float4 h_q = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 h_result;
    float4 *d_q, *d_result;

    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_q, &h_q, sizeof(float4), cudaMemcpyHostToDevice));

    testQuatNormKernel<<<1, 1>>>(d_q, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_result.x, 0.0f, EPSILON);
    EXPECT_NEAR(h_result.y, 0.0f, EPSILON);
    EXPECT_NEAR(h_result.z, 0.0f, EPSILON);
    EXPECT_NEAR(h_result.w, 1.0f, EPSILON);

    cudaFree(d_q);
    cudaFree(d_result);
}

TEST(QuatTest, Conj)
{
    float4 h_q = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 h_result;
    float4 *d_q, *d_result;

    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_q, &h_q, sizeof(float4), cudaMemcpyHostToDevice));

    testQuatConjKernel<<<1, 1>>>(d_q, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, -1.0f);
    EXPECT_FLOAT_EQ(h_result.y, -2.0f);
    EXPECT_FLOAT_EQ(h_result.z, -3.0f);
    EXPECT_FLOAT_EQ(h_result.w, 4.0f);

    cudaFree(d_q);
    cudaFree(d_result);
}

TEST(QuatTest, CompIdentity)
{
    // Composing with identity should return original
    float4 h_a = make_float4(0.5f, 0.5f, 0.5f, 0.5f);  // Some rotation
    float4 h_identity = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_identity, sizeof(float4), cudaMemcpyHostToDevice));

    testQuatCompKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_result.x, h_a.x, EPSILON);
    EXPECT_NEAR(h_result.y, h_a.y, EPSILON);
    EXPECT_NEAR(h_result.z, h_a.z, EPSILON);
    EXPECT_NEAR(h_result.w, h_a.w, EPSILON);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(QuatTest, MultIdentity)
{
    // Rotating by identity should not change vector
    float4 h_v = make_float4(1.0f, 2.0f, 3.0f, 0.0f);
    float4 h_identity = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 h_result;
    float4 *d_v, *d_q, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, &h_identity, sizeof(float4), cudaMemcpyHostToDevice));

    testQuatMultKernel<<<1, 1>>>(d_v, d_q, d_result, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_result.x, h_v.x, EPSILON);
    EXPECT_NEAR(h_result.y, h_v.y, EPSILON);
    EXPECT_NEAR(h_result.z, h_v.z, EPSILON);

    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_result);
}

TEST(QuatTest, MultRotation90Z)
{
    // 90 degree rotation around Z axis
    // q = (0, 0, sin(45°), cos(45°)) = (0, 0, √2/2, √2/2)
    float sqrt2_2 = sqrtf(2.0f) / 2.0f;
    float4 h_v = make_float4(1.0f, 0.0f, 0.0f, 0.0f);  // X axis
    float4 h_q = make_float4(0.0f, 0.0f, sqrt2_2, sqrt2_2);
    float4 h_result;
    float4 *d_v, *d_q, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, &h_q, sizeof(float4), cudaMemcpyHostToDevice));

    testQuatMultKernel<<<1, 1>>>(d_v, d_q, d_result, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    // X axis rotated 90° around Z becomes Y axis
    EXPECT_NEAR(h_result.x, 0.0f, EPSILON);
    EXPECT_NEAR(h_result.y, 1.0f, EPSILON);
    EXPECT_NEAR(h_result.z, 0.0f, EPSILON);

    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_result);
}

TEST(QuatTest, MultInverseRoundtrip)
{
    // Rotating then inverse rotating should return original
    float sqrt2_2 = sqrtf(2.0f) / 2.0f;
    float4 h_v = make_float4(1.0f, 2.0f, 3.0f, 0.0f);
    float4 h_q = make_float4(0.0f, 0.0f, sqrt2_2, sqrt2_2);
    float4 h_intermediate, h_result;
    float4 *d_v, *d_q, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, &h_q, sizeof(float4), cudaMemcpyHostToDevice));

    // Forward rotation
    testQuatMultKernel<<<1, 1>>>(d_v, d_q, d_result, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_intermediate, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    // Copy intermediate to d_v for inverse rotation
    CUDA_CHECK(cudaMemcpy(d_v, &h_intermediate, sizeof(float4), cudaMemcpyHostToDevice));

    // Inverse rotation
    testQuatMultKernel<<<1, 1>>>(d_v, d_q, d_result, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    // Should be back to original
    EXPECT_NEAR(h_result.x, 1.0f, EPSILON);
    EXPECT_NEAR(h_result.y, 2.0f, EPSILON);
    EXPECT_NEAR(h_result.z, 3.0f, EPSILON);

    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_result);
}
