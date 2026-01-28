#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>

#include "CudaMath.cuh"
#include "CudaCommon.cuh"

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

__global__ void testQuatToWorldKernel(float4* v, float4* q, float4* result)
{
    result[0] = quat::toWorld(v[0], q[0]);
}

__global__ void testQuatToLocalKernel(float4* v, float4* q, float4* result)
{
    result[0] = quat::toLocal(v[0], q[0]);
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

    testQuatToWorldKernel<<<1, 1>>>(d_v, d_q, d_result);
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

    testQuatToWorldKernel<<<1, 1>>>(d_v, d_q, d_result);
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
    testQuatToWorldKernel<<<1, 1>>>(d_v, d_q, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_intermediate, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    // Copy intermediate to d_v for inverse rotation
    CUDA_CHECK(cudaMemcpy(d_v, &h_intermediate, sizeof(float4), cudaMemcpyHostToDevice));

    // Inverse rotation
    testQuatToLocalKernel<<<1, 1>>>(d_v, d_q, d_result);
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

// ============================================================================
// Global min/max tests
// ============================================================================

__global__ void testScalarMinKernel(float* a, float* b, float* result)
{
    result[0] = ::min(a[0], b[0]);
}

__global__ void testScalarMaxKernel(float* a, float* b, float* result)
{
    result[0] = ::max(a[0], b[0]);
}

__global__ void testScalarMinIntKernel(int* a, int* b, int* result)
{
    result[0] = ::min(a[0], b[0]);
}

__global__ void testScalarMaxIntKernel(int* a, int* b, int* result)
{
    result[0] = ::max(a[0], b[0]);
}

TEST(ScalarTest, MinFloat)
{
    float h_a = 5.0f, h_b = 3.0f, h_result;
    float *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice));

    testScalarMinKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result, 3.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(ScalarTest, MaxFloat)
{
    float h_a = 5.0f, h_b = 3.0f, h_result;
    float *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice));

    testScalarMaxKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result, 5.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(ScalarTest, MinInt)
{
    int h_a = 10, h_b = -5, h_result;
    int *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));

    testScalarMinIntKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result, -5);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(ScalarTest, MaxInt)
{
    int h_a = 10, h_b = -5, h_result;
    int *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));

    testScalarMaxIntKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result, 10);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

// ============================================================================
// Vec3 additional tests (div, min, max, clamp)
// ============================================================================

__global__ void testVec3DivKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec3::div(a[0], b[0]);
}

__global__ void testVec3MinKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec3::min(a[0], b[0]);
}

__global__ void testVec3MaxKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec3::max(a[0], b[0]);
}

__global__ void testVec3ClampKernel(float4* v, float4* lo, float4* hi, float4* result)
{
    result[0] = vec3::clamp(v[0], lo[0], hi[0]);
}

TEST(Vec3Test, Div)
{
    float4 h_a = make_float4(10.0f, 20.0f, 30.0f, 0.0f);
    float4 h_b = make_float4(2.0f, 4.0f, 5.0f, 1.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3DivKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 5.0f);
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);
    EXPECT_FLOAT_EQ(h_result.z, 6.0f);
    EXPECT_FLOAT_EQ(h_result.w, 0.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, Min)
{
    float4 h_a = make_float4(1.0f, 5.0f, 3.0f, 0.0f);
    float4 h_b = make_float4(4.0f, 2.0f, 6.0f, 0.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3MinKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 1.0f);
    EXPECT_FLOAT_EQ(h_result.y, 2.0f);
    EXPECT_FLOAT_EQ(h_result.z, 3.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, Max)
{
    float4 h_a = make_float4(1.0f, 5.0f, 3.0f, 0.0f);
    float4 h_b = make_float4(4.0f, 2.0f, 6.0f, 0.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3MaxKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 4.0f);
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);
    EXPECT_FLOAT_EQ(h_result.z, 6.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Test, Clamp)
{
    float4 h_v = make_float4(-1.0f, 5.0f, 50.0f, 0.0f);
    float4 h_lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 h_hi = make_float4(10.0f, 10.0f, 10.0f, 0.0f);
    float4 h_result;
    float4 *d_v, *d_lo, *d_hi, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_lo, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_hi, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lo, &h_lo, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hi, &h_hi, sizeof(float4), cudaMemcpyHostToDevice));

    testVec3ClampKernel<<<1, 1>>>(d_v, d_lo, d_hi, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 0.0f);   // Clamped up from -1
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);   // Within range
    EXPECT_FLOAT_EQ(h_result.z, 10.0f);  // Clamped down from 50

    cudaFree(d_v);
    cudaFree(d_lo);
    cudaFree(d_hi);
    cudaFree(d_result);
}

// ============================================================================
// Vec3 float3 tests
// ============================================================================

__global__ void testVec3AddFloat3Kernel(float3* a, float3* b, float3* result)
{
    result[0] = vec3::add(a[0], b[0]);
}

__global__ void testVec3SubFloat3Kernel(float3* a, float3* b, float3* result)
{
    result[0] = vec3::sub(a[0], b[0]);
}

__global__ void testVec3MultFloat3Kernel(float3* v, float s, float3* result)
{
    result[0] = vec3::mult(v[0], s);
}

__global__ void testVec3DotFloat3Kernel(float3* a, float3* b, float* result)
{
    result[0] = vec3::dot(a[0], b[0]);
}

__global__ void testVec3CrossFloat3Kernel(float3* a, float3* b, float3* result)
{
    result[0] = vec3::cross(a[0], b[0]);
}

__global__ void testVec3NormFloat3Kernel(float3* v, float3* result)
{
    result[0] = vec3::norm(v[0]);
}

TEST(Vec3Float3Test, Add)
{
    float3 h_a = make_float3(1.0f, 2.0f, 3.0f);
    float3 h_b = make_float3(4.0f, 5.0f, 6.0f);
    float3 h_result;
    float3 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float3), cudaMemcpyHostToDevice));

    testVec3AddFloat3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float3), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 5.0f);
    EXPECT_FLOAT_EQ(h_result.y, 7.0f);
    EXPECT_FLOAT_EQ(h_result.z, 9.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Float3Test, Sub)
{
    float3 h_a = make_float3(5.0f, 7.0f, 9.0f);
    float3 h_b = make_float3(1.0f, 2.0f, 3.0f);
    float3 h_result;
    float3 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float3), cudaMemcpyHostToDevice));

    testVec3SubFloat3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float3), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 4.0f);
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);
    EXPECT_FLOAT_EQ(h_result.z, 6.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Float3Test, Mult)
{
    float3 h_v = make_float3(1.0f, 2.0f, 3.0f);
    float h_s = 2.5f;
    float3 h_result;
    float3 *d_v, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float3), cudaMemcpyHostToDevice));

    testVec3MultFloat3Kernel<<<1, 1>>>(d_v, h_s, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float3), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 2.5f);
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);
    EXPECT_FLOAT_EQ(h_result.z, 7.5f);

    cudaFree(d_v);
    cudaFree(d_result);
}

TEST(Vec3Float3Test, Dot)
{
    float3 h_a = make_float3(1.0f, 2.0f, 3.0f);
    float3 h_b = make_float3(4.0f, 5.0f, 6.0f);
    float h_result;
    float3 *d_a, *d_b;
    float *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float3), cudaMemcpyHostToDevice));

    testVec3DotFloat3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // 1*4 + 2*5 + 3*6 = 32
    EXPECT_FLOAT_EQ(h_result, 32.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Float3Test, Cross)
{
    // i x j = k
    float3 h_a = make_float3(1.0f, 0.0f, 0.0f);
    float3 h_b = make_float3(0.0f, 1.0f, 0.0f);
    float3 h_result;
    float3 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float3), cudaMemcpyHostToDevice));

    testVec3CrossFloat3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float3), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 0.0f);
    EXPECT_FLOAT_EQ(h_result.y, 0.0f);
    EXPECT_FLOAT_EQ(h_result.z, 1.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Float3Test, Norm)
{
    float3 h_v = make_float3(3.0f, 4.0f, 0.0f);
    float3 h_result;
    float3 *d_v, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float3), cudaMemcpyHostToDevice));

    testVec3NormFloat3Kernel<<<1, 1>>>(d_v, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float3), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_result.x, 0.6f, EPSILON);
    EXPECT_NEAR(h_result.y, 0.8f, EPSILON);
    EXPECT_NEAR(h_result.z, 0.0f, EPSILON);

    cudaFree(d_v);
    cudaFree(d_result);
}

// ============================================================================
// Vec3 int3 tests
// ============================================================================

__global__ void testVec3AddInt3Kernel(int3* a, int3* b, int3* result)
{
    result[0] = vec3::add(a[0], b[0]);
}

__global__ void testVec3SubInt3Kernel(int3* a, int3* b, int3* result)
{
    result[0] = vec3::sub(a[0], b[0]);
}

__global__ void testVec3MultInt3Kernel(int3* v, int s, int3* result)
{
    result[0] = vec3::mult(v[0], s);
}

__global__ void testVec3DotInt3Kernel(int3* a, int3* b, int* result)
{
    result[0] = vec3::dot(a[0], b[0]);
}

__global__ void testVec3ProdKernel(int3* v, int* result)
{
    result[0] = vec3::prod(v[0]);
}

__global__ void testVec3MinInt3Kernel(int3* a, int3* b, int3* result)
{
    result[0] = vec3::min(a[0], b[0]);
}

__global__ void testVec3MaxInt3Kernel(int3* a, int3* b, int3* result)
{
    result[0] = vec3::max(a[0], b[0]);
}

TEST(Vec3Int3Test, Add)
{
    int3 h_a = make_int3(1, 2, 3);
    int3 h_b = make_int3(4, 5, 6);
    int3 h_result;
    int3 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int3)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int3), cudaMemcpyHostToDevice));

    testVec3AddInt3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int3), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 5);
    EXPECT_EQ(h_result.y, 7);
    EXPECT_EQ(h_result.z, 9);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Int3Test, Sub)
{
    int3 h_a = make_int3(10, 20, 30);
    int3 h_b = make_int3(1, 2, 3);
    int3 h_result;
    int3 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int3)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int3), cudaMemcpyHostToDevice));

    testVec3SubInt3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int3), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 9);
    EXPECT_EQ(h_result.y, 18);
    EXPECT_EQ(h_result.z, 27);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Int3Test, Mult)
{
    int3 h_v = make_int3(2, 3, 4);
    int h_s = 5;
    int3 h_result;
    int3 *d_v, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int3)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(int3), cudaMemcpyHostToDevice));

    testVec3MultInt3Kernel<<<1, 1>>>(d_v, h_s, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int3), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 10);
    EXPECT_EQ(h_result.y, 15);
    EXPECT_EQ(h_result.z, 20);

    cudaFree(d_v);
    cudaFree(d_result);
}

TEST(Vec3Int3Test, Dot)
{
    int3 h_a = make_int3(1, 2, 3);
    int3 h_b = make_int3(4, 5, 6);
    int h_result;
    int3 *d_a, *d_b;
    int *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int3), cudaMemcpyHostToDevice));

    testVec3DotInt3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // 1*4 + 2*5 + 3*6 = 32
    EXPECT_EQ(h_result, 32);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Int3Test, Prod)
{
    int3 h_v = make_int3(2, 3, 4);
    int h_result;
    int3 *d_v;
    int *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(int3), cudaMemcpyHostToDevice));

    testVec3ProdKernel<<<1, 1>>>(d_v, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // 2 * 3 * 4 = 24
    EXPECT_EQ(h_result, 24);

    cudaFree(d_v);
    cudaFree(d_result);
}

TEST(Vec3Int3Test, Min)
{
    int3 h_a = make_int3(1, 5, 3);
    int3 h_b = make_int3(4, 2, 6);
    int3 h_result;
    int3 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int3)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int3), cudaMemcpyHostToDevice));

    testVec3MinInt3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int3), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 1);
    EXPECT_EQ(h_result.y, 2);
    EXPECT_EQ(h_result.z, 3);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec3Int3Test, Max)
{
    int3 h_a = make_int3(1, 5, 3);
    int3 h_b = make_int3(4, 2, 6);
    int3 h_result;
    int3 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int3)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int3), cudaMemcpyHostToDevice));

    testVec3MaxInt3Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int3), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 4);
    EXPECT_EQ(h_result.y, 5);
    EXPECT_EQ(h_result.z, 6);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

// ============================================================================
// Vec4 additional tests (div, min, max, clamp)
// ============================================================================

__global__ void testVec4DivKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec4::div(a[0], b[0]);
}

__global__ void testVec4MinKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec4::min(a[0], b[0]);
}

__global__ void testVec4MaxKernel(float4* a, float4* b, float4* result)
{
    result[0] = vec4::max(a[0], b[0]);
}

__global__ void testVec4ClampKernel(float4* v, float4* lo, float4* hi, float4* result)
{
    result[0] = vec4::clamp(v[0], lo[0], hi[0]);
}

TEST(Vec4Test, Div)
{
    float4 h_a = make_float4(10.0f, 20.0f, 30.0f, 40.0f);
    float4 h_b = make_float4(2.0f, 4.0f, 5.0f, 8.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec4DivKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 5.0f);
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);
    EXPECT_FLOAT_EQ(h_result.z, 6.0f);
    EXPECT_FLOAT_EQ(h_result.w, 5.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Test, Min)
{
    float4 h_a = make_float4(1.0f, 5.0f, 3.0f, 8.0f);
    float4 h_b = make_float4(4.0f, 2.0f, 6.0f, 1.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec4MinKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 1.0f);
    EXPECT_FLOAT_EQ(h_result.y, 2.0f);
    EXPECT_FLOAT_EQ(h_result.z, 3.0f);
    EXPECT_FLOAT_EQ(h_result.w, 1.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Test, Max)
{
    float4 h_a = make_float4(1.0f, 5.0f, 3.0f, 8.0f);
    float4 h_b = make_float4(4.0f, 2.0f, 6.0f, 1.0f);
    float4 h_result;
    float4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(float4), cudaMemcpyHostToDevice));

    testVec4MaxKernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 4.0f);
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);
    EXPECT_FLOAT_EQ(h_result.z, 6.0f);
    EXPECT_FLOAT_EQ(h_result.w, 8.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Test, Clamp)
{
    float4 h_v = make_float4(-1.0f, 5.0f, 50.0f, 100.0f);
    float4 h_lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 h_hi = make_float4(10.0f, 10.0f, 10.0f, 10.0f);
    float4 h_result;
    float4 *d_v, *d_lo, *d_hi, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_lo, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_hi, sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lo, &h_lo, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hi, &h_hi, sizeof(float4), cudaMemcpyHostToDevice));

    testVec4ClampKernel<<<1, 1>>>(d_v, d_lo, d_hi, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float4), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_result.x, 0.0f);   // Clamped up from -1
    EXPECT_FLOAT_EQ(h_result.y, 5.0f);   // Within range
    EXPECT_FLOAT_EQ(h_result.z, 10.0f);  // Clamped down from 50
    EXPECT_FLOAT_EQ(h_result.w, 10.0f);  // Clamped down from 100

    cudaFree(d_v);
    cudaFree(d_lo);
    cudaFree(d_hi);
    cudaFree(d_result);
}

// ============================================================================
// Vec4 int4 tests
// ============================================================================

__global__ void testVec4AddInt4Kernel(int4* a, int4* b, int4* result)
{
    result[0] = vec4::add(a[0], b[0]);
}

__global__ void testVec4SubInt4Kernel(int4* a, int4* b, int4* result)
{
    result[0] = vec4::sub(a[0], b[0]);
}

__global__ void testVec4MultInt4Kernel(int4* v, int s, int4* result)
{
    result[0] = vec4::mult(v[0], s);
}

__global__ void testVec4DotInt4Kernel(int4* a, int4* b, int* result)
{
    result[0] = vec4::dot(a[0], b[0]);
}

__global__ void testVec4MinInt4Kernel(int4* a, int4* b, int4* result)
{
    result[0] = vec4::min(a[0], b[0]);
}

__global__ void testVec4MaxInt4Kernel(int4* a, int4* b, int4* result)
{
    result[0] = vec4::max(a[0], b[0]);
}

TEST(Vec4Int4Test, Add)
{
    int4 h_a = make_int4(1, 2, 3, 4);
    int4 h_b = make_int4(5, 6, 7, 8);
    int4 h_result;
    int4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int4), cudaMemcpyHostToDevice));

    testVec4AddInt4Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int4), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 6);
    EXPECT_EQ(h_result.y, 8);
    EXPECT_EQ(h_result.z, 10);
    EXPECT_EQ(h_result.w, 12);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Int4Test, Sub)
{
    int4 h_a = make_int4(10, 20, 30, 40);
    int4 h_b = make_int4(1, 2, 3, 4);
    int4 h_result;
    int4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int4), cudaMemcpyHostToDevice));

    testVec4SubInt4Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int4), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 9);
    EXPECT_EQ(h_result.y, 18);
    EXPECT_EQ(h_result.z, 27);
    EXPECT_EQ(h_result.w, 36);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Int4Test, Mult)
{
    int4 h_v = make_int4(1, 2, 3, 4);
    int h_s = 3;
    int4 h_result;
    int4 *d_v, *d_result;

    CUDA_CHECK(cudaMalloc(&d_v, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int4)));
    CUDA_CHECK(cudaMemcpy(d_v, &h_v, sizeof(int4), cudaMemcpyHostToDevice));

    testVec4MultInt4Kernel<<<1, 1>>>(d_v, h_s, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int4), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 3);
    EXPECT_EQ(h_result.y, 6);
    EXPECT_EQ(h_result.z, 9);
    EXPECT_EQ(h_result.w, 12);

    cudaFree(d_v);
    cudaFree(d_result);
}

TEST(Vec4Int4Test, Dot)
{
    int4 h_a = make_int4(1, 2, 3, 4);
    int4 h_b = make_int4(5, 6, 7, 8);
    int h_result;
    int4 *d_a, *d_b;
    int *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int4), cudaMemcpyHostToDevice));

    testVec4DotInt4Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_EQ(h_result, 70);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Int4Test, Min)
{
    int4 h_a = make_int4(1, 5, 3, 8);
    int4 h_b = make_int4(4, 2, 6, 1);
    int4 h_result;
    int4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int4), cudaMemcpyHostToDevice));

    testVec4MinInt4Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int4), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 1);
    EXPECT_EQ(h_result.y, 2);
    EXPECT_EQ(h_result.z, 3);
    EXPECT_EQ(h_result.w, 1);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

TEST(Vec4Int4Test, Max)
{
    int4 h_a = make_int4(1, 5, 3, 8);
    int4 h_b = make_int4(4, 2, 6, 1);
    int4 h_result;
    int4 *d_a, *d_b, *d_result;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int4)));
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(int4), cudaMemcpyHostToDevice));

    testVec4MaxInt4Kernel<<<1, 1>>>(d_a, d_b, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int4), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_result.x, 4);
    EXPECT_EQ(h_result.y, 5);
    EXPECT_EQ(h_result.z, 6);
    EXPECT_EQ(h_result.w, 8);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}
