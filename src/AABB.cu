// #include "Constants.cuh"
// #include "AABB.cuh"

// // Axis-aligned bounding box calculations

// CUDA_HD void calculateCarAABB(const Car& car, Vec3& minAABB, Vec3& maxAABB) {
//     // Get car half extents (dimensions)
//     Vec3 halfExtents = OCTANE_HALF_EXTENTS;

//     // Get car rotation matrix
//     Mat3 rotation = car.rotation;

//     // Convert to absolute rotation matrix
//     Mat3 absRotation = rotation.absolute();

//     // Transpose prior to dot product
//     Mat3 absRotationT = absRotation.transpose();

//     // Dot with half-extents to get them in world axes
//     Vec3 worldExtents = absRotationT.dot(halfExtents) + CAR_MARGIN;

//     // Construct AABB bounds
//     minAABB = car.position - worldExtents;
//     maxAABB = car.position + worldExtents;
// }

// CUDA_HD void calculateBallAABB(const Ball& ball, Vec3& minAABB, Vec3& maxAABB) {
//     // Get radius with margin
//     float radius = BALL_RADIUS + BALL_MARGIN;

//     // Construct AABB bounds
//     minAABB = ball.position - radius;
//     maxAABB = ball.position + radius;
// }

// CUDA_HD void calculateBoostPadAABB(const BoostPad& pad, Vec3& minAABB, Vec3& maxAABB) {
//     // Boost pad interaction extents
//     Vec3 padExtents;

//     if (pad.isBig) {
//         // Large boost pad bounds
//         padExtents = BOOST_PAD_LARGE_EXTENTS;
//     } else {
//         // Small boost pad bounds
//         padExtents = BOOST_PAD_SMALL_EXTENTS;
//     }

//     // Construct AABB bounds
//     minAABB = pad.position - padExtents;
//     maxAABB = pad.position + padExtents;
// }

// CUDA_HD void calculateTriangleAABB(const Vec3* vertices, Vec3& minAABB, Vec3& maxAABB) {
//     // Start with first vertex
//     minAABB = vertices[0];
//     maxAABB = vertices[0];

//     #pragma unroll
//     for (int i = 1; i < 3; i++) {
//         // Check each vertex for new min/max coordinates
//         minAABB = minAABB.min(vertices[i]);
//         maxAABB = maxAABB.max(vertices[i]);
//     }

//     // Add margin to bounds
//     minAABB -= TRI_MARGIN;
//     maxAABB += TRI_MARGIN;
// }