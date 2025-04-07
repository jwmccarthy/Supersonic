#pragma once

#include "CollisionShape.h"
#include "thrust/device_vector.h"

struct Face {
    thrust::device_vector<int> indices;
    float plane[4];
};

struct BoxShape : CollisionShape {
    int shapeType = BOX_SHAPE;

    // Box shape components
    thrust::device_vector<Face> faces;
    thrust::device_vector<CudaVec> vertices;
    thrust::device_vector<CudaVec> uniqueEdges;

    // Shape extent
    CudaVec localCenter;
    CudaVec extents;
    float radius;
    CudaVec mC;
    CudaVec mE;

    CUDA_BOTH ~BoxShape() override = default;

    CUDA_BOTH void Project(
        const CudaTransform& t, 
        const CudaVec& d, 
        float& minProj, 
        float& maxProj,
        CudaVec& witnesPtMin,
        CudaVec& witnesPtMax
    ) const;

    CUDA_BOTH CudaVec GetHalfExtentsWithMargin() const;
};