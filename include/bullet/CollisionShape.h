#pragma once

#include "MathTypes.h"

struct alignas(16) CollisionShape {
    enum ShapeType {
        NONE_SHAPE, 
        SPHERE_SHAPE, 
        BOX_SHAPE
    };

    int shapeType = NONE_SHAPE;

    mutable bool aabbCached = false;
    mutable CudaVec aabbMinCache;
    mutable CudaVec aabbMaxCache;
    mutable CudaTransform aabbTransCache;

    CUDA_BOTH CollisionShape() = default;
    CUDA_BOTH virtual ~CollisionShape() = default;

    // Bounding volumes
    CUDA_BOTH float GetMargin() const;
    CUDA_BOTH void GetAabb(const CudaTransform& t, CudaVec& aabbMin, CudaVec& aabbMax) const;
    CUDA_BOTH void GetBoundingSphere(CudaVec& center, float& radius) const;

    CUDA_BOTH void CalculateTemporalAabb(
        const CudaTransform& curTrans, 
        const CudaVec& vel, 
        const CudaVec& angVel, 
        float timeStep, 
        CudaVec& aabbMin, 
        CudaVec& aabbMax
    ) const;
    
    // Utility methods
    CUDA_BOTH float GetAngularMotionDisc() const;
    CUDA_BOTH float GetContactBreakingThreshold(float defaultFactor) const;

    CUDA_BOTH virtual void CalculateLocalInertia(float mass, CudaVec& inertia) const = 0;
};