#include "CollisionShape.h"
#include "BoxShape.h"
#include "SphereShape.h"

CUDA_BOTH float CollisionShape::GetMargin() const {
    switch (shapeType) {
    case BOX_SHAPE:
        return ((BoxShape*)this)->GetMargin();
    case SPHERE_SHAPE:
        return ((SphereShape*)this)->GetMargin();
    }
}

CUDA_BOTH void CollisionShape::GetAabb(const CudaTransform& t, CudaVec& aabbMin,CudaVec& aabbMax) const {
    // Faster to recalculate sphere
    if (shapeType == SPHERE_SHAPE) {
        return ((SphereShape*)this)->GetAabb(t, aabbMin, aabbMax);
    }

    if (t != aabbTransCache || !aabbCached) {
        switch(shapeType) {
        case BOX_SHAPE:
            ((BoxShape*)this)->GetAabb(t, aabbMin, aabbMax);
            break;
        }
    } else {
        aabbMin = aabbMinCache;
        aabbMax = aabbMaxCache;
    }
}

CUDA_BOTH void CollisionShape::GetBoundingSphere(CudaVec& center, float& radius) const {
    if (shapeType == SPHERE_SHAPE) {
        center = CudaVec();
        radius = 0.08f;
        // radius = ((SphereShape*)this)->GetRadius() + 0.08;
    } else {
        CudaTransform tr = CudaTransform::GetIdentity();
        CudaVec aabbMin, aabbMax;
        GetAabb(tr, aabbMin, aabbMax);
        radius = (aabbMin - aabbMax).Length() * 0.5f;
        center = (aabbMin + aabbMax) * 0.5f;
    }
}

CUDA_BOTH void CollisionShape::CalculateTemporalAabb(
    const CudaTransform& curTrans,
    const CudaVec& vel,
    const CudaVec& angVel,
    float timeStep,
    CudaVec& aabbMin,
    CudaVec& aabbMax
) const {
    // Start w/ static aabb
    GetAabb(curTrans, aabbMin, aabbMax);

    // Add linear motion
    CudaVec linMot = vel * timeStep;
    aabbMin += CudaVec::Min(linMot, CudaVec(0, 0, 0));
    aabbMax += CudaVec::Max(linMot, CudaVec(0, 0, 0));

    // Add conservative angular motion
    float angMot = angVel.Length() * GetAngularMotionDisc() * timeStep;
    CudaVec angularMotion3d(angMot, angMot, angMot);

    aabbMin -= angularMotion3d;
    aabbMax += angularMotion3d;
}

CUDA_BOTH float CollisionShape::GetAngularMotionDisc() const {
    CudaVec center;
    float disc;
    GetBoundingSphere(center, disc);
    disc += center.Length();
    return disc;
}

CUDA_BOTH float CollisionShape::GetContactBreakingThreshold(float defaultFactor) const {
    return GetAngularMotionDisc() * defaultFactor;
}