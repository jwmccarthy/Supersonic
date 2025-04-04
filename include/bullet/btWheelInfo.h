#pragma once

#include "MathTypes.h"

struct btWheelInfoConstructionInfo
{
	CudaVec m_chassisConnectionCS;
	CudaVec m_wheelDirectionCS;
	CudaVec m_wheelAxleCS;
	float m_suspensionRestLength;
	float m_maxSuspensionTravelCm;
	float m_wheelRadius;

	float m_suspensionStiffness;
	float m_wheelsDampingCompression;
	float m_wheelsDampingRelaxation;
	float m_frictionSlip;
	float m_maxSuspensionForce;
	bool m_bIsFrontWheel;
};

struct btWheelInfo {

    // Set by raycaster
    struct RaycastInfo {
		CudaVec m_contactNormalWS;  // Contact normal
		CudaVec m_contactPointWS;   // Raycast hitpoint
		float m_suspensionLength;
		CudaVec m_hardPointWS;      // Raycast starting point
		CudaVec m_wheelDirectionWS; // Direction in worldspace
		CudaVec m_wheelAxleWS;      // Axle in worldspace
		bool m_isInContact;
		void* m_groundObject;
	};

    RaycastInfo m_raycastInfo;
}