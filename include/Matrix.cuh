#pragma once

#include "Vector.cuh"
#include "CudaCommon.cuh"

struct CUDA_HD __align__(16) Mat3 {
	Vec3 f, r, u;  // column vectors

	// Constructors
	CUDA_HD Mat3();
	CUDA_HD Mat3(Vec3 f_, Vec3 r_, Vec3 u_) : f{f_}, r{r_}, u{u_} {}

	// Static constructors
	CUDA_HD inline static Mat3 Identity() {
		return Mat3(
			Vec3{ 1.0f, 0.0f, 0.0f },
			Vec3{ 0.0f, 1.0f, 0.0f },
			Vec3{ 0.0f, 0.0f, 1.0f }
		);
	}

	CUDA_HD inline static Mat3 FromEulerAngles(float y, float p, float r) {
		float
			cy = cosf(y),  sy = sinf(y),
			cp = cosf(-p), sp = sinf(-p),
			cr = cosf(-r), sr = sinf(-r);

		return Mat3{
			// Forward vector
			Vec3{cy * cp, sy * cp, -sp},

			// Right vector
			Vec3{cy * sp * sr - sy * cr,
				 sy * sp * sr - cy * cr,
				 cp * sr},

			// Up vector
			Vec3{cy * sp * cr + sy * sr,
				 sy * sp * cr - cy * sr,
				 cp * cr}
		};
	}

	// Matrix × Vector
	CUDA_HD inline Vec3 dot(const Vec3& v) const {
		return Vec3{
			f.x() * v.x() + r.x() * v.y() + u.x() * v.z(),
			f.y() * v.x() + r.y() * v.y() + u.y() * v.z(),
			f.z() * v.x() + r.z() * v.y() + u.z() * v.z()
		};
	}

	// Matrix × Matrix
	CUDA_HD inline Mat3 dot(const Mat3& m) const {
		return Mat3{
			(*this).dot(m.f),
			(*this).dot(m.r),
			(*this).dot(m.u)
		};
	}

	// Transpose
	CUDA_HD inline Mat3 transpose() const {
		return Mat3{
			Vec3{ f.x(), r.x(), u.x() },
			Vec3{ f.y(), r.y(), u.y() },
			Vec3{ f.z(), r.z(), u.z() }
		};
	}

	// Absolute value of each element
	CUDA_HD inline Mat3 absolute() const {
		return Mat3{
			f.absolute(),
			r.absolute(),
			u.absolute()
		};
	}
};
