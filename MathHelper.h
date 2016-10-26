#pragma once

namespace mf {
	const float SGL_EPSILON = 1.0E-05f;
	inline bool isAlmostZero(float value) {
		return value < 10.0f * SGL_EPSILON && value > -10.0f * SGL_EPSILON;
	}

}