#pragma once

#include "CommonIncludes.h"
#include "MathHelper.h"

namespace mf {
	
	template<typename Sorry = int>
	
	class Plane {
		public:
			Plane(float displac, Vector3f norm) : displacement(displac), normal(norm) {
				assert(isAlmostZero(glm::length(normal) - 1.0f));
			}
			~Plane() { }
			
			const float displacement;
			const Vector3f normal;

			/// Returns signed distance from this plane to given point.
			float countDistanceToPt(Vector3f pt) {
				return pt.x * normal.x + pt.y * normal.y + pt.z * normal.z + displacement;
			}

		private:
	};

	template<class ResultIterator>
	void generateRandomPlanes(size_t count, float displacementDelta, ResultIterator resultIt) {
		assert(displacementDelta >= 0);

		for (size_t i = 0; i < count; ++i) {
			float d = ((2 * displacementDelta) * ((float)rand() / (float)RAND_MAX)) - displacementDelta;

			// not perfectly homogeneous but easy to implement
			Vector3f n = Vector3f((float)rand() / (float)RAND_MAX - 0.5f, (float)rand() / (float)RAND_MAX - 0.5f, (float)rand() / (float)RAND_MAX - 0.5f);
			*resultIt++ = Plane<>(d, glm::normalize(n));
		}
	}

}