#pragma once
#include "CommonIncludes.h"
#include <vector>
#include <iterator>
#include <map>


namespace mf {

	template<typename Sorry = int>
	
	class SubdividableSphere {
		protected:
			std::vector<Vector3f> m_vertices;
			std::vector<Vector3i> m_triangles;
			typedef std::pair<int, int> indexPair;


		public:
			SubdividableSphere() { }
			~SubdividableSphere() { }

			size_t verticesCount() const {
				return m_vertices.size();
			}
			
			size_t trianglesCount() const {
				return m_triangles.size();
			}

			const std::vector<Vector3f>& getVertices() const {
				return m_vertices;
			}

			const std::vector<Vector3i>& getTriangles() const {
				return m_triangles;
			}

			// void initAsTetrahedron() {	
			// 	m_vertices.resize(4);
			// 	m_triangles.resize(4);

			// 	float oneNorm = (float)(1.0 / sqrt(3.0));
							
			// 	m_vertices[0] = Vector3f(oneNorm, oneNorm, oneNorm);
			// 	m_vertices[1] = Vector3f(-oneNorm, -oneNorm, oneNorm);
			// 	m_vertices[2] = Vector3f(-oneNorm, oneNorm, -oneNorm);
			// 	m_vertices[3] = Vector3f(oneNorm, -oneNorm, -oneNorm);
				
			// 	m_triangles[0] = Vector3i(0, 1, 2);
			// 	m_triangles[1] = Vector3i(0, 3, 1);
			// 	m_triangles[2] = Vector3i(0, 2, 3);
			// 	m_triangles[3] = Vector3i(1, 3, 2);

			// }

			// http://en.wikipedia.org/wiki/Icosahedron
			void initAsIcosahedron() {
				m_vertices.resize(12);
				m_triangles.resize(20);

				double gr = (1.0 + sqrt(5.0)) / 2.0;  // golden ratio
				
				float norm = (float)(1.0 / sqrt(gr * gr + 1.0));
				float grNorm = (float)(gr * norm);
				float oneNorm = 1 * norm;

				// x = 0, yz plane
				m_vertices[0] = Vector3f(0, grNorm, oneNorm);
				m_vertices[1] = Vector3f(0, grNorm,-oneNorm);
				m_vertices[2] = Vector3f(0, -grNorm, oneNorm);
				m_vertices[3] = Vector3f(0, -grNorm,-oneNorm);
				
				// y = 0, xz plane
				m_vertices[4] = Vector3f(oneNorm, 0, grNorm);
				m_vertices[5] = Vector3f(-oneNorm, 0, grNorm);
				m_vertices[6] = Vector3f(oneNorm, 0, -grNorm);
				m_vertices[7] = Vector3f(-oneNorm, 0, -grNorm);
				
				// z = 0, xy plane
				m_vertices[8] = Vector3f(grNorm, oneNorm, 0);
				m_vertices[9] = Vector3f(grNorm,-oneNorm, 0);
				m_vertices[10] = Vector3f(-grNorm, oneNorm, 0);
				m_vertices[11] = Vector3f(-grNorm,-oneNorm, 0);
				
				m_triangles[0] = Vector3i(0, 1, 8);
				m_triangles[1] = Vector3i(0, 10, 1);
				m_triangles[2] = Vector3i(2, 9, 3);
				m_triangles[3] = Vector3i(2, 3, 11);
				
				m_triangles[4] = Vector3i(4, 5, 0);
				m_triangles[5] = Vector3i(4, 2, 5);
				m_triangles[6] = Vector3i(6, 1, 7);
				m_triangles[7] = Vector3i(6, 7, 3);

				m_triangles[8] = Vector3i(8, 9, 4);
				m_triangles[9] = Vector3i(8, 6, 9);
				m_triangles[10] = Vector3i(10, 5, 11);
				m_triangles[11] = Vector3i(10, 11, 7);
				
				m_triangles[12] = Vector3i(1, 6, 8);
				m_triangles[13] = Vector3i(0, 8, 4);
				
				m_triangles[14] = Vector3i(1, 10, 7);
				m_triangles[15] = Vector3i(0, 5, 10);
				
				m_triangles[16] = Vector3i(3, 9, 6);
				m_triangles[17] = Vector3i(2, 4, 9);

				m_triangles[18] = Vector3i(3, 7, 11);
				m_triangles[19] = Vector3i(2, 11, 5);

			}
			
			void subdivide() {		
				assert(m_vertices.size() > 0);
				assert(m_triangles.size() > 0);

				std::vector<Vector3i> newTriangles;
				std::map<indexPair, int> newVertIndicesMap;

				for (auto it = m_triangles.begin(); it != m_triangles.end(); ++it) {
					Vector3i indices = *it;
					
					int i1 = indices.x;
					int i2 = indices.y;
					int i3 = indices.z;
					
					int i12 = getMidpointIndex(i1, i2, newVertIndicesMap);
					int i23 = getMidpointIndex(i2, i3, newVertIndicesMap);
					int i13 = getMidpointIndex(i1, i3, newVertIndicesMap);
					
					// add elements at end 
					newTriangles.push_back(Vector3i(i1, i12, i13));
					newTriangles.push_back(Vector3i(i2, i23, i12));
					newTriangles.push_back(Vector3i(i3, i13, i23));
					newTriangles.push_back(Vector3i(i12, i23, i13));

				}
				m_triangles.clear();
				std::copy(newTriangles.begin(), newTriangles.end(), std::back_inserter(m_triangles));
			}


		private:
			int getMidpointIndex(int i1, int i2, std::map<indexPair, int>& indexPairs) {
				if (i1 > i2) {
					int actualI1 = i1;
					i1 = i2;
					i2 = actualI1;
				}

				assert(i1 < i2);
				auto indexIt = indexPairs.find(indexPair(i1, i2));

				if (indexIt != indexPairs.end()) {
					return indexIt->second;
				}
				else {
					Vector3f v1 = m_vertices[i1];
					Vector3f v2 = m_vertices[i2];

					int index = (int)m_vertices.size();
					Vector3f mid = (v1 + v2) / 2.0f;

					m_vertices.push_back(mid);
					indexPairs[indexPair(i1, i2)] = index;
					return index;
				}
			}
	};

}

