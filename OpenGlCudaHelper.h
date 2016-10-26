#pragma once

#include <iostream>
#include <sstream>
#include <GL/glew.h>
#include <GL/freeglut.h>

namespace mf {

	#define checkCudaErrors(val)    checkCudaResult((val), #val, __FILE__, __LINE__)
	

	template<typename T>
	void checkCudaResult(T result, char const *const func, const char *const file, int const line) {
		if (result) {
			std::stringstream ss;
			ss << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
				<< " \"" << func << "\"";
			std::cerr << ss.str() << std::endl;
		}
	}

	inline void createVbo(GLuint* vbo, GLenum target, unsigned int size) {
		// create buffer object
		glGenBuffers(1, vbo);
		glBindBuffer(target, *vbo);

		// initialize buffer
		glBufferData(target, size, 0, GL_DYNAMIC_DRAW);
		glBindBuffer(target, 0);

		glutReportErrors();
	}

	inline void deleteVbo(GLuint* vbo) {
		if (*vbo == 0) {
			return;
		}

		glBindBuffer(GL_ARRAY_BUFFER, *vbo);
		glDeleteBuffers(1, vbo);

		*vbo = 0;
		
		glutReportErrors();
	}
		
	inline cudaError_t deleteCudaSharedVbo(GLuint* vbo, cudaGraphicsResource* cudaResource) {
		deleteVbo(vbo);
		
		if (cudaResource == nullptr) {
			return cudaSuccess;
		}
		return cudaGraphicsUnregisterResource(cudaResource);
	}

}