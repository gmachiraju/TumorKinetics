#pragma once
#include <algorithm>
#include <assert.h>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

typedef glm::ivec2 Vector2i;
typedef glm::ivec3 Vector3i;
typedef glm::mediump_vec3 Vector3f;
typedef glm::mediump_vec4 Vector4f;
typedef glm::tvec4<unsigned char> Vector4c;

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

