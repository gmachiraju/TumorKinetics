#include "CommonIncludes.h"
#include "MfIncludes.h"
#include "OpenGlCudaHelper.h"
#include "TimerHelper.h"
#include "RandomFaultsKernel.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


namespace mf {

	MfSphere sphere;
	
	bool wireframe = true;
	bool hiContrastMode = false;
	bool backFaceCulling = true;
	int screenWidth = 1024;
	int screenHeight = 768;

	typedef StopWatchInterface StopWatch;
	StopWatch* stopWatch;
	StopWatch* kernelStopWatch;
	
	bool lastProcessOnGpu = false;
	float lastProcessTime = 0;
	float lastKernelProcessTime = 0;

	// parameters
	size_t planesPerIteration = 1024;
	float displacement = 0.0004f;
	bool planesOnlyInCenter = false;
	bool menuVisible = true;
	std::vector<Vector4f> colorGradient;

	// device data
	GLuint vertsVbo = 0;
	GLuint indexVbo = 0;
	GLuint colorsVbo = 0;
	//GLuint normalVbo = 0;
	cudaGraphicsResource* cudaVertsVboResource = nullptr;
	cudaGraphicsResource* cudaColorsVboResource = nullptr;
	
	float3* d_vertices = nullptr;
	uchar4* d_colors = nullptr;
	float4* d_planes = nullptr;
	float4* d_gradient = nullptr;

	// mouse controls
	Vector2i mouseOld;
	int mouseButtons = 0;
	Vector3f rotate = Vector3f(0.0f, 0.0f, 0.0f);
	Vector3f translate = Vector3f(0.0f, 0.0f, -2.5f);


	void drawString(float x, float y, float z, const std::string& text) {
		glRasterPos3f(x, y, z);
		for (size_t i = 0; i < text.size(); i++) {
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13, text[i]);
		}
	}


	void drawControls() {
		float i = 0.5;
		float incI = 14;
		std::stringstream ss;

		ss.str("");
		ss << "        Mesh subdivisions:  " << sphere.verticesCount() << " verts";
		drawString(10, ++i * incI, 0, ss.str());
		
		ss.str("");
		ss << "[+] [-] Cut planes per iteration: " << planesPerIteration;
		drawString(10, ++i * incI, 0, ss.str());
		
		// should be dependent on N_T
		ss.str("");
		ss << "        Displacement amount: " << displacement;
		drawString(10, ++i * incI, 0, ss.str());
		
		drawString(10, ++i * incI, 0, "[space] Do random faults on GPU");
		drawString(10, ++i * incI, 0, "  [c]   Do random faults on CPU");

		ss.str("");
		ss << "  [e]   Cut planes go through origin: " << (planesOnlyInCenter ? "yes" : "no");
		drawString(10, ++i * incI, 0, ss.str());

		drawString(10, ++i * incI, 0, "  [w]   Toggle wireframe");
		ss.str("");

		ss << "  [b]   Toggle back face culling: " << (backFaceCulling ? "yes" : "no");
		drawString(10, ++i * incI, 0, ss.str());
		//drawString(10, ++i * incI, 0, "  [t]   Init as tetrahedron");
		//drawString(10, ++i * incI, 0, "  [i]   Init as icosahedron");
		drawString(10, ++i * incI, 0, "  [r]   Reset elevation and color");
		drawString(10, ++i * incI, 0, "  [h]   High contrast mode");
		drawString(10, ++i * incI, 0, "  [q]   Run benchmark");
		drawString(10, ++i * incI, 0, "  [m]   Hide/show this menu");

		i = screenHeight / incI - 4;

		ss.str("");
		ss << "Last process time (" << (lastProcessOnGpu ? "GPU" : "CPU") << "):";
		drawString(10, ++i * incI, 0, ss.str());

		ss.str("");
		ss << "Core: " << lastKernelProcessTime << " ms";
		drawString(10, ++i * incI, 0, ss.str());

		ss.str("");
		ss << "Total: " << lastProcessTime << " ms (including planes gen and mem transfer)";
		drawString(10, ++i * incI, 0, ss.str());
	}


	// GPU function call
	void generateAndAllocPlanes() {
		std::vector<MfPlane> planes;
		generateRandomPlanes(planesPerIteration, planesOnlyInCenter ? 0.0f : 1.0f, std::back_inserter(planes));

		float4* planesBuff = new float4[planes.size()];
		for (size_t i = 0; i < planes.size(); ++i) {
			const MfPlane& p = planes[i];
			planesBuff[i].w = p.displacement;
			planesBuff[i].x = p.normal.x;
			planesBuff[i].y = p.normal.y;
			planesBuff[i].z = p.normal.z;
		}

		checkCudaErrors(cudaMalloc((void**)&d_planes, planes.size() * sizeof(float4)));
		checkCudaErrors(cudaMemcpy(d_planes, planesBuff, planes.size() * sizeof(float4), cudaMemcpyHostToDevice));

		delete[] planesBuff;
	}


	// GPU function call
	void allocColorGradient() {
		assert(sizeof(float4) == sizeof(float4));

		checkCudaErrors(cudaMalloc((void**)&d_gradient, colorGradient.size() * sizeof(float4)));
		checkCudaErrors(cudaMemcpy(d_gradient, &colorGradient[0], colorGradient.size() * sizeof(float4), cudaMemcpyHostToDevice));
	}
	

	void resetSphereGeometry() {
		size_t bytesCount;
		checkCudaErrors(cudaGraphicsMapResources(1, &cudaVertsVboResource));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &bytesCount, cudaVertsVboResource));

		checkCudaErrors(cudaGraphicsMapResources(1, &cudaColorsVboResource));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &bytesCount, cudaColorsVboResource));
		
		uchar4 color;
		color.x = (unsigned char)(colorGradient[2].x * 255.0f);
		color.y = (unsigned char)(colorGradient[2].y * 255.0f);
		color.z = (unsigned char)(colorGradient[2].z * 255.0f);
		color.w = 0;
		runResetGeometryKernel(d_vertices, (unsigned int)sphere.verticesCount(), d_colors, color);
		
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaVertsVboResource));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaColorsVboResource));

	}
	

	void readVbos(Vector3f* vertices, Vector4c* colors, size_t verticesCount) {
		assert(sizeof(GLfloat) * 3 == sizeof(Vector3f));
		
		glBindBuffer(GL_ARRAY_BUFFER, vertsVbo);
		const Vector3f* vboMappedVertices = (Vector3f*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
		assert(vboMappedVertices != nullptr);
		std::copy(vboMappedVertices, vboMappedVertices + verticesCount, vertices);
		glUnmapBuffer(GL_ARRAY_BUFFER);

		assert(sizeof(GLubyte) * 4 == sizeof(Vector4c));
		
		glBindBuffer(GL_ARRAY_BUFFER, colorsVbo);
		const Vector4c* vboMappedColors = (Vector4c*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
		assert(vboMappedColors != nullptr);
		std::copy(vboMappedColors, vboMappedColors + verticesCount, colors);
		glUnmapBuffer(GL_ARRAY_BUFFER);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}


	void runRandomFaultsOnGpu() {
		stopWatch->reset();
		stopWatch->start();

		generateAndAllocPlanes();
		allocColorGradient();

		// map buffers to CUDA
		size_t bytesCount;
		checkCudaErrors(cudaGraphicsMapResources(1, &cudaVertsVboResource));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &bytesCount, cudaVertsVboResource));

		checkCudaErrors(cudaGraphicsMapResources(1, &cudaColorsVboResource));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &bytesCount, cudaColorsVboResource));
		
		// run random faults kernel
		kernelStopWatch->reset();
		runFaultsKernel(d_vertices, (unsigned int)sphere.verticesCount(), d_planes, 0, planesPerIteration, displacement, d_colors, d_gradient, colorGradient.size(), kernelStopWatch);
		lastKernelProcessTime = kernelStopWatch->getTime();
		
		// unmap buffers and free memory
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaVertsVboResource));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaColorsVboResource));
		
		checkCudaErrors(cudaFree(d_planes));
		checkCudaErrors(cudaFree(d_gradient));

		stopWatch->stop();
		lastProcessTime = stopWatch->getTime();
		lastProcessOnGpu = true;
	}
	

	void runRandomFaultsOnCpu() {
		stopWatch->reset();
		stopWatch->start();

		size_t vertsCount = sphere.verticesCount();
		Vector3f* vertices = new Vector3f[vertsCount];
		Vector4c* colors = new Vector4c[vertsCount];
		readVbos(vertices, colors, vertsCount);
		
		std::vector<MfPlane> planes;
		generateRandomPlanes(planesPerIteration, planesOnlyInCenter ? 0.0f : 1.0f, std::back_inserter(planes));

		kernelStopWatch->reset();
		kernelStopWatch->start();

		for (size_t vertexIndex = 0; vertexIndex < vertsCount; ++vertexIndex) {
			Vector3f v = vertices[vertexIndex];
			int displacementSteps = 0;
	
			// compute contributions for displacement
			for (size_t planeIndex = 0; planeIndex < planesPerIteration; ++planeIndex) {
				MfPlane plane = planes[planeIndex];
				if (glm::dot(v, plane.normal) + plane.displacement > 0) {
					++displacementSteps;
				}
				else {
					--displacementSteps;
				}
			}

			// displace vector
			v += (displacement * displacementSteps) * glm::normalize(v);
			vertices[vertexIndex] = v;

			float elevation = glm::length(v);
			// count color
			size_t gradientLength = colorGradient.size();
			size_t i = 0;
			while (i < gradientLength && colorGradient[i].w < elevation) {
				++i;
			}
	
			Vector4c resultColor;
			resultColor.w = 0;
	
			if (i == 0) {
				resultColor.x = (unsigned char)(colorGradient[0].x * 255.0f);
				resultColor.y = (unsigned char)(colorGradient[0].y * 255.0f);
				resultColor.z = (unsigned char)(colorGradient[0].z * 255.0f);
			}
			else if (colorGradient[i].w < elevation) {
				resultColor.x = (unsigned char)(colorGradient[gradientLength - 1].x * 255.0f);
				resultColor.y = (unsigned char)(colorGradient[gradientLength - 1].y * 255.0f);
				resultColor.z = (unsigned char)(colorGradient[gradientLength - 1].z * 255.0f);
			}
			else {
				int i1 = i - 1;
				// interpolate between color at (i-1) and (i)
				float t = (elevation - colorGradient[i1].w) / (colorGradient[i].w - colorGradient[i1].w);	
				Vector4f color = colorGradient[i1] + t * (colorGradient[i] - colorGradient[i1]);

				resultColor.x = (unsigned char)(color.x * 255.0f);
				resultColor.y = (unsigned char)(color.y * 255.0f);
				resultColor.z = (unsigned char)(color.z * 255.0f);
			}

			colors[vertexIndex] = resultColor;
		}

		kernelStopWatch->stop();
		lastKernelProcessTime = kernelStopWatch->getTime();
		
		assert(sizeof(GLfloat) * 3 == sizeof(Vector3f));
		glBindBuffer(GL_ARRAY_BUFFER, vertsVbo);
		glBufferData(GL_ARRAY_BUFFER, vertsCount * sizeof(Vector3f), vertices, GL_DYNAMIC_DRAW);
		
		assert(sizeof(GLubyte) * 4 == sizeof(Vector4c));
		glBindBuffer(GL_ARRAY_BUFFER, colorsVbo);
		glBufferData(GL_ARRAY_BUFFER, vertsCount * sizeof(Vector4c), colors, GL_DYNAMIC_DRAW);
		
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		delete[] colors;
		delete[] vertices;

		stopWatch->stop();
		lastProcessTime = stopWatch->getTime();
		lastProcessOnGpu = false;
	}


	void fillVboBuffers() {
		const std::vector<Vector3f>& vertices = sphere.getVertices();
		const std::vector<Vector3i>& triangles = sphere.getTriangles();

		assert(sizeof(GLfloat) * 3 == sizeof(Vector3f));

		glBindBuffer(GL_ARRAY_BUFFER, vertsVbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vector3f), &vertices[0], GL_DYNAMIC_DRAW);
			
		assert(sizeof(GLuint) * 3 == sizeof(Vector3i));

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.size() * sizeof(Vector3i), &triangles[0], GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}


	void reinitSphere() {
		// delete sphere buffers
		checkCudaErrors(deleteCudaSharedVbo(&vertsVbo, cudaVertsVboResource));
		checkCudaErrors(deleteCudaSharedVbo(&colorsVbo, cudaColorsVboResource));
		deleteVbo(&indexVbo);

		// create new sphere buffers
		createVbo(&vertsVbo, GL_ARRAY_BUFFER, (unsigned int)(sphere.verticesCount() * sizeof(GLfloat) * 3));
		createVbo(&colorsVbo, GL_ARRAY_BUFFER, (unsigned int)(sphere.verticesCount() * sizeof(GLubyte) * 4));
		createVbo(&indexVbo, GL_ELEMENT_ARRAY_BUFFER, (unsigned int)(sphere.trianglesCount() * sizeof(GLshort) * 3));

		// fill buffers with sphere data
		fillVboBuffers();
		
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaVertsVboResource, vertsVbo, cudaGraphicsMapFlagsNone));
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaColorsVboResource, colorsVbo, cudaGraphicsMapFlagsWriteDiscard));
		resetSphereGeometry();
	}


	void runBenchmark() {
		resetSphereGeometry();
		size_t measurementIterations = 9;
		size_t subdivisions = 12;
		
		std::ofstream resultsStream;
		resultsStream.open("benchmark.csv");
		assert(resultsStream.good());
		
		resultsStream << "Cuts count:;" << planesPerIteration << std::endl;
		resultsStream << "Measurement iterations:;" << measurementIterations << std::endl;
		resultsStream << "Vertices;CPU time;GPU time" << std::endl;
		
		std::vector<float> cpuTimes;
		std::vector<float> gpuTimes;
		
		cpuTimes.resize(measurementIterations);
		gpuTimes.resize(measurementIterations);

		float lastCpuTime = 0;
		
		for (size_t s = 0; s < subdivisions; ++s) {
			std::cout << (s + 1) << "/" << subdivisions << " (" << sphere.verticesCount() << " vertices)" << std::endl;
			
			bool cpuSlow = lastCpuTime > 10000;
			
			// skip CPU if last pass was longer than 4 seconds
			std::cout << "CPU..." << std::endl;
			for (size_t i = cpuSlow ? measurementIterations - 1 : 1; i < measurementIterations; ++i) {
				runRandomFaultsOnCpu();
				cpuTimes[i] = lastKernelProcessTime;
			}

			resetSphereGeometry();
		
			std::cout << "GPU..." << std::endl;
			for (size_t i = 0; i < measurementIterations; ++i) {
				runRandomFaultsOnGpu();
				gpuTimes[i] = lastKernelProcessTime;
			}
			
			if (!cpuSlow) {
				std::sort(cpuTimes.begin(), cpuTimes.end());
				lastCpuTime = cpuTimes[measurementIterations / 2];
			}
			std::sort(gpuTimes.begin(), gpuTimes.end());
			
			resultsStream << sphere.verticesCount() << ";"
				<< (cpuSlow ? cpuTimes[measurementIterations - 1] : cpuTimes[measurementIterations / 2]) << ";"
				<< gpuTimes[measurementIterations / 2] << std::endl;
			
			if (s + 1 < subdivisions) {
				std::cout << "Subdividing..." << std::endl;
				sphere.subdivide();
				reinitSphere();
			}
		}

		std::cout << "Done..." << std::endl;
	}


	void renderSphere() {
		glBindBuffer(GL_ARRAY_BUFFER, vertsVbo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo); 
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);

		if (!wireframe || !hiContrastMode) {
			glBindBuffer(GL_ARRAY_BUFFER, colorsVbo);
			glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);
		}

		glDrawElements(GL_TRIANGLES, (GLsizei)(sphere.trianglesCount() * 3), GL_UNSIGNED_INT, NULL);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}


	//---------------------------
	// Begins callback functions 
	//---------------------------
	void displayCallback() {
		if (hiContrastMode) {
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		}
		else{
			glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// set view matrix
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(translate.x, translate.y, translate.z);
		glRotatef(rotate.x, 1.0, 0.0, 0.0);
		glRotatef(rotate.y, 0.0, 1.0, 0.0);

		if (backFaceCulling) {
			glEnable(GL_CULL_FACE);
		}
		else{
			glDisable(GL_CULL_FACE);
		}
		glPolygonMode(GL_FRONT_AND_BACK, wireframe? GL_LINE : GL_FILL);

		glEnable(GL_DEPTH_TEST);
		glColor3f(0.0f, 0.0f, 0.0f);
		renderSphere();

		if (menuVisible) {
			// setup orthogonal projection for text
			glMatrixMode(GL_PROJECTION);
			glPushMatrix();
			glLoadIdentity();
			glOrtho(0, screenWidth, screenHeight - 5, -5, -100, 100);
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
			glDisable(GL_DEPTH_TEST);

			if (hiContrastMode) {
				glColor3f(0.2f, 0.2f, 0.2f);
			}
			else {
				glColor3f(0.8f, 0.8f, 0.8f);
			}
			drawControls();
		
			glPopMatrix();
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();
		}

		glutSwapBuffers();
		glutReportErrors();
	}


	void keyboardCallback(unsigned char key, int /*x*/, int /*y*/) {
		switch (key) {
			case 27:
				exit(EXIT_SUCCESS);
			case ' ':
				runRandomFaultsOnGpu();
				break;
			// case 'c':
			// 	runRandomFaultsOnCpu();
				// break;
			// case 's':
			// 	sphere.subdivide();
			// 	reinitSphere();
			// 	break;
			// case 't':
			// 	sphere.initAsTetrahedron();
			// 	reinitSphere();
			// 	break;
			// case 'i':
			// 	sphere.initAsIcosahedron();
			// 	reinitSphere();
			// 	break;
			case '+':
				planesPerIteration <<= 1;
				if (planesPerIteration == 0) {
					planesPerIteration = 1;
				}
				break;
			case '-':
				planesPerIteration >>= 1;
				if (planesPerIteration == 0) {
					planesPerIteration = 1;
				}
				break;
			case 'd':
				displacement *= 0.5;
				break;
			case 'f':
				displacement *= 2;
				break;
			case 'e':
				planesOnlyInCenter ^= true;
				break;
			case 'r':
				resetSphereGeometry();
				break;
			case 'm':
				menuVisible ^= true;
				break;
			case 'w':
				wireframe = !wireframe;
				break;
			case 'q':
				runBenchmark();
				break;
			case 'h':
				hiContrastMode ^= true;
				break;
			case 'b':
				backFaceCulling ^= true;
				break;
		}
		glutPostRedisplay();
	}


	void mouseCallback(int button, int state, int x, int y) {
		if (button == 3 || button == 4) { // stroll a wheel event
			// Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
			if (state == GLUT_UP) {
				return; // Disregard redundant GLUT_UP events
			}

			translate.z *= (button == 3) ? 1/1.1f : 1.1f;
			glutPostRedisplay();
			return;
		}

		if (state == GLUT_DOWN) {
			mouseButtons |= 1 << button;
		}
		else if (state == GLUT_UP) {
			mouseButtons &= ~(1 << button);
		}

		mouseOld.x = x;
		mouseOld.y = y;
		glutPostRedisplay();
	}


	void motionCallback(int x, int y) {
		float dx = (float)(x - mouseOld.x);
		float dy = (float)(y - mouseOld.y);

		if (mouseButtons == 1) {
			rotate.x += dy * 0.2f;
			rotate.y += dx * 0.2f;
		}
		else if (mouseButtons == 2) {
			translate.x += dx * 0.01f;
			translate.y -= dy * 0.01f;
		}
		else if (mouseButtons == 3) {
			translate.z += dy * 0.01f;
		}
	
		mouseOld.x = x;
		mouseOld.y = y;
		glutPostRedisplay();
	}


	void reshapeCallback(int w, int h) {
		screenWidth = w;
		screenHeight = h;

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(60.0, (float)screenWidth / (float)screenHeight, 0.1, 1000.0);
		glMatrixMode(GL_MODELVIEW);
		glViewport(0, 0, screenWidth, screenHeight);
	}
	//--------------------------------------------------------------------------------
	

	void initGradient() {
		//colorGradient.push_back(Vector4f(0.00f, 0.00f, 0.00f, 0.8f));
		colorGradient.push_back(Vector4f(0.25f, 0.35f, 0.60f, 1.0f));
		colorGradient.push_back(Vector4f(0.10f, 0.35f, 0.05f, 1.001f));
		colorGradient.push_back(Vector4f(0.20f, 0.30f, 0.10f, 1.05f));
		colorGradient.push_back(Vector4f(0.40f, 0.35f, 0.20f, 1.10f));
		colorGradient.push_back(Vector4f(1.00f, 1.00f, 1.00f, 1.20f));
	}


	bool initGL(int* argc, char** argv) {
		glutInit(argc, argv);  // Create GL context
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
		glutInitWindowSize(screenWidth, screenHeight);
		glutCreateWindow("CPU vs GPU: Random faults on sphere by Marek Fiser");
	
		glewInit();

		if (!glewIsSupported("GL_VERSION_2_0")) {
			std::cerr << "ERROR: Support for necessary OpenGL extensions missing." << std::endl;
			return false;
		}
		
		glFrontFace(GL_CW);
		glCullFace(GL_BACK);

		glEnable(GL_MULTISAMPLE);	
		glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
		glutReportErrors();
		return true;
	}


	void runGui(int argc, char** argv) {
		// error check GL initlization
		if (!initGL(&argc, argv)) {
			return;
		}
		
		initGradient();
		checkCudaErrors(cudaGLSetGLDevice(0));
	
		// register callbacks
		glutDisplayFunc(displayCallback);
		glutKeyboardFunc(keyboardCallback);
		glutMouseFunc(mouseCallback);
		glutMotionFunc(motionCallback);
		glutReshapeFunc(reshapeCallback);
	
		// init sphere
		sphere.initAsIcosahedron();
		reinitSphere();
		// want to have a set subdivided sphere
		sphere.subdivide();
		reinitSphere();
		sphere.subdivide();
		reinitSphere();
		sphere.subdivide();
		reinitSphere();
		sphere.subdivide();
		reinitSphere();
		sphere.subdivide();
		reinitSphere();

		
		sdkCreateTimer(&stopWatch);
		sdkCreateTimer(&kernelStopWatch);
		glutMainLoop();
	}
}


int main(int argc, char** argv) {

	// load in CSV data file - adapted from StackOverflow
	std::ifstream in("data/tumorSimData.csv");
    std::string line, field;
    std::vector<std::vector<std::string>> data;  // the 2D array
    std::vector<std::string> v;                   // array of values for one line only

    while (getline(in,line)) {                    // get next line in file
        v.clear();
        std::stringstream ss(line);
        while (getline(ss,field,',')) {          // break line into comma delimitted fields
            v.push_back(field);                  // add each field to the 1D array
        }
        data.push_back(v);                      // add the 1D array to the 2D array
    }

    // print out what was read in
	// for (int i=0; i < data.size(); i++)
 //    	std::cout << data[i][0] << data[i][1] << data[i][2] << '\n';

 

	mf::runGui(argc, argv);	
	return EXIT_SUCCESS;
}


