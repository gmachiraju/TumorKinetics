#include <stdio.h>
#include <cuda_runtime.h>
#include "RandomFaultsKernel.h"
#include "CudaMathHelper.h"
#include "OpenGlCudaHelper.h"

__device__ uchar4 getColor(float elevation, const float4* gradient, unsigned int gradientLength) {
	unsigned int i = 0;
	while (i < gradientLength && gradient[i].w < elevation) {
		++i;
	}
	
	uchar4 resultColor;
	resultColor.w = 0;
	
	if (i == 0) {
		resultColor.x = (unsigned char)(gradient[0].x * 255.0f);
		resultColor.y = (unsigned char)(gradient[0].y * 255.0f);
		resultColor.z = (unsigned char)(gradient[0].z * 255.0f);
	}
	else if (gradient[i].w < elevation) {
		resultColor.x = (unsigned char)(gradient[gradientLength - 1].x * 255.0f);
		resultColor.y = (unsigned char)(gradient[gradientLength - 1].y * 255.0f);
		resultColor.z = (unsigned char)(gradient[gradientLength - 1].z * 255.0f);
	}
	else {
		int i1 = i - 1;
		// interpolate between color at (i-1) and (i)
		float t = (elevation - gradient[i1].w) / (gradient[i].w - gradient[i1].w);	
		float4 color = gradient[i1] + t * (gradient[i] - gradient[i1]);

		resultColor.x = (unsigned char)(color.x * 255.0f);
		resultColor.y = (unsigned char)(color.y * 255.0f);
		resultColor.z = (unsigned char)(color.z * 255.0f);
	}

	return resultColor;
}


__global__ void faultsKernel(float3* vertices, unsigned int vertsCount,
		const float4* planes, unsigned int planesStartIndex, unsigned int planesToProcess, float displacement,
		uchar4* colors, const float4* gradient, unsigned int gradientLength) {
			
	unsigned int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	unsigned int vertexIndex = __mul24(blockId, blockDim.x) + threadIdx.x;
	//printf("%i %i %i\n", blockId, threadIdx.x, vertexIndex);

	if (vertexIndex >= vertsCount) {
		return;
	}

	float3 v = vertices[vertexIndex];
	unsigned int planeMaxIndex = planesStartIndex + planesToProcess;
	int displacementSteps = 0;

	// compute contributions for displacement
	for (unsigned int planeIndex = planesStartIndex; planeIndex < planeMaxIndex; ++planeIndex) {
		float4 plane = planes[planeIndex];
		if (v.x * plane.x + v.y * plane.y + v.z * plane.z + plane.w > 0) {
			++displacementSteps;
		}
		else {
			--displacementSteps;
		}
	}

	// displace vector
	v += (displacement * displacementSteps) * normalize(v);
	vertices[vertexIndex] = v;

	// count color
	colors[vertexIndex] = getColor(length(v), gradient, gradientLength);
}


__global__ void resetGeometryKernel(float3* vertices, unsigned int vertsCount, uchar4* colors, uchar4 color) {
	unsigned int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	unsigned int vertexIndex = __mul24(blockId, blockDim.x) + threadIdx.x;
	
	if (vertexIndex >= vertsCount) {
		return;
	}
	
	float3 v = vertices[vertexIndex];
	vertices[vertexIndex] = normalize(v);
	colors[vertexIndex] = color;
}


void runFaultsKernel(float3* vertices, unsigned int vertsCount,
		const float4* planes, unsigned int planesStartIndex, unsigned int planesToProcess, float displacement,
		uchar4* colors, const float4* gradient, unsigned int gradientLength, StopWatchInterface* sw) {

	unsigned int threadsCount = 512;
	unsigned int requiredBlocksCount = (vertsCount + threadsCount - 1) / threadsCount;
	dim3 dimGrid;
	if (requiredBlocksCount < 65536) {
		dimGrid.x = requiredBlocksCount;
		dimGrid.y = 1;
		dimGrid.z = 1;
	}
	else {
		dimGrid.x = 65535;
		dimGrid.y = (requiredBlocksCount + 65534) / 65535;
		dimGrid.z = 1;
	}

	sw->start();
	faultsKernel<<<dimGrid, threadsCount>>>(vertices, vertsCount, planes, planesStartIndex, planesToProcess, displacement, colors, gradient, gradientLength);
	mf::checkCudaErrors(cudaDeviceSynchronize());
	sw->stop();
}


void runResetGeometryKernel(float3* vertices, unsigned int vertsCount, uchar4* colors, uchar4 color) {
	unsigned int threadsCount = 512;
	unsigned int requiredBlocksCount = (vertsCount + threadsCount - 1) / threadsCount;
	dim3 dimGrid;
	if (requiredBlocksCount < 65536) {
		dimGrid.x = requiredBlocksCount;
		dimGrid.y = 1;
		dimGrid.z = 1;
	}
	else {
		dimGrid.x = 65535;
		dimGrid.y = (requiredBlocksCount + 65534) / 65535;
		dimGrid.z = 1;
	}

	resetGeometryKernel<<<dimGrid, threadsCount>>>(vertices, vertsCount, colors, color);
	mf::checkCudaErrors(cudaDeviceSynchronize());
}

