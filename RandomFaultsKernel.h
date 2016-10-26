#pragma once
#include "cuda_runtime.h"
#include "TimerHelper.h"


extern "C" void runFaultsKernel(float3* vertices, unsigned int vertsCount,
		const float4* planes, unsigned int planesStartIndex, unsigned int planesToProcess, float displacement,
		uchar4* colors, const float4* gradient, unsigned int gradientLength, StopWatchInterface* sw);


extern "C" void runResetGeometryKernel(float3* vertices, unsigned int vertsCount, uchar4* colors, uchar4 color);

