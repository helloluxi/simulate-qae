
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <curand_kernel.h>

__device__ constexpr double PI = 3.141592653589793;

constexpr int MY_BLOCK_DIM = 256;
constexpr int MY_GRID_DIM  = 256;
constexpr int TOTAL_THREAD_NUM = MY_GRID_DIM * MY_BLOCK_DIM;
#define __KERNEL_ARGS__ <<< MY_GRID_DIM, MY_BLOCK_DIM >>>

#ifdef _WIN32
#define saturate(x) max(x, 0)
#define __FLT_MAX__ FLT_MAX
#define __DBL_MAX__ DBL_MAX
#endif

#pragma region Utils

__device__ double pow2(double d) {
    return d * d;
}

__device__  double probQPE(double phase, double phaseCenter, int T) {
    double angle = PI * (phase - phaseCenter);
    return abs(angle) < 1e-15 ? 1 : pow2(sinf(T * angle) / (T * sinf(angle)));
}

__device__ int oneShot(double prob, curandState* rd) {
	return curand_uniform(rd) < prob;
}

__device__ int binarySample(double prob, int nShot, curandState* rd) {
    int sum = 0;
    for (int i = 0; i < nShot; i++)
    {
        sum += curand_uniform(rd) < prob;
    }
    return sum;
}

__device__ int Hash(int n)
{
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return n;
}

#pragma endregion

#pragma region Malloc

void* mallocArr(int size) {
    void* arr;
    checkCudaErrors(cudaMalloc(&arr, size));
    return arr;
}

void* mallocArrManaged(int size) {
    void* arr;
    checkCudaErrors(cudaMallocManaged(&arr, size));
    return arr;
}

int* copyIntArr(int size, int* src) {
    int* arr;
    checkCudaErrors(cudaMalloc((void**)&arr, sizeof(int) * size));
    checkCudaErrors(cudaMemcpy(arr, src, sizeof(int) * size, cudaMemcpyHostToDevice));
    return arr;
}

double* copyDoubleArr(int size, double* src) {
    double* arr;
    checkCudaErrors(cudaMalloc((void**)&arr, sizeof(double) * size));
    checkCudaErrors(cudaMemcpy(arr, src, sizeof(double) * size, cudaMemcpyHostToDevice));
    return arr;
}

#pragma endregion

#pragma endregion

#pragma region standard QPE

__device__ double sampleStandardQPE(double phase, int T, curandState* rd) {
    double bias = 0;
    int t = T;
    while (t > 1) {
        double prob = (1 + cosf(t * (phase - bias) * PI)) * 0.5f;
        if (curand_uniform(rd) > prob)
            bias += 1.0f / t;
        t >>= 1;
    }
    return bias;
}

__global__ void qpe_kernel(double* data, int nAll, int T, int R, bool unbiased, double* freeMem) {
    int kernelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState ref_rd, * rd = &ref_rd;
	curand_init(Hash(kernelIdx), 0, 0, rd);
    double* res = freeMem + kernelIdx * R;
    for (int dataIdx = kernelIdx; dataIdx < nAll; dataIdx += TOTAL_THREAD_NUM)
    {
        if (R == 1) {
            double bias = curand_uniform(rd) * unbiased;
            data[dataIdx] = pow2(sinf(PI*(sampleStandardQPE(asinf(sqrtf(data[dataIdx]))/PI+bias, T, rd)-bias)));
        }
        else {
            double center = asinf(sqrtf(data[dataIdx])) / PI;
            for (int i = 0; i < R; i++) {
                double bias = curand_uniform(rd) * unbiased;
                res[i] = sampleStandardQPE(center + bias, T, rd) - bias;
            }

            int precision = R * T * 4, sameBysCounter = 0;
            double maxBysPhase = -1;
            double maxLogBys = -__DBL_MAX__;
            for (int i = 0; i <= precision; i++)
            {
                double estimatedPhase = (double)i / precision;
                double logBys = 0;
                for (int j = 0; j < R; j++) {
                    if(res[j] != estimatedPhase){
                        double delta = PI * (res[j] - estimatedPhase);
                        logBys += log(abs(sin(T*delta)/(sin(delta))));
                    }
                }
                if (logBys > maxLogBys) {
                    maxLogBys = logBys;
                    maxBysPhase = estimatedPhase;
                    sameBysCounter = 1;
                }
                else if (logBys == maxLogBys) {
                    maxBysPhase = (maxBysPhase * sameBysCounter + estimatedPhase) / ++sameBysCounter;
                }
            }
            data[dataIdx] = pow2(sinf(PI*maxBysPhase));
        }
    }
}

extern "C" __declspec(dllexport) void call_qpe_kernel(double* data, int nAll, int T, int R, bool unbiased) {
    double* d_data = copyDoubleArr(nAll, data);
    double* freeMem = (double*)mallocArr(TOTAL_THREAD_NUM * R * sizeof(double));
    qpe_kernel __KERNEL_ARGS__ (d_data, nAll, T, R, unbiased, freeMem);
    checkCudaErrors(cudaMemcpy(data, d_data, sizeof(double) * nAll, cudaMemcpyDeviceToHost));
}

#pragma endregion

#pragma region mlae

__device__ double mleForMlaeLog(int length,
	int* Ms, int* Rs, int* Hs, int precision) {
	int sameBysCount = 0, est = 0;
	double maxLogP = -__DBL_MAX__;
	for (int test = 0; test <= precision; test++)
	{
		double iAngle = asin(sqrt((double)test / precision)), logP = 0;
		for (int t = 0; t < length; t++)
		{
			double sin2 = pow2(sin(iAngle * Ms[t]));
			if (Hs[t])
				logP += Hs[t] * log(sin2);
			if (Rs[t] - Hs[t])
				logP += (Rs[t] - Hs[t]) * log(1 - sin2);
		}
		if (logP > maxLogP) {
			est = test;
			maxLogP = logP;
			sameBysCount = 1;
		}
		else if (logP == maxLogP) {
			est = (est * sameBysCount + (double)test / precision);
			est /= ++sameBysCount;
		}
	}
	return (double)est / precision;
}

__global__ void mlae_kernel(double* data, int nAll,
	int length, int* Ms, int* Rs, int precision, int* freeMem)
{
	int kernelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState ref_rd, * rd = &ref_rd;
	curand_init(Hash(kernelIdx), 0, 0, rd);
	int* Hs = freeMem + kernelIdx * length;
	for (int dataIdx = kernelIdx; dataIdx < nAll; dataIdx += TOTAL_THREAD_NUM)
	{
		double angle = asin(sqrt(data[dataIdx]));
		for (int t = 0; t < length; t++)
		{
			Hs[t] = binarySample(pow2(sin(angle * Ms[t])), Rs[t], rd);
		}
		data[dataIdx] = mleForMlaeLog(length, Ms, Rs, Hs, precision);
	}
}

extern "C" __declspec(dllexport) void call_mlae_kernel(double* data, int nAll,
	int length, int* Ms, int* Rs, int precision)
{
	double* d_data = copyDoubleArr(nAll, data);
	int* d_Ms = copyIntArr(length, Ms);
	int* d_Rs = copyIntArr(length, Rs);
	int* freeMem = (int*)mallocArr(TOTAL_THREAD_NUM * length * sizeof(int));

	mlae_kernel __KERNEL_ARGS__ (d_data, nAll, length, d_Ms, d_Rs, precision, freeMem);
	checkCudaErrors(cudaMemcpy(data, d_data, sizeof(double) * nAll, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceReset());
}

#pragma endregion

// #pragma region rqae

// __device__ double maximumLikelihoodEstimationLog_Ms(int MLength, int* Ms, int* Hs, int precision) {
// 	int sameBysCount = 0;
// 	double est = 0;
// 	double maxLogP = -__DBL_MAX__;
// 	for (int test = 0; test <= precision; test++)
// 	{
// 		double iAngle = asin(sqrt((double)test / precision)), logP = 0;
// 		for (int t = 0; t < MLength; t++)
// 		{
// 			double sin2 = pow2(sin(iAngle * Ms[t]));
// 			logP += log(Hs[t] ? sin2 : 1 - sin2);
// 		}
// 		if (logP > maxLogP) {
// 			est = (double)test / precision;
// 			maxLogP = logP;
// 			sameBysCount = 1;
// 		}
// 		else if (logP == maxLogP) {
// 			est = (est * sameBysCount + (double)test / precision);
// 			est /= ++sameBysCount;
// 		}
// 	}
// 	return est;
// }

// __global__ void rqae_kernel(double* data, int nAll,
// 	int nIter, int R, int precision, int* freeMem)
// {
// 	int kernelIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     curandState ref_rd, * rd = &ref_rd;
// 	curand_init(Hash(kernelIdx), 0, 0, rd);

// 	int MLength = R * nIter;
// 	int* Ms = freeMem + MLength * (kernelIdx * 2);
// 	int* Hs = freeMem + MLength * (kernelIdx * 2 + 1);

// 	for (int dataIdx = kernelIdx; dataIdx < nAll; dataIdx += TOTAL_THREAD_NUM)
// 	{
// 		double angle = asin(sqrt(data[dataIdx]));
// 		for (int h = 0; h < MLength; ++h) {
// 			Hs[h] = 0;
// 		}
// 		for (int t = 0; t < MLength; t++)
// 		{
// 			int iter = t / R;
// 			Ms[t] = (int)((1 - curand_uniform(rd)) * (1 << iter)) | 1 << iter;
// 			Hs[t] = curand_uniform_double(rd) < pow2(sin(angle * Ms[t]));
// 		}
// 		data[dataIdx] = maximumLikelihoodEstimationLog_Ms(MLength, Ms, Hs, precision);
// 	}
// }

// extern "C" __declspec(dllexport) void call_rqae_kernel(double* data, int nAll, int T, int R)
// {
// 	double* d_data = copyDoubleArr(nAll, data);
// 	int nIter = (int)round(log2(T));
// 	int* freeMem = (int*)mallocArr(TOTAL_THREAD_NUM * nIter * R * 2 * sizeof(int));
// 	rqae_kernel __KERNEL_ARGS__ (d_data, nAll, nIter, R, R * T * 3 / 2, freeMem);
// 	checkCudaErrors(cudaMemcpy(data, d_data, sizeof(double) * nAll, cudaMemcpyDeviceToHost));
// 	checkCudaErrors(cudaDeviceReset());
// }

// #pragma endregion

// #pragma region sine-initial state QPE

// __device__ double getProbability(int T, double* amps, double theta){
// 	double re = 0, im = 0;
// 	for (int i = 0; i < T; i++)
// 	{
// 		re += amps[i] * cos(i * theta);
// 		im += amps[i] * sin(i * theta);
// 	}
// 	return (pow2(re) + pow2(im)) / T;
// }

// __global__ void oqae_kernel(double* data, int nAll, int T, double* amps) {
// 	int kernelIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     curandState ref_rd, * rd = &ref_rd;
// 	curand_init(Hash(kernelIdx), 0, 0, rd);
// 	for (int dataIdx = kernelIdx; dataIdx < nAll; dataIdx += TOTAL_THREAD_NUM)
// 	{
// 		double thetaCenter = acos(1 - data[dataIdx] * 2);
// 		double rdFlag = curand_uniform_double(rd);
// 		int centerIdx = (int)round(thetaCenter / (2 * PI) * T),
// 			dir = thetaCenter > centerIdx * 2 * PI / T ? 1 : -1;
// 		for (int i = 0; i < T; i++)
// 		{
// 			int snakeIdx = centerIdx + (i % 2 == 1 ? dir : -dir) * ((i + 1) / 2);
// 			if ((rdFlag -= getProbability(T, amps, thetaCenter - (double)snakeIdx / T * PI * 2)) < 0)
// 			{
// 				data[dataIdx] = 0.5 * (1 - cos(2.0 * PI * snakeIdx / T));
// 				break;
// 			}
// 		}
// 	}
// }

// extern "C" __declspec(dllexport) void call_oqae_kernel(double* data, int nAll, int T) {
// 	double* d_data = copyDoubleArr(nAll, data);

// 	// sin state
// 	double* d_amps = (double*)mallocArrManaged(sizeof(double) * T);
// 	for (int i = 0; i < T; i++)
// 	{
// 		d_amps[i] = sin((i + 1) * PI / (T + 1)) * sqrt(2.0 / (T + 1));
// 	}

// 	oqae_kernel __KERNEL_ARGS__ (d_data, nAll, T, d_amps);
// 	checkCudaErrors(cudaMemcpy(data, d_data, sizeof(double) * nAll, cudaMemcpyDeviceToHost));
// 	checkCudaErrors(cudaDeviceReset());
// }

// #pragma endregion

#pragma region rqae

__device__ double mleForRqaeLog(int length,
	int* Ms, int* Hs, double* etaPows, int precision, double* cosCache) {
	int sameBysCount = 0, est = 0;
	double maxLogP = -__DBL_MAX__;
	for (int testIdx = 0; testIdx <= precision; testIdx++)
	{
		double iAngle = asin(sqrt((double)testIdx / precision)), logP = 0;
		for (int t = 0; t < length; t++)
		{
			logP += log(
				Hs[t]       * ((1 - etaPows[Ms[t]] * cosCache[(precision + 1) * Ms[t] + testIdx]) / 2) +
				(1 - Hs[t]) * ((1 + etaPows[Ms[t]] * cosCache[(precision + 1) * Ms[t] + testIdx]) / 2)
			);
		}
		if (logP > maxLogP) {
			est = testIdx;
			maxLogP = logP;
			sameBysCount = 1;
		}
		else if (logP == maxLogP) {
			est = (est * sameBysCount + (double)testIdx / precision);
			est /= ++sameBysCount;
		}
	}
	return (double)est / precision;
}

__global__ void rqae_kernel(double* data, int nAll,
	int length, int* Ms, double* etaPows, int precision, int* freeMem, double* cosCache)
{
	int kernelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState ref_rd, * rd = &ref_rd;
	curand_init(Hash(kernelIdx), 0, 0, rd);
	int* Hs = freeMem + kernelIdx * length;
	for (int dataIdx = kernelIdx; dataIdx < nAll; dataIdx += TOTAL_THREAD_NUM)
	{
		double angle = asin(sqrt(data[dataIdx]));
		for (int t = 0; t < length; t++)
		{
			Hs[t] = oneShot(0.5 * (1 - etaPows[Ms[t]] * cos(2 * angle * Ms[t])), rd);
		}
		data[dataIdx] = mleForRqaeLog(length, Ms, Hs, etaPows, precision, cosCache);
	}
}

extern "C" __declspec(dllexport) void call_rqae_kernel(double* data, int nAll,
	int length, int* Ms, double eta, int precision)
{
	double* d_data = copyDoubleArr(nAll, data);
	int* freeMem = (int*)mallocArr(TOTAL_THREAD_NUM * length * sizeof(int));
	int* d_Ms = copyIntArr(length, Ms);

	int maxM = 0;
	for (int i = 0; i < length; i++)
	{
		if (Ms[i] > maxM) {
			maxM = Ms[i];
		}
	}
	double* d_etas = (double*)mallocArrManaged(sizeof(double) * (maxM + 1));
	double* d_cosCache = (double*)mallocArrManaged(sizeof(double) * (maxM + 1) * (precision + 1));
	double* angles = (double*)malloc(sizeof(double) * (precision + 1));
	for (int i = 0; i <= precision; i++)
	{
		angles[i] = asin(sqrt((double)i / precision));
	}
	for (int M = 0; M < maxM + 1; M++)
	{
		d_etas[M] = pow(eta, 2*M);
		for (int j = 0; j <= precision; j++)
		{
			d_cosCache[M * (precision + 1) + j] = cos(2 * M * angles[j]);
		}
	}

	rqae_kernel __KERNEL_ARGS__ (d_data, nAll, length, d_Ms, d_etas, precision, freeMem, d_cosCache);
	checkCudaErrors(cudaMemcpy(data, d_data, sizeof(double) * nAll, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceReset());
}

#pragma endregion
