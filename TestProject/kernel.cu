
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>
#include <stdio.h>
#include "timer.h"

const int GpuThreadCount = 256;
const int MAXBLOCKS = 65535;

	//const int testSize = 1073741824;
	//const int testSize = 268435456;
	//const int testSize = 134217728;
	//const int testSize = 67108864;
	const int testSize = 33554432;
    //const int testSize = 16777216;
	//const int testSize = 4200000;
	//const int testSize = 4194304;
	//const int testSize = 65536*2;  

int calcBlockSize(int dataSize, int threadCount)
{
	int retval = (dataSize/threadCount);
	if ((dataSize % threadCount) != 0)
	{
		retval++;
	}
	if (retval > MAXBLOCKS)
		retval = MAXBLOCKS;
	return retval;
}

struct collatzResult
{
	collatzResult()
	{
		sequenceStart = 0;
		numberOfSteps = 0;
	}

	static collatzResult Reduce(collatzResult *subResults, int size)
	{
		collatzResult result;
		int greatestNumberOfSteps = 0;
		for(int i = 0; i < size; i++)
		{
			if (subResults[i].numberOfSteps > greatestNumberOfSteps)
			{
				result = subResults[i];
				greatestNumberOfSteps = result.numberOfSteps;
			}
		}
		return result;
	}
public:
	int sequenceStart;
	int numberOfSteps;
};


struct gpu_collatzResult
{
	int sequenceStart;
	int numberOfSteps;
};

int cpu_calcCollatzNumber(int sequenceStart)
{
	int count = 0;
	long long current = sequenceStart;
	while(current != 1)
	{
		if (current & 1)
		{
			current = current * 3 + 1;
		}
		else
		{
			current = current / 2;
		}
		count++;
		if (current < sequenceStart)
			break;
	}
	return count;
}

collatzResult cpu_calcCollatzNumbers(int size)
{
	collatzResult result;
	int sequenceStart = 1;
	for(int i = 1; i < size; i++)
	{
		int steps = cpu_calcCollatzNumber(sequenceStart);
		if (steps < 0)
		{
			throw 0;
		}
		if (steps > result.numberOfSteps)
		{
			result.numberOfSteps = steps;
			result.sequenceStart = sequenceStart;
		}
		sequenceStart += 2;
	}
	return result;
}


bool cpu_verifyResult(collatzResult result)
{
	int steps = cpu_calcCollatzNumber(result.sequenceStart);
	return (steps == result.numberOfSteps);
}

__device__ void reduce(gpu_collatzResult *blockResults)
{
	int i = blockDim.x;
	if ((i % 2) != 0)
	{
		i--;
		i |= (i >> 1);
		i |= (i >> 2);
		i |= (i >> 4);
		i |= (i >> 8);
		i |= (i >> 16);
		i++;
	}
	while(i >= 1)
	{
		if ((threadIdx.x < i) &&
			(threadIdx.x + i) < blockDim.x)
		{
			if (blockResults[threadIdx.x].numberOfSteps < blockResults[threadIdx.x + i].numberOfSteps)
			{
				blockResults[threadIdx.x].numberOfSteps = blockResults[threadIdx.x + i].numberOfSteps;
				blockResults[threadIdx.x].sequenceStart = blockResults[threadIdx.x + i].sequenceStart;
			}
		}
		__syncthreads();
		i = i >> 1;
	}
}

__global__ void collatzKernel(collatzResult *results)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ gpu_collatzResult blockResults[GpuThreadCount];
	blockResults[threadIdx.x].numberOfSteps = 0;
	blockResults[threadIdx.x].sequenceStart = 0;

	while (i < testSize)
	{
		int sequenceStart = i*2 + 1;
		int count = 0;
		long long current = sequenceStart;
		while(current != 1)
		{
			if (current & 1) 
			{
				current = current * 3 + 1;
			}
			else
			{
				current = current / 2;
			}
			count++;
			if (current < sequenceStart)
				break;
		}

		if (count > blockResults[threadIdx.x].numberOfSteps)
		{
			blockResults[threadIdx.x].numberOfSteps = count;
			blockResults[threadIdx.x].sequenceStart = sequenceStart;
		}

		i += blockDim.x * gridDim.x;
	}

	__syncthreads();

#if 1
	reduce(blockResults);
#else
	i = blockDim.x;
	if ((i % 2) != 0)
	{
		i--;
		i |= (i >> 1);
		i |= (i >> 2);
		i |= (i >> 4);
		i |= (i >> 8);
		i |= (i >> 16);
		i++;
	}
	while(i >= 1)
	{
		if ((threadIdx.x < i) &&
			(threadIdx.x + i) < blockDim.x)
		{
			if (blockResults[threadIdx.x].numberOfSteps < blockResults[threadIdx.x + i].numberOfSteps)
			{
				blockResults[threadIdx.x].numberOfSteps = blockResults[threadIdx.x + i].numberOfSteps;
				blockResults[threadIdx.x].sequenceStart = blockResults[threadIdx.x + i].sequenceStart;
			}
		}
		__syncthreads();
		i = i >> 1;
	}
#endif
	if (threadIdx.x == 0)
	{
		results[blockIdx.x].numberOfSteps = blockResults[0].numberOfSteps;
		results[blockIdx.x].sequenceStart = blockResults[0].sequenceStart;
	}
}

collatzResult gpu_calcCollatzNumbers(int size, int threadCount)
{
	collatzResult result;

    cudaError_t cudaStatus;

	int blocks = calcBlockSize(size, threadCount);
	fprintf(stdout, "Threads %d Blocks %d\n", threadCount, blocks);

	collatzResult *results = new collatzResult[blocks];

	collatzResult *dev_results = 0;

	    // Allocate GPU buffer
	int allocsize = blocks * sizeof(collatzResult);
    cudaStatus = cudaMalloc((void**)&dev_results, allocsize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_results, results, allocsize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


	collatzKernel<<<blocks, threadCount>>>(dev_results);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	// Copy resultsfrom GPU buffer to host memory.
    cudaStatus = cudaMemcpy(results, dev_results, allocsize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	result = collatzResult::Reduce(results, blocks);

Error:
	delete[] results;
    cudaFree(dev_results);

    
    return result;
}

  


collatzResult cpu_Test1(int size)
{
	Timer t;

	t.Start();
	collatzResult result = cpu_calcCollatzNumbers(size);
	t.Stop();
	fprintf(stdout, "CPUTest1 Result:  Start %d Steps %d\n", result.sequenceStart, result.numberOfSteps);
	fprintf(stdout, "CPUTest1: Total %fms elements per ms: %f\n", t.Elapsed(), testSize / t.Elapsed());

	return result;
}


collatzResult gpu_Test(int size, int threadCount)
{
	float cudaTime;
	cudaEvent_t startGpu, stopGpu;
	collatzResult result;
	const int iterations = 1;
	cudaError_t cudaStatus;
#if 0	
	collatzResult cpuResult = cpu_Test1(testSize);
	fprintf(stdout, "CPUResult Start %d Steps %d\n", cpuResult.sequenceStart, cpuResult.numberOfSteps);
#endif
	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	float maxRate = 0.0;
	for(int threads = 15; threads <= threadCount; threads++)
	{
		cudaEventCreate(&stopGpu);
		cudaEventCreate(&startGpu);
		cudaEventRecord(startGpu, 0);
		for(int i = 0; i < iterations; i++)
		{
			result = gpu_calcCollatzNumbers(testSize, threads);
		}
		cudaEventRecord(stopGpu, 0);
		cudaEventSynchronize(stopGpu);

		cudaEventElapsedTime(&cudaTime, startGpu, stopGpu);
//		fprintf(stdout, "GPUTest1 Result: Start %d Steps %d\n",  result.sequenceStart, result.numberOfSteps);
		float rate = size * iterations / cudaTime;
		if (rate > maxRate)
		{
			fprintf(stdout, "GPUTest1: Total %fms elements per ms: %f\n",cudaTime, rate);
			maxRate = rate;
		}
		if (!cpu_verifyResult(result))
		{
			fprintf(stderr, "Result Invalid!!\n");
		}
	}

Error:
	return result;
}

int main()
{

	fprintf(stdout, "CollatzTest Size=%ld\n", testSize);
//	collatzResult result = cpu_Test1(testSize);

	collatzResult gpuResult = gpu_Test(testSize, GpuThreadCount);



    return 0;
}