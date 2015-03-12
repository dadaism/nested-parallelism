#ifndef __FINDMIN_KERNEL__
#define __FINDMIN_KERNEL__

#define INF 1073741824

__global__ void findMin1_kernel(int *costArray, char *commit, int *bufferBlock, int *minValue, int n)
{
	__shared__ int sdata[1024];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; //(65533*1024)
	int threshold = *minValue;

	if ( i>=n || commit[i]==1 )
		sdata[tid] = INF;
	else
		sdata[tid] = costArray[i];

	if ( sdata[tid]<threshold )
		sdata[tid] = INF;

	__syncthreads();
	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid<s){
			if ( sdata[tid]>sdata[tid+s] )
				sdata[tid] = sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid == 0)
		bufferBlock[blockIdx.x] = sdata[0];
}

__global__ void findMin2_kernel(int *bufferBlock, int *costArray, int n)
{
	__shared__ int sdata[1024];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; //(less than 65533)
	if (i<n){
		sdata[tid] = costArray[i];
	}
	else {
		sdata[tid] = INF;
	}
	__syncthreads();
	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid<s){
			if ( sdata[tid]>sdata[tid+s] )
				sdata[tid] = sdata[tid+s];
		}
		__syncthreads();
	}
	if (tid == 0)
		bufferBlock[blockIdx.x] = sdata[0];
}

__global__ void findMin3_kernel( int *bufferBlock, int n, int *minValue )
{
	__shared__ int sdata[1024];

	unsigned int tid = threadIdx.x;		// node number is less than 1024
	if (tid<n){
		sdata[tid] = bufferBlock[tid];
	}
	else {
		sdata[tid] = INF;
	}
	__syncthreads();
	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid<s){
			if ( sdata[tid]>sdata[tid+s] )
				sdata[tid] = sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid == 0)
		*minValue = sdata[tid];
}
#endif
