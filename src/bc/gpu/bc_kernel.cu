#ifndef __UBFS_KERNEL__
#define __UBFS_KERNEL__

#define MAX_LEVEL 9999
#define MAXDIMGRID 65535
#define MAXDIMBLOCK 1024

#define THREADS_PER_BLOCK 192
#define THREASHOLD 16
#define SHM_BUFF_SIZE 256
#define NESTED_BLOCK_SIZE 64
#define MAX_STREAM_NUM 16
#define WARP_SIZE 32

//#define CPU_PROFILE
//#define GPU_PROFILE

#ifdef GPU_PROFILE

__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
        printf("====> GPU #%u - number of nested kernel calls:%u\n",solution, nested_calls);
}
#endif

__device__ inline double __shfl_down (double var, unsigned int src_lane, int width=32)
{
	int2 a = *reinterpret_cast<int2*>(&var);
	a.x = __shfl_down(a.x, src_lane, width);
	a.y = __shfl_down(a.y, src_lane, width);
	return *reinterpret_cast<double*>(&a);
}

__inline__ __device__ double warp_reduce_sum(double val) {
	for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)  
		val += __shfl_down(val, offset);
	return val;
}

__inline__ __device__ double block_reduce_sum(double val) {
	static __shared__ double shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % WARP_SIZE;
	int wid = threadIdx.x / WARP_SIZE;
	val = warp_reduce_sum(val);     // Each warp performs partial reduction
	if (lane==0) shared[wid]=val;   // Write reduced value to shared memory
	__syncthreads();              // Wait for all partial reductions
	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
	if (wid==0) val = warp_reduce_sum(val); //Final reduce within first warp
	return val;
}

__device__ double atomicAdd(double* address, double val) 
{ 
	unsigned long long int* address_as_ull = (unsigned long long int*)address; 
	unsigned long long int old = *address_as_ull, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
	} while (assumed != old); 
	return __longlong_as_double(old); 
}

#include "bc_fp_kernel.cu"
#include "bc_bp_kernel.cu"

__global__ void countWorkingset_kernel( char *update, unsigned int *qCounter, 
										unsigned int qMaxLength, int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber && update[tid] )
		atomicInc(qCounter, qMaxLength);
}

__global__ void checkWorkingset_kernel( char *update, unsigned int *nonstop,  int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber && update[tid] )
		*nonstop = 1;
}

__global__ void gen_bitmap_workset_kernel( char *frontier, char *update, unsigned *nonstop, int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber && update[tid] ) {
		frontier[tid] = 1; update[tid] = 0;
		*nonstop = 1;
	}
}

__global__ void gen_queue_workset_kernel(	char *update, int *queue, unsigned int *queue_length, 
											unsigned int queue_max_length, int nodeNumber)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		/* write node number to queue */
		unsigned int q_idx = atomicInc(queue_length, queue_max_length);
		queue[q_idx] = tid;
	}
}

 __global__ void gen_dual_queue_workset_kernel(int *vertexArray, char *update, int nodeNumber,
								int *queue_l, unsigned int *queue_length_l, unsigned int queue_max_length_l,
								int *queue_h, unsigned int *queue_length_h, unsigned int queue_max_length_h)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
		/* write vertex number to LOW degree queue */
			unsigned int q_idx = atomicInc(queue_length_l, queue_max_length_l);
			queue_l[q_idx] = tid;
		}
		else {
		/* write vertex number to HIGH degree queue */
			unsigned int q_idx = atomicInc(queue_length_h, queue_max_length_h);
			queue_h[q_idx] = tid;
		}
	}
}

#endif
