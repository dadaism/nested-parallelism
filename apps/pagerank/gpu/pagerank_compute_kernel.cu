#ifndef __PG_KERNEL__
#define __PG_KERNEL__

#include <stdio.h>
#include <cuda.h>

#define MAX_LEVEL 9999
#define MAXDIMGRID 65535
#define MAXDIMBLOCK 1024

#define THREASHOLD 256
#define SHM_BUFF_SIZE 256

#define NESTED_BLOCK_SIZE 64
#define WARP_SIZE 32
#define MAX_STREAM_NUM 16

typedef float FLOAT_T;

//#define GPU_PROFILE

__device__ unsigned int gm_idx_pool[MAXDIMGRID*MAXDIMBLOCK/WARP_SIZE];

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
	if (lane==0) shared[wid]=val;	// Write reduced value to shared memory
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


__global__ void pg_process_neighbors(	int *r_edge_array, int *outdegree_array, FLOAT_T *rank_array,
										FLOAT_T *new_rank_array, FLOAT_T rank_random_walk, FLOAT_T rank_dangling_node,
										FLOAT_T damping, int start, int end, int nid)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x + start;
	FLOAT_T new_rank_tid = 0.0;
	if (tid < end) {
		int parent = r_edge_array[tid];
		FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
		new_rank_tid += damping * rank;
	}
	__syncthreads();
	new_rank_tid = block_reduce_sum( new_rank_tid );
	if ( threadIdx.x==0 ) {
		atomicAdd(&new_rank_array[nid], new_rank_tid);
	}
	if ( blockIdx.x==0 && threadIdx.x==0) {
		atomicAdd(&new_rank_array[nid], rank_random_walk+rank_dangling_node);
	}
}

/* processes the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void pg_process_buffer(	int *child_vertex_array, int *r_edge_array, int *outdegree_array, 
									FLOAT_T *rank_array, FLOAT_T *new_rank_array, FLOAT_T rank_random_walk,
									FLOAT_T rank_dangling_node, FLOAT_T damping, int node_num, int *buffer,
									unsigned int buffer_size)
{
	int bid = blockIdx.x;
	FLOAT_T new_rank_tid = 0.0;
	if ( bid<buffer_size ) {   // block-based mapping
		int curr = buffer[bid]; //nodes processed by current block
		/* get neighbour range */
		int start = child_vertex_array[curr];
		int end = child_vertex_array[curr+1];
		/* access neighbours */
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x) { // eid is the identifier of the edge processed by the current thread
			int parent = r_edge_array[eid]; // neighbour id
			FLOAT_T rank = rank_array[parent] / outdegree_array[parent];
			new_rank_tid += damping * rank;
		}
		__syncthreads();
		new_rank_tid = block_reduce_sum( new_rank_tid );
		if ( threadIdx.x==0 ) {
			new_rank_tid = new_rank_tid + rank_random_walk + rank_dangling_node;
        	new_rank_array[curr] = new_rank_tid;
		}
	}
}
#endif
