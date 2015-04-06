#ifndef __UBFS_KERNEL__
#define __UBFS_KERNEL__

#define MAX_LEVEL 9999
#define MAXDIMGRID 65535
#define MAX_THREAD_PER_BLOCK 1024

#define THREASHOLD 256
#define SHM_BUFF_SIZE 256
#define NESTED_BLOCK_SIZE 64

#define WARP_SIZE 32

// Uncomment if you want to profile GPU
//#define GPU_PROFILE

// Uncomment if you want reduction
#define SIMULATE_REAL

// Choose 1 from 3
//#define ARITH_INTENSE
#define MIX_ARITH_IO 1
//#define IO_INTENSE 1

#include "work_device.cu"

__device__ FLOAT_T global_array[2048];


#ifdef GPU_PROFILE
__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
        printf("====> GPU #%u - number of nested kernel calls:%u\n",solution, nested_calls);
}
#endif 

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

__device__ inline double __shfl_down (double var, unsigned int src_lane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, src_lane, width);
    a.y = __shfl_down(a.y, src_lane, width);
    return *reinterpret_cast<double*>(&a);
}

__inline__ __device__ FLOAT_T warp_reduce_sum(FLOAT_T val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__inline__ __device__ FLOAT_T block_reduce_sum(FLOAT_T val) {
    static __shared__ FLOAT_T shared[32]; // Shared mem for 32 partial sums
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

__global__ void np_thread_kernel(int *iter, FLOAT_T *data, FLOAT_T *rst, int data_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	FLOAT_T value = 0.0;
	if ( tid<data_num ) {
		for (int i=0; i<iter[tid]; ++i) {
			value += work(data, global_array, tid);
		}
		rst[tid] = value;
	}
}

__global__ void np_thread_queue_kernel(int *iter, FLOAT_T *data, FLOAT_T *rst, 
										int *queue, unsigned int *queue_length, int data_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontier_no = *queue_length;
	FLOAT_T value = 0.0;
	if ( tid<frontier_no ){
		int idx = queue[tid];	//	grab a work from queue, tid is queue index
		for (int i=0; i<iter[idx]; ++i) {
			value += work(data, global_array, idx);
		}
		rst[idx] = value;
	}
}

__global__ void np_block_queue_kernel(int *iter, FLOAT_T *data, FLOAT_T *rst, 
									int *queue, unsigned int *queue_length)
{
	int bid = blockIdx.x+blockIdx.y*gridDim.x;	//*MAX_THREAD_PER_BLOCK + threadIdx.x;
	int idx = 0;
	int frontier_no = *queue_length;
	FLOAT_T value = 0.0;
	if ( bid<frontier_no ) {
		idx = queue[bid];	//	grab a work from queue, bid is queue index
		for (int i=threadIdx.x; i<iter[idx]; i+=blockDim.x) {
			value += work(data, global_array, idx);
		}
#ifdef SIMULATE_REAL
		__syncthreads();
		value = block_reduce_sum( value );
		if ( threadIdx.x==0 )
			rst[idx] = value;
#else
		rst[idx] += value;
#endif
	}
}

/* LOAD BALANCING THROUGH DELAYED BUFFER */

/* implements a delayed buffer in shared memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - in phase 2, the blocks access the nodes in the delayed-buffer in a block-based mapping (one neighbor per thread)
*/
__global__ void np_shared_delayed_buffer_kernel( int *iter, FLOAT_T *data, FLOAT_T *rst, int data_num )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int t_idx = 0;	//	thread-based variable used to index inside the delayed buffer
	__shared__ int buffer[SHM_BUFF_SIZE]; //delayed buffer
	__shared__ unsigned int idx; //index within the delayed buffer

	if (threadIdx.x==0) idx = 0;
	__syncthreads();
	
	// 1st phase - thread-based mapping
	if ( tid<data_num ) {
		FLOAT_T value= 0.0;
		int iter_num = iter[tid];
		if ( iter_num < THREASHOLD ) {
			for (int i=0; i<iter_num; ++i) {
				value += work(data, global_array, tid);
			}
			rst[tid] = value;
		}
		else {	//	insert into delayed buffer
			t_idx = atomicInc(&idx, SHM_BUFF_SIZE);
			buffer[t_idx] = tid;
		}
	}
	__syncthreads();
	// 2nd phase - each block processed all the elements in its shared memory buffer; each thread process a different neighbor
#ifdef GPU_PROFILE
	if (tid==0 && idx!=0) {
		printf("In Block %d # delayed nodes : %d\n", blockIdx.x, idx);
	}
#endif
	for (int i=0; i<idx; i++) {
		FLOAT_T value = 0.0;
		int curr = buffer[i]; //grab an element from the buffer
		int iter_num = iter[curr];
		// access neighbors - one thread per neigbor;
		for (int i=threadIdx.x; i<iter_num; i+=blockDim.x){
			value += work(data, global_array, curr);
      	}
#ifdef SIMULATE_REAL
		rst[curr] = value;
		__syncthreads();
		value = block_reduce_sum( value );
		if ( threadIdx.x==0 ) {
			rst[curr] = value;
		}
#else
		rst[curr] += value;
#endif
    }
}

/* implements phase 1 of delayed buffer (buffer) in global memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - phase 2 must be implemented by separately invoking the "process_buffer" kernel
*/
__global__ void np_global_delayed_buffer_kernel(int *iter, FLOAT_T *data, FLOAT_T *rst, 
												int *buffer, unsigned int *idx, int data_num )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int t_idx = 0;

	// 1st phase
	if ( tid<data_num) {
		FLOAT_T value = 0.0;
		int iter_num = iter[tid];
		if ( iter_num<THREASHOLD ) {
			/* access neighbours */
			for (int i=0; i<iter_num; ++i) {
				value += work(data, global_array, tid);
			}
			rst[tid] = value;
		}
		else {
			t_idx = atomicInc(idx, GM_BUFF_SIZE);
			buffer[t_idx] = tid;
		}
	}
}

/* LOAD BALANCING THROUGH DYNAMIC PARALLELISM */
/* Child kernel invoked by the dynamic parallelism implementation with multiple kernel calls
   This kernel processes the neighbors of a certain node. The starting and ending point (start and end parameters) within the edge array are given as parameter
*/
__global__ void process_neighbors( int idx, FLOAT_T *data, FLOAT_T *rst, int iter_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	FLOAT_T value = 0.0;
	if (tid < iter_num) {
		value += work(data, global_array, idx);
	}
#ifdef SIMULATE_REAL
	__syncthreads();
	value = block_reduce_sum( value );
	if ( threadIdx.x==0 ) {
		atomicAdd( &rst[idx], value );
	}
	
#else
	rst[idx] += value;
#endif
}

__global__ void np_multidp_kernel(int *iter, FLOAT_T *data, FLOAT_T *rst, int data_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid<data_num ) {
		FLOAT_T value = 0.0;
		rst[tid] = 0;
		int iter_num = iter[tid];
		if ( iter_num<THREASHOLD ) {		
			for (int i=0; i<iter_num; ++i) {
				value += work(data, global_array, tid);
			}
			rst[tid] = value;
		}
		else {
#ifdef GPU_PROFILE
			nested_calls++;
			//  printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
     		process_neighbors<<<iter_num/NESTED_BLOCK_SIZE+1, NESTED_BLOCK_SIZE>>>(tid, data, rst, iter_num);
		}
	}
}

/* processes the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void process_buffer( int *iter, FLOAT_T *data, FLOAT_T *rst, int data_num, int *buffer, unsigned int buffer_size)
{
	int bid = blockIdx.x;
	FLOAT_T value = 0.0;
	if ( bid<buffer_size ) {   // block-based mapping
		int curr = buffer[bid]; //nodes processed by current block
		int iter_num = iter[curr];
		for (int i=threadIdx.x; i<iter_num; i+=blockDim.x) { // eid is the identifier of the edge processed by the current thread
			value += work(data, global_array, curr);
		}
#ifdef SIMULATE_REAL
		__syncthreads();
		value = block_reduce_sum( value );
		if ( threadIdx.x==0 ) {
			rst[curr] = value;
		}
#else
		rst[curr] += value;
#endif
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void np_singledp_kernel(int *iter, FLOAT_T *data, FLOAT_T *rst, int *buffer, int data_num)
{
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
    unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    __shared__ unsigned int block_index;                            // index of each block within its sub-buffer
    int t_idx = 0;                                          // used to access the buffer
    if (threadIdx.x == 0) block_index = 0;
    __syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<data_num ){
		FLOAT_T value = 0.0;
		int iter_num = iter[tid];
		if ( iter_num<THREASHOLD ) {
			for (int i=0; i<iter_num; ++i) {
				value += work(data, global_array, tid);
			}
			rst[tid] = value;
		}
		else {
			t_idx = atomicInc(&block_index, per_block_buffer);
			buffer[t_idx+block_offset] = tid;
		}
	}
	__syncthreads();
	
	//2nd phase - nested kernel call
	if (threadIdx.x==0 && block_index!=0){
#ifdef GPU_PROFILE
		nested_calls++;
#endif
		process_buffer<<<block_index,NESTED_BLOCK_SIZE>>>( iter, data, rst, data_num, buffer+block_offset, block_index);
	}
}

__global__ void gen_dual_queue_workset_kernel(int *iter, int data_num,
								int *queue_l, unsigned int *queue_length_l, unsigned int queue_max_length_l,
								int *queue_h, unsigned int *queue_length_h, unsigned int queue_max_length_h)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<data_num) {
		int iter_size = iter[tid];
		if ( iter_size < THREASHOLD ) {
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
