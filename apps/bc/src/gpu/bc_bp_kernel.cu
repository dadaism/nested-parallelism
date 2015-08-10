#ifndef __BC_BP_KERNEL__
#define __BC_BP_KERNEL__

__global__ void backward_init_kernel(float *delta, int *sigma, int nodeNumber)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if ( tid<nodeNumber) {
                delta[tid] = 0.0;
        }
}

__global__ void backward_propagation_kernel(int *vertexArray, int *edgeArray, int *level, char *p,
											int *sigma, float *delta, int nodeNumber, int dist)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid<nodeNumber ) {
    	/* get neighbour range */
		int start = vertexArray[tid];
        int end = vertexArray[tid+1];
        /* access neighbours */
        for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			if ( level[nid]==dist-1 && p[(long)tid*nodeNumber+nid]==1 ) { // p[tid][nid]
				delta[tid] = delta[tid] + (double)sigma[tid]/sigma[nid]*(1+delta[nid]);
			}
		}
	}
}

__global__ void backward_sum_kernel(int *level, float *bc, float *delta, int nodeNumber, int dist, int source)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && level[tid]==dist-1 && tid!=source ) {
		bc[tid] = bc[tid] + delta[tid];
	}
}

__global__ void backward_prop_thread_queue_kernel(int *vertexArray, int *edgeArray, int *level, char *p,
											int *sigma, float *delta, int nodeNumber, int dist,
											int *queue, unsigned int *queue_size )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontier_no = *queue_size;	
	if ( tid<frontier_no ) {
		int curr = queue[tid];
    	/* get neighbour range */
		int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        /* access neighbours */
        for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			if ( level[nid]==dist-1 && p[(long)curr*nodeNumber+nid]==1 ) { // p[tid][nid]
				delta[curr] = delta[curr] + (double)sigma[curr]/sigma[nid]*(1+delta[nid]);
			}
		}
	}
}

__global__ void backward_prop_block_queue_kernel(int *vertexArray, int *edgeArray, int *level, char *p,
											int *sigma, float *delta, int nodeNumber, int dist,
											int *queue, unsigned int *queue_size )
{
	int bid = gridDim.x*blockIdx.y + blockIdx.x;
	int frontier_no = *queue_size;	
	if ( bid<frontier_no ) {
		double acc_delta = 0.0;
		int curr = queue[bid];
    	/* get neighbour range */
		int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        /* access neighbours */
        for (int i=start+threadIdx.x; i<end; i+=blockDim.x) {
			int nid = edgeArray[i];
			if ( level[nid]==dist-1 && p[(long)curr*nodeNumber+nid]==1 ) { // p[tid][nid]
				acc_delta = acc_delta + (double)sigma[curr]/sigma[nid]*(1+delta[nid]);
			}
		}
		__syncthreads();
		acc_delta = block_reduce_sum( acc_delta );
		if ( threadIdx.x==0 ) {
			delta[curr] += acc_delta;
		}
	}
}

/* LOAD BALANCING THROUGH DELAYED BUFFER */
 
/* implements a delayed buffer in shared memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - in phase 2, the blocks access the nodes in the delayed-buffer in a block-based mapping (one neighbor per thread)
*/
__global__ void backward_shared_delayed_buffer_kernel(int *vertexArray, int *edgeArray, 
												int *level, char *p, int *sigma, 
												float *delta, int nodeNumber, int dist)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int t_idx = 0;  //  thread-based variable used to index inside the delayed buffer
	__shared__ int buffer[SHM_BUFF_SIZE]; //delayed buffer
	__shared__ unsigned int idx; //index within the delayed buffera

	if (threadIdx.x==0) idx = 0;
	__syncthreads();
	// 1st phase - thread-based mapping
	if ( tid<nodeNumber ) {
    	/* get neighbour range */
		int start = vertexArray[tid];
        int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
        	/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid]==dist-1 && p[(long)tid*nodeNumber+nid]==1 ) { // p[tid][nid]
					delta[tid] = delta[tid] + (double)sigma[tid]/sigma[nid]*(1+delta[nid]);
				}
			}
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
		printf("BP - In Block %d # delayed nodes : %d\n", blockIdx.x, idx);
	}	
#endif
	for (int i=0; i<idx; i++) {
		int curr = buffer[i]; //grab an element from the buffer
		double acc_delta = 0.0;
		// get neighbour range    
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		// access neighbors - one thread per neigbor;
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x){
			int nid = edgeArray[eid];
			if ( level[nid]==dist-1 && p[(long)curr*nodeNumber+nid]==1 ) { // p[tid][nid]
				acc_delta = acc_delta + (double)sigma[curr]/sigma[nid]*(1+delta[nid]);
				//delta[curr] = delta[curr] + (double)sigma[curr]/sigma[nid]*(1+delta[nid]);
				//atomicAdd(&delta[curr] ,(double)sigma[curr]/sigma[nid]*(1+delta[nid]) );
			}
		}
		__syncthreads();
		acc_delta = block_reduce_sum( acc_delta );
		if ( threadIdx.x==0 ) {
			delta[curr] += acc_delta;
		}
	}
}

/* implements phase 1 of delayed buffer (buffer) in global memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - phase 2 must be implemented by separately invoking the "process_buffer" kernel
*/
__global__ void backward_global_delayed_buffer_kernel( int *vertexArray, int *edgeArray, int *level, char *p,
                                    		int *sigma, float *delta, int *buffer, unsigned int *idx, 
											int nodeNumber, int dist )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx = 0;

    // 1st phase
    if ( tid<nodeNumber) {
        /* get neighbour range */
        int start = vertexArray[tid];
        int end = vertexArray[tid+1];
        int edge_num = end - start;
        if ( edge_num<THREASHOLD ) { 
            /* access neighbours */
            for (int i=start; i<end; ++i) {
                int nid = edgeArray[i];
				if ( level[nid]==dist-1 && p[(long)tid*nodeNumber+nid]==1 ) { // p[tid][nid]
					delta[tid] = delta[tid] + (double)sigma[tid]/sigma[nid]*(1+delta[nid]);
				}
			}
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
__global__ void bp_process_neighbors( int *edgeArray, int *level, char *p, int *sigma, float *delta, 
										int dist, int nodeNumber, int curr, int start, int end)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x + start;
    double acc_delta = 0.0;
	if (tid < end) {
        int nid = edgeArray[tid];
		if ( level[nid]==dist-1 && p[(long)curr*nodeNumber+nid]==1 ) { // p[tid][nid]
			acc_delta = acc_delta + (double)sigma[curr]/sigma[nid]*(1+delta[nid]);
		}
	}
	__syncthreads();
	acc_delta = block_reduce_sum( acc_delta );
	if ( threadIdx.x==0 ) {
		atomicAdd( &delta[curr], acc_delta);
	}       
}

__global__ void backward_multidp_kernel(int *vertexArray, int *edgeArray, int *level, char *p, 
										int *sigma, float *delta, int *queue, unsigned int *queue_length,
										int nodeNumber, int dist)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid<nodeNumber ) {
		/* get neighbour range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid]==dist-1 && p[(long)tid*nodeNumber+nid]==1 ) { // p[tid][nid]
					delta[tid] = delta[tid] + (double)sigma[tid]/sigma[nid]*(1+delta[nid]);
				}
			}
		}
		else {
#ifdef GPU_PROFILE
			atomicInc(nested_calls, INF);
			//  printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
			bp_process_neighbors<<<edge_num/NESTED_BLOCK_SIZE+1, NESTED_BLOCK_SIZE>>>(edgeArray, level, p, sigma, delta, dist, nodeNumber, tid, start, end);
		}
	}
}

/* process the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void bp_process_buffer( int *vertexArray, int *edgeArray, int *level, char *p, int *sigma, 
								float *delta, int dist, int nodeNumber, int *buffer, unsigned int buffer_size )
{
	int bid = blockIdx.x;
	if ( bid<buffer_size ) {   // block-based mapping
		int curr = buffer[bid]; //nodes processed by current block
        double acc_delta = 0.0;
		/* get neighbour range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        /* access neighbours */
        for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x) { // eid is the identifier of the edge processed by the current thread
        	int nid = edgeArray[eid]; // neighbour id
            if ( level[nid]==dist-1 && p[(long)curr*nodeNumber+nid]==1 ) { // p[curr][nid]
            	acc_delta = acc_delta + (double)sigma[curr]/sigma[nid]*(1+delta[nid]);
            }
        }
		__syncthreads();
		acc_delta = block_reduce_sum( acc_delta );
		if ( threadIdx.x==0 ) {
			delta[curr] += acc_delta;
		}   
    }   
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void backward_singledp_kernel(int *vertexArray, int *edgeArray, int *level, char *p, 
											int *sigma, float *delta, int *buffer,
											int nodeNumber, int dist)
{
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
	unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
	__shared__ unsigned int block_index;                            // index of each block within its sub-buffer
	int t_idx = 0;                                          // used to access the buffer
	if (threadIdx.x == 0) block_index = 0;
	__syncthreads();
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<nodeNumber ) {
		/* get neighbour range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid]==dist-1 && p[(long)tid*nodeNumber+nid]==1 ) { // p[tid][nid]
					delta[tid] = delta[tid] + (double)sigma[tid]/sigma[nid]*(1+delta[nid]);
                }   
			}
		}
		else {
			t_idx = atomicInc(&block_index, per_block_buffer);
			buffer[t_idx+block_offset] = tid;
		}
	}
	__syncthreads();

	// 2nd phase - nested kernel call
	if (threadIdx.x==0 && block_index!=0) {
#ifdef GPU_PROFILE
		atomicInc(nested_calls, INF);
		//  printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
		bp_process_buffer<<<block_index, NESTED_BLOCK_SIZE>>>(vertexArray, edgeArray, level, p, sigma, delta, dist, nodeNumber, buffer+block_offset, block_index);
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void backward_warp_dp_kernel(int *vertexArray, int *edgeArray, int *level, char *p,
											int *sigma, float *delta, int *buffer,
											int nodeNumber, int dist)
{
	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
	}
	int warpId = threadIdx.x / WARP_SIZE;
	int warpDim = blockDim.x / WARP_SIZE;
	int total_warp_num = gridDim.x * warpDim;
	unsigned per_warp_buffer = GM_BUFF_SIZE/total_warp_num; 	// amount of the buffer available to each thread block
	unsigned warp_offset = (blockIdx.x * warpDim + warpId) * per_warp_buffer;  // block offset within the buffer

	__shared__ unsigned int warp_index[MAXDIMBLOCK/WARP_SIZE];				// index of each block within its sub-buffer
	int t_idx = 0;						// used to access the buffer
	warp_index[warpId] = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<nodeNumber ) {
		/* get neighbour range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid]==dist-1 && p[(long)tid*nodeNumber+nid]==1 ) { // p[tid][nid]
					delta[tid] = delta[tid] + (double)sigma[tid]/sigma[nid]*(1+delta[nid]);
                }
			}
		}
		else {
			t_idx = atomicInc(&warp_index[warpId], per_warp_buffer);
			buffer[t_idx+warp_offset] = tid;
		}
	}
	__syncthreads();
	//2nd phase - nested kernel call
	if (threadIdx.x%WARP_SIZE==0 && warp_index[warpId]!=0){
#ifdef GPU_PROFILE
		atomicInc(nested_calls, INF);
#endif
      	bp_process_buffer<<<warp_index[warpId],NESTED_BLOCK_SIZE,0, s[threadIdx.x%MAX_STREAM_NUM]>>>(
      														vertexArray, edgeArray, level,
      														p, sigma, delta, dist, nodeNumber,
      														buffer+warp_offset, warp_index[warpId]);
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void backward_block_dp_kernel(int *vertexArray, int *edgeArray, int *level, char *p,
											int *sigma, float *delta, int *buffer,
											int nodeNumber, int dist)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
	__shared__ int shm_buffer[MAXDIMBLOCK];
	unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
	__shared__ unsigned int block_index;                            // index of each block within its sub-buffer
	int t_idx = 0;                                          // used to access the buffer
	if (threadIdx.x == 0) block_index = 0;
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<nodeNumber ) {
		/* get neighbour range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid]==dist-1 && p[(long)tid*nodeNumber+nid]==1 ) { // p[tid][nid]
					delta[tid] = delta[tid] + (double)sigma[tid]/sigma[nid]*(1+delta[nid]);
                }
			}
		}
		else {
			t_idx = atomicInc(&block_index, per_block_buffer);
			shm_buffer[t_idx] = tid;
		}
	}
	__syncthreads();
	// dump shm_buffer to global buffer
	if (threadIdx.x<block_index) {
		int idx = threadIdx.x + block_offset;
		buffer[idx] = shm_buffer[threadIdx.x];
	}
	__syncthreads();
	// 2nd phase - nested kernel call
	if (threadIdx.x==0 && block_index!=0) {
#ifdef GPU_PROFILE
		atomicInc(nested_calls, INF);
		//  printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
		bp_process_buffer<<<block_index,NESTED_BLOCK_SIZE,0,s>>>(vertexArray, edgeArray, level, p, sigma, delta, dist, nodeNumber, buffer+block_offset, block_index);
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void backward_grid_dp_kernel(int *vertexArray, int *edgeArray, int *level, char *p,
											int *sigma, float *delta, int *buffer,
											int nodeNumber, int dist, unsigned int *idx, unsigned int *count)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
	__shared__ int shm_buffer[MAXDIMBLOCK];
	__shared__ unsigned int block_index;                            // index of each block within its sub-buffer
	__shared__ int offset;
	int t_idx = 0;                                          // used to access the buffer
	if (threadIdx.x == 0) block_index = 0;
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 1st phase
	if ( tid<nodeNumber ) {
		/* get neighbour range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid]==dist-1 && p[(long)tid*nodeNumber+nid]==1 ) { // p[tid][nid]
					delta[tid] = delta[tid] + (double)sigma[tid]/sigma[nid]*(1+delta[nid]);
                }
			}
		}
		else {
			t_idx = atomicInc(&block_index, per_block_buffer);
			shm_buffer[t_idx] = tid;
		}
	}
	__syncthreads();
	// reorganize consolidation buffer for load balance (get offset per block)
	if (threadIdx.x==0) {
		offset = atomicAdd(idx, block_index);
		//printf("blockIdx.x: %d block idx: %d idx: %d\n", blockIdx.x, block_index, offset);
	}
	__syncthreads();
	// dump shm_buffer to global buffer
	if (threadIdx.x<block_index) {
		int gm_idx = threadIdx.x + offset;
		buffer[gm_idx] = shm_buffer[threadIdx.x];
	}
	__syncthreads();
	// 2nd phase, grid level consolidation
	if (threadIdx.x==0) {
		// count up
		if ( atomicInc(count, MAXDIMGRID) >= (gridDim.x-1) ) {//
			//printf("gridDim.x: %d buffer: %d\n", gridDim.x, *idx);
#ifdef GPU_PROFILE
			atomicInc(nested_calls, INF);
#endif
			dim3 dimGridB(1,1,1);
			if (*idx<=MAXDIMGRID) {
				dimGridB.x = *idx;
			}
			else if (*idx<=MAXDIMGRID*NESTED_BLOCK_SIZE) {
				dimGridB.x = MAXDIMGRID;
				dimGridB.y = *idx/MAXDIMGRID+1;
			}
			else {
				printf("Too many elements in queue\n");
			}
			backward_prop_block_queue_kernel<<<dimGridB, NESTED_BLOCK_SIZE,0,s>>>(
														vertexArray, edgeArray, level, p,
														sigma, delta, nodeNumber, dist,
														buffer, idx );
		}
	}
}

#endif
