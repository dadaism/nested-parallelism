#ifndef __SSSP_KERNEL__
#define __SSSP_KERNEL__

#define MAXDIMGRID 65535
#define MAXDIMBLOCK 1024

#define THRESHOLD 64
#define SHM_BUFF_SIZE 256
#define NESTED_BLOCK_SIZE 64
#define MAX_STREAM_NUM 16

#ifdef GPU_PROFILE

__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
        printf("====> GPU #%u - number of nested kernel calls:%u\n",solution, nested_calls);
}
#endif

__device__ unsigned int gm_idx_pool[MAXDIMGRID*MAXDIMBLOCK/WARP_SIZE];

__global__ void unorder_threadQueue_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
											char *update, int nodeNumber, int *queue,unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int frontierNo = *qLength;
	if ( tid<frontierNo ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		int costCurr = costArray[curr];
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			int alt = costCurr + weightArray[i];
			if ( costArray[nid] > alt ) {
				atomicMin(costArray+nid, alt);
				update[nid] = 1;	// update neighbour needed
			}
		}
	}
}

__global__ void unorder_blockQueue_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
											char *update, int nodeNumber, int *queue, unsigned int *qLength)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x; //*MAX_THREAD_PER_BLOCK + threadIdx.x;	
	int frontierNo = *qLength;
	if ( bid<frontierNo ) {
		int curr = queue[bid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		int costCurr = costArray[curr];
		for (int eid=threadIdx.x+start; eid<end; eid += blockDim.x) {
			int nid = edgeArray[eid];	// neighbour id
			int alt = costCurr + weightArray[eid];
			if ( costArray[nid] > alt ) {
				atomicMin(costArray+nid, alt);				
				update[nid] = 1;	// update neighbour needed
			}
		}
	}
}


/* processes the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void sssp_process_buffer( int *vertexArray, int *edgeArray, int *weightArray, int *costArray, 
				     char *update, int nodeNumber, int *buffer, unsigned int buffer_length)
{
	int bid = blockIdx.x; 
	if ( bid<buffer_length ) {   // block-based mapping
		int curr = buffer[bid];	//nodes processed by current block
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		int costCurr = costArray[curr];
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x){ // eid is the identifier of the edge processed by the current thread
			if ( eid<end ){
				int nid = edgeArray[eid];	// neighbour id
				int alt = costCurr + weightArray[eid];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);				
					update[nid] = 1;	// update neighbour needed
				}
			}
		}
	}
}

/* LOAD BALANCING THROUGH DELAYED BUFFER */

/* implements a delayed buffer in shared memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - in phase 2, the blocks access the nodes in the delayed-buffer in a block-based mapping (one neighbor per thread)
*/
__global__ void unorder_threadQueue_shared_delaybuffer_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
								char *update, int nodeNumber, int *queue,unsigned int *queue_length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int t_idx = 0; //thread-based variable used to index inside the dealyed buffer
	__shared__ int buffer[SHM_BUFF_SIZE]; //delayed buffer
	__shared__ unsigned int idx; //index within the delayed buffer
	
	if (threadIdx.x==0)	idx = 0;
	__syncthreads();

	// 1st phase - thread-based mapping
	if ( tid<*queue_length ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edgeNum = end - start;
		if ( edgeNum<THRESHOLD ) {
			/* access neighbours */
			int costCurr = costArray[curr];
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				int alt = costCurr + weightArray[i];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);
					update[nid] = 1;	// update neighbour needed
				}
			}
		}
		else { // insert into delayed buffer
			t_idx = atomicInc(&idx, SHM_BUFF_SIZE);
			buffer[t_idx] = curr;
		}
	}
	__syncthreads();
	// 2nd phase - each block processed all the elements in its shared memory buffer; each thread processes a different neighbor
#ifdef GPU_PROFILE
	if (tid==0 && idx!=0) {
		printf("In Block %d # delayed nodes : %d\n", blockIdx.x, idx);
	}
#endif
	for (int i=0; i<idx; i++) {
		int curr = buffer[i];	//grab an element from the buffer
		// get neighbour range		
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int costCurr = costArray[curr];
		//access neighbors - one thread per neigbor;
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x){
			if ( eid < end ){
				int nid = edgeArray[eid];	// neighbour id
				int alt = costCurr + weightArray[eid];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);				
					update[nid] = 1;	// update neighbour needed
				}
			}
		}
	}
}

/* implements phase 1 of delayed buffer (buffer) in global memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - phase 2 must be implemented by separately invoking the "sssp_process_buffer" kernel
*/

__global__ void unorder_threadQueue_global_delaybuffer_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
						  char *update, int nodeNumber, int *queue, unsigned int *queue_length,
						  int *buffer, unsigned int *idx)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int t_idx = 0;
	// 1st phase
	if ( tid<*queue_length ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edgeNum = end - start;
		if ( edgeNum<THRESHOLD ) {
			/* access neighbours */
			int costCurr = costArray[curr];
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				int alt = costCurr + weightArray[i];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);
					update[nid] = 1;	// update neighbour needed
				}
			}
		}
		else { // insert into delayed buffer in global memory
			t_idx = atomicInc(idx, GM_BUFF_SIZE);
			buffer[t_idx] = queue[tid];
		}
	}
}

/* LOAD BALANCING THROUGH DYNAMIC PARALLELISM */

/* Child kernel invoked by the dynamic parallelism implementation with multiple kernel calls
   This kernel processes the neighbors of a certain node. The starting and ending point (start and end parameters) within the edge array are given as parameter
*/
__global__ void sssp_process_neighbors(	int *edgeArray, int *weightArray, int *costArray, 
					char *update, int costCurr, int start, int end)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x + start;
	if (tid < end) {
       		int nid = edgeArray[tid];
		int alt = costCurr + weightArray[tid];
		if ( costArray[nid] > alt ) {
			atomicMin(costArray+nid, alt);
			update[nid] = 1;	// update neighbour needed
		}
	}
}

/* thread queue with dynamic parallelism and potentially multiple nested kernel calls */
__global__ void unorder_threadQueue_multiple_dp_kernel(	int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
							char *update, int nodeNumber, int *queue,unsigned int *queue_length)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	int frontierNo = *queue_length;
	cudaStream_t s[MAX_STREAM_NUM];
	for (int i=0; i<MAX_STREAM_NUM; ++i) {
		cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
		//cudaStreamCreateWithFlags(&s[i], cudaStreamDefault);
	}
	if ( tid<frontierNo ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */				
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edgeNum = end - start;
		if ( edgeNum<THRESHOLD ) {
			/* access neighbours */
			int costCurr = costArray[curr];
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				int alt = costCurr + weightArray[i];
				if ( costArray[nid] > alt ) {
					atomicMin(costArray+nid, alt);
					update[nid] = 1;	// update neighbour needed
				}
			}
		}
		else {
#ifdef GPU_PROFILE
		nested_calls++;
		//	printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
		int costCurr = costArray[curr];

		sssp_process_neighbors<<<edgeNum/NESTED_BLOCK_SIZE+1, NESTED_BLOCK_SIZE, 0, s[threadIdx.x%MAX_STREAM_NUM] >>>(
					 	 edgeArray, weightArray, costArray, update, costCurr, start, end);

		}
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void unorder_threadQueue_single_dp_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
                                                  char *update, int nodeNumber, int *queue, unsigned int *queue_length,
                                                  int *buffer)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x; 	// amount of the buffer available to each thread block
	unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    unsigned int *block_index = &gm_idx_pool[blockIdx.x];				// index of each block within its sub-buffer
    int t_idx = 0;						// used to access the buffer
	if (threadIdx.x == 0) *block_index = 0;
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // 1st phase
    if ( tid<*queue_length ) {
        int curr = queue[tid];  //      grab a work from queue, tid is queue index
        /* get neighbour range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        int edgeNum = end - start;
        if ( edgeNum<THRESHOLD ) {
            /* access neighbours */
            int costCurr = costArray[curr];
            for (int i=start; i<end; ++i) {
     	        int nid = edgeArray[i];
                int alt = costCurr + weightArray[i];
                if ( costArray[nid] > alt ) {
        	         atomicMin(costArray+nid, alt);
                     update[nid] = 1;        // update neighbour needed
                }
            }
        }
        else { // insert into delayed buffer in global memory
        	t_idx = atomicInc(block_index, per_block_buffer);
            buffer[t_idx+block_offset] = queue[tid];
        }
	}
	__syncthreads();
 
    //2nd phase - nested kernel call
	if (threadIdx.x==0 && *block_index!=0){
#ifdef GPU_PROFILE
		nested_calls++;
#endif
      	sssp_process_buffer<<<*block_index,NESTED_BLOCK_SIZE,0,s>>>( vertexArray, edgeArray, weightArray, costArray,
        						  		update, nodeNumber, buffer+block_offset, *block_index);
        }
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void consolidate_warp_dp_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
                                                  char *update, int nodeNumber, int *queue, unsigned int *queue_length,
                                                  int *buffer)
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

	unsigned int *warp_index = &gm_idx_pool[blockIdx.x * warpDim + warpId];				// index of each block within its sub-buffer
	int t_idx = 0;						// used to access the buffer
	*warp_index = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 1st phase
    if ( tid<*queue_length ) {
        int curr = queue[tid];  //      grab a work from queue, tid is queue index
        /* get neighbor range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        int edgeNum = end - start;
        if ( edgeNum<THRESHOLD ) {
            /* access neighbors */
            int costCurr = costArray[curr];
            for (int i=start; i<end; ++i) {
     	        int nid = edgeArray[i];
                int alt = costCurr + weightArray[i];
                if ( costArray[nid] > alt ) {
        	         atomicMin(costArray+nid, alt);
                     update[nid] = 1;        // update neighbour needed
                }
            }
        }
        else { // insert into delayed buffer in global memory
        	t_idx = atomicInc(warp_index, per_warp_buffer);
            buffer[t_idx+warp_offset] = queue[tid];
        }
	}

    //2nd phase - nested kernel call
	if (threadIdx.x%WARP_SIZE==0 && *warp_index!=0){
#ifdef GPU_PROFILE
		nested_calls++;
#endif
      	sssp_process_buffer<<<*warp_index,NESTED_BLOCK_SIZE,0, s[warpId%MAX_STREAM_NUM]>>>( vertexArray, edgeArray, weightArray, costArray,
        						  		update, nodeNumber, buffer+warp_offset, *warp_index);
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void consolidate_block_dp_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
                                                  char *update, int nodeNumber, int *queue, unsigned int *queue_length,
                                                  int *buffer)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x; 	// amount of the buffer available to each thread block
	unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    __shared__ int shm_buffer[MAXDIMBLOCK];
	unsigned int *block_index = &gm_idx_pool[blockIdx.x];	// index of each block within its sub-buffer
	int t_idx = 0;						// used to access the buffer
	if (threadIdx.x == 0) *block_index = 0;
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // 1st phase
    if ( tid<*queue_length ) {
        int curr = queue[tid];  //      grab a work from queue, tid is queue index
        /* get neighbour range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        int edgeNum = end - start;
        if ( edgeNum<THRESHOLD ) {
            /* access neighbours */
            int costCurr = costArray[curr];
            for (int i=start; i<end; ++i) {
     	        int nid = edgeArray[i];
                int alt = costCurr + weightArray[i];
                if ( costArray[nid] > alt ) {
        	         atomicMin(costArray+nid, alt);
                     update[nid] = 1;        // update neighbour needed
                }
            }
        }
        else { // insert into delayed buffer in global memory
        	t_idx = atomicInc(block_index, per_block_buffer);
            //buffer[t_idx+block_offset] = queue[tid];
        	shm_buffer[t_idx] = queue[tid];
        }
	}
	__syncthreads();
	// dump shm_buffer to global buffer
	if (threadIdx.x<*block_index) {
		int idx = threadIdx.x + block_offset;
		buffer[idx] = shm_buffer[threadIdx.x];
	}
	__syncthreads();

    //2nd phase - nested kernel call
	if (threadIdx.x==0 && *block_index!=0){
#ifdef GPU_PROFILE
		nested_calls++;
#endif
      	sssp_process_buffer<<<*block_index,NESTED_BLOCK_SIZE,0,s>>>( vertexArray, edgeArray, weightArray, costArray,
        						  		update, nodeNumber, buffer+block_offset, *block_index);
        }
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void consolidate_grid_dp_kernel( int *vertexArray, int *edgeArray, int *costArray, int *weightArray,
                                                  char *update, int nodeNumber, int *queue, unsigned int *queue_length,
                                                  int *buffer, unsigned int *idx, unsigned int *count)
{
	cudaStream_t s;
	cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x; 	// amount of the buffer available to each thread block
	//unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    __shared__ int shm_buffer[MAXDIMBLOCK];
	unsigned int *block_index = &gm_idx_pool[blockIdx.x];				// index of each block within its sub-buffer
	__shared__ int offset;
	int t_idx = 0;						// used to access the buffer
	if (threadIdx.x == 0) *block_index = 0;
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
        // 1st phase
    if ( tid<*queue_length ) {
        int curr = queue[tid];  //      grab a work from queue, tid is queue index
        /* get neighbour range */
        int start = vertexArray[curr];
        int end = vertexArray[curr+1];
        int edgeNum = end - start;
        if ( edgeNum<THRESHOLD ) {
            /* access neighbours */
            int costCurr = costArray[curr];
            for (int i=start; i<end; ++i) {
     	        int nid = edgeArray[i];
                int alt = costCurr + weightArray[i];
                if ( costArray[nid] > alt ) {
        	         atomicMin(costArray+nid, alt);
                     update[nid] = 1;        // update neighbour needed
                }
            }
        }
        else { // insert into delayed buffer in global memory
        	t_idx = atomicInc(block_index, per_block_buffer);
            //buffer[t_idx+block_offset] = queue[tid];
        	shm_buffer[t_idx] = queue[tid];
        }
	}
	__syncthreads();
	// reorganize consolidation buffer for load balance (get offset per block)
	if (threadIdx.x==0) {
		offset = atomicAdd(idx, *block_index);
		//printf("blockIdx.x: %d block idx: %d idx: %d\n", blockIdx.x, block_index, offset);
	}
	__syncthreads();
	// dump shm_buffer to global buffer
	if (threadIdx.x<*block_index) {
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
			nested_calls++;
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
			unorder_blockQueue_kernel<<<dimGridB, NESTED_BLOCK_SIZE,0,s>>>(	vertexArray, edgeArray, costArray,
																		weightArray, update, nodeNumber,
																		buffer, idx);
		}
	}
}

/* LOAD BALANCING BY USING MULTIPLE QUEUES */
/* divides the nodes into two queues */ 
__global__ void unorder_gen_multiQueue_kernel(	int *vertexArray, char *update, int nodeNumber, 
						int *queue_l, unsigned int *qCounter_l, unsigned int qMaxLength_l,
						int *queue_h, unsigned int *qCounter_h, unsigned int qMaxLength_h)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		int edgeNum = end - start;
		if ( edgeNum<THRESHOLD ) {
			/* write vertex number to LOW degree queue */
			unsigned int qIndex = atomicInc(qCounter_l, qMaxLength_l);
			queue_l[qIndex] = tid;
		}
		else {
			/* write vertex number to HIGH degree queue */
			unsigned int qIndex = atomicInc(qCounter_h, qMaxLength_h);
			queue_h[qIndex] = tid;
		}
	}
}
__global__ void unorder_generateQueue_kernel(	char *update, int nodeNumber, int *queue, 
												unsigned int *qCounter, unsigned int qMaxLength)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	if ( tid<nodeNumber && update[tid] ) {
		update[tid] = 0;
		/* write node number to queue */
		unsigned int qIndex = atomicInc(qCounter, qMaxLength);
		queue[qIndex] = tid;
	}
}
#endif
