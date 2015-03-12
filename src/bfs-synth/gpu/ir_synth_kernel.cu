#ifndef __UBFS_KERNEL__
#define __UBFS_KERNEL__

#define MAX_LEVEL 9999
#define MAXDIMGRID 65535
#define MAX_THREAD_PER_BLOCK 1024

#define THREASHOLD 150
#define SHM_BUFF_SIZE 256
#define NESTED_BLOCK_SIZE 128


__device__ void work()
{
	int i=0;
	++i;
}

__global__ void bfs_bitmap_init_kernel(char *frontier, int *level, int source, int nodeNumber)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber ) {
		if ( tid==source ) {
			frontier[tid] = 1;
			level[tid] = 0;
		}
		else {
			frontier[tid] = 0;
			level[tid] = MAX_LEVEL;
		}
	}
}

__global__ void bfs_queue_init_kernel(int *level, int *queue, unsigned int *queue_length, int source, int nodeNumber)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid<nodeNumber ) {
		if ( tid==source ) {
			level[tid] = 0;
			queue[0] = source;
			*queue_length = 1;
		}
		else {
			level[tid] = MAX_LEVEL;
		}
	}
}

__global__ void bfs_bitmap_kernel(	int *vertexArray, int *edgeArray, int *level, 
									char *frontier, char *update, int nodeNumber )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	if ( tid<nodeNumber && frontier[tid] ) {
		frontier[tid] = 0;
		/* get neighbour range */
		int start = vertexArray[tid];
		int end = vertexArray[tid+1];
		/* access neighbours */
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			if ( level[nid] > level[tid]+1 ) {
				level[nid] = level[tid]+1;	// set level number
				update[nid] = 1;	// set as frontier
				work();
			}
		}
	}
}

__global__ void bfs_queue_kernel(	int *vertexArray, int *edgeArray, int *level, 
									int *queue, unsigned int *queue_length, char *update, int nodeNumber )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontier_size = *queue_length;
	if ( tid<frontier_size ) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		//printf("%d\n", curr);
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		//printf("start: %d  end: %d\n", start, end);
		/* access neighbours */
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			//printf("%d - > %d\n", curr, nid);
			if ( level[nid] > level[curr]+1 ) {
				level[nid] = level[curr]+1;	// set level number
				update[nid] = 1;	// set as frontier
				work();
			}
		}
	}
}

/* LOAD BALANCING THROUGH DELAYED BUFFER */

/* implements a delayed buffer in shared memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - in phase 2, the blocks access the nodes in the delayed-buffer in a block-based mapping (one neighbor per thread)
*/
__global__ void bfs_queue_shared_delayed_buffer_kernel(	int *vertexArray, int *edgeArray, int *level, 
									int *queue, unsigned int *queue_length, char *update, int nodeNumber )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontierNo = *queue_length;

	int t_idx = 0;	//	thread-based variable used to index inside the delayed buffer
	__shared__ int buffer[SHM_BUFF_SIZE]; //delayed buffer
	__shared__ unsigned int idx; //index within the delayed buffer

	if (threadIdx.x==0) idx = 0;
	
	__syncthreads();
	
	// 1st phase - thread-based mapping
	if ( tid<frontierNo) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edge_num = end - start;
		if ( edge_num < THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				//printf("%d - > %d\n", curr, nid);
				if ( level[nid] > level[curr]+1 ) {
					level[nid] = level[curr]+1;	// set level number
					update[nid] = 1;	// set as frontier
					work();
				}
			}
		}
		else {	//	insert into delayed buffer
			t_idx = atomicInc(&idx, SHM_BUFF_SIZE);
			buffer[t_idx] = curr;
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
		int curr = buffer[i]; //grab an element from the buffer
		// get neighbour range    
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		// access neighbors - one thread per neigbor;
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x){
			if ( eid < end ) {
        		int nid = edgeArray[eid]; // neighbour id
    			if ( level[nid] > level[curr]+1 ) {
					level[nid] = level[curr]+1;	// set level number
					update[nid] = 1;	// set as frontier
					work();
				}
      		}
    	}
	}
}

/* implements phase 1 of delayed buffer (buffer) in global memory:
   - in phase 1, the threads access the nodes in the queue with a thread-based mapping (one node per thread)
   - phase 2 must be implemented by separately invoking the "process_buffer" kernel
*/
__global__ void bfs_queue_global_delayed_buffer_kernel(	int *vertexArray, int *edgeArray, int *level, 
									int *queue, unsigned int *queue_length, char *update, 
									int *buffer, unsigned int *idx, int nodeNumber )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontierNo = *queue_length;
	int t_idx = 0;

	// 1st phase
	if ( tid<frontierNo) {
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid] > level[curr]+1 ) {
					level[nid] = level[curr]+1;	// set level number
					update[nid] = 1;	// set as frontier
					work();
				}
			}
		}
		else {
			t_idx = atomicInc(idx, GM_BUFF_SIZE);
			buffer[t_idx] = queue[tid];
		}
	}
}

/* LOAD BALANCING THROUGH DYNAMIC PARALLELISM */
/* Child kernel invoked by the dynamic parallelism implementation with multiple kernel calls
   This kernel processes the neighbors of a certain node. The starting and ending point (start and end parameters) within the edge array are given as parameter
*/
__global__ void process_neighbors( int *edgeArray, int *level, char *update, int start, int end)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x + start;
	if (tid < end) {
		int nid = edgeArray[tid];
		if ( level[nid] > level[tid]+1 ) {
			level[nid] = level[tid]+1;
			update[nid] = 1;  // update neighbour needed
			work();
		}
	}
}

__global__ void bfs_thread_queue_multidp_kernel(int *vertexArray, int *edgeArray, int *level, char *update, 
												int nodeNumber, int *queue, unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontierNo = *qLength;
	if ( tid<frontierNo ){
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {		
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid] > level[curr]+1 ){	// neighbour's level can be reduced
					level[nid] = level[curr]+1;
					update[nid] = 1;	
					work();
				}
			}
		}
		else {
#ifdef GPU_PROFILE
			nested_calls++;
			//  printf("calling nested kernel for %d neighbors\n", edgeNum);
#endif
     		process_neighbors<<<edge_num/NESTED_BLOCK_SIZE+1, NESTED_BLOCK_SIZE>>>(
 				           edgeArray, level, update, start, end);
		}
	}
}

/* processes the elements in a buffer in block-based fashion. The buffer stores nodes ids in a queue */
__global__ void process_buffer( int *vertexArray, int *edgeArray, int *level, char *update, 
								int nodeNumber, int *buffer, unsigned int buffer_size)
{
	int bid = blockIdx.x;
	if ( bid<buffer_size ) {   // block-based mapping
		int curr = buffer[bid]; //nodes processed by current block
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		for (int eid=start+threadIdx.x; eid<end; eid+=blockDim.x) { // eid is the identifier of the edge processed by the current thread
			if ( eid<end ) {
				int nid = edgeArray[eid]; // neighbour id
				if ( level[nid] > level[curr]+1 ) {
					level[nid] = level[curr]+1;
					update[nid] = 1;  // update neighbour needed
        			work();
				}
			}
		}
	}
}

/* thread queue with dynamic parallelism and a single nested kernel call per thread-block*/
__global__ void bfs_thread_queue_singledp_kernel(int *vertexArray, int *edgeArray, int *level, char *update, 
										int nodeNumber, int *queue, unsigned int *qLength, int *buffer)
{
	unsigned per_block_buffer = GM_BUFF_SIZE/gridDim.x;     // amount of the buffer available to each thread block
    unsigned block_offset = blockIdx.x * per_block_buffer;  // block offset within the buffer
    __shared__ unsigned int block_index;                            // index of each block within its sub-buffer
    int t_idx = 0;                                          // used to access the buffer
    if (threadIdx.x == 0) block_index = 0;
    __syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontierNo = *qLength;
	// 1st phase
	if ( tid<frontierNo ){
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		int edge_num = end - start;
		if ( edge_num<THREASHOLD ) {
			/* access neighbours */
			for (int i=start; i<end; ++i) {
				int nid = edgeArray[i];
				if ( level[nid] > level[curr]+1 ){	// neighbour's level can be reduced
					level[nid] = level[curr]+1;
					update[nid] = 1;	
					work();
				}
			}
		}
		else {
			t_idx = atomicInc(&block_index, per_block_buffer);
			buffer[t_idx+block_offset] = queue[tid];
		}
	}
	__syncthreads();
	
	//2nd phase - nested kernel call
	if (threadIdx.x==0 && block_index!=0){
#ifdef GPU_PROFILE
		nested_calls++;
#endif
		process_buffer<<<block_index,NESTED_BLOCK_SIZE>>>( vertexArray, edgeArray, level, update, 
															nodeNumber, buffer+block_offset, block_index);
	}
}


__global__ void bfs_thread_queue_kernel(int *vertexArray, int *edgeArray, int *level, char *update, 
										int nodeNumber, int *queue, unsigned int *qLength)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int frontierNo = *qLength;
	if ( tid<frontierNo ){
		int curr = queue[tid];	//	grab a work from queue, tid is queue index
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		for (int i=start; i<end; ++i) {
			int nid = edgeArray[i];
			int nlevel = level[nid];
			int tlevel = level[curr];
			if ( nlevel > tlevel+1 ){	// neighbour's level can be reduced
				atomicMin(level+nid, tlevel + 1);
				update[nid] = 1;	
				work();
			}
		}
	}
}

__global__ void bfs_block_queue_kernel(	int *vertexArray, int *edgeArray, int *level, char *update, 
										int nodeNumber, int *queue, unsigned int *qLength)
{
	int bid = blockIdx.x+blockIdx.y*gridDim.x;	//*MAX_THREAD_PER_BLOCK + threadIdx.x;
	int frontierNo = *qLength;
	if ( bid<frontierNo ){
		int curr = queue[bid];	//	grab a work from queue, bid is queue index
		/* get neighbour range */
		int start = vertexArray[curr];
		int end = vertexArray[curr+1];
		/* access neighbours */
		for (int i=start+threadIdx.x; i<end; i+=blockDim.x) {
			int nid = edgeArray[i];
			int nlevel = level[nid];
			int tlevel = level[curr];
			if ( nlevel > tlevel+1 ){	// neighbour's level can be reduced
				atomicMin(level+nid, tlevel + 1);
				update[nid] = 1;	
				work();
			}
		}
	}
}

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
