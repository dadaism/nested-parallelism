#include "bfs.h"
#include "stats.h"
#include "cuda_util.h"
#include "util.h"

#ifndef PROFILE_GPU //used to record the number of kernel calls performed
#define PROFILE_GPU 0
#endif

#ifndef THREADS_PER_BLOCK_FLAT //block size for flat parallelism
#define THREADS_PER_BLOCK_FLAT 256 
#endif

#ifndef NUM_BLOCKS_FLAT //number of blocks for flat parallelism
#define NUM_BLOCKS_FLAT 256
#endif

#ifndef THREADS_PER_BLOCK //block size
#define THREADS_PER_BLOCK 32 
#endif

#ifndef NUM_BLOCKS //number of blocks
#define NUM_BLOCKS 32
#endif

#ifndef STREAMS //number of streams
#define STREAMS 0
#endif 

#if (PROFILE_GPU!=0)
// records the number of kerbel calls performed
__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
	printf("====> GPU #%u - number of kernel calls:%u\n",solution, nested_calls);
}

__global__ void reset_gpu_statistics(){
	nested_calls = 0;
}
#endif

// iterative, flat BFS traversal (note: synchronization-free implementation)
__global__ void bfs_kernel_flat(unsigned level, node_t num_nodes, node_t *vertexArray, node_t *edgeArray, unsigned *levelArray, bool *queue_empty){
#if (PROFILE_GPU!=0)
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nested_calls++;
#endif
	unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
	for (node_t node = tid; node < num_nodes; node +=blockDim.x * gridDim.x){
		if(node < num_nodes && levelArray[node]==level){
			for (node_t edge=vertexArray[node];edge<vertexArray[node+1];edge++){
				node_t neighbor=edgeArray[edge];
				if (levelArray[neighbor]==UNDEFINED || levelArray[neighbor]>(level+1)){
					levelArray[neighbor]=level+1;
					*queue_empty=false;
				}
			}	
		}
	}	
}

// recursive naive NFS traversal
__global__ void bfs_kernel_dp(node_t node, node_t *vertexArray, node_t *edgeArray, unsigned *levelArray){
#if (PROFILE_GPU!=0)
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nested_calls++;
#endif

#if (STREAMS!=0)
	cudaStream_t s[STREAMS];
	for (int i=0; i<STREAMS; ++i)  cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);	
#endif

	unsigned num_children = vertexArray[node+1]-vertexArray[node];
	for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x){ // may change this to use multiple blocks
		if (childp < num_children){
			node_t child = edgeArray[vertexArray[node]+childp];
			unsigned node_level = levelArray[node];
			unsigned child_level = levelArray[child];
			if (child_level==UNDEFINED || child_level>(node_level+1)){
				unsigned old_level = atomicMin(&levelArray[child],node_level+1);
				if (old_level == child_level){
					unsigned num_grandchildren=vertexArray[child+1]-vertexArray[child];
					unsigned block_size = min(num_grandchildren, THREADS_PER_BLOCK);
#if (STREAMS!=0)
				        if (block_size!=0) bfs_kernel_dp<<<1,block_size, 0, s[threadIdx.x%STREAMS]>>>(child, vertexArray, edgeArray, levelArray);
#else
				        if (block_size!=0) bfs_kernel_dp<<<1,block_size>>>(child, vertexArray, edgeArray, levelArray);

#endif
				}
			}
		}
	}
}

// recursive hierarchical BFS traversal
__global__ void bfs_kernel_dp_hier(node_t node, node_t *vertexArray, node_t *edgeArray, unsigned *levelArray){
#if (PROFILE_GPU!=0)
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nested_calls++;
#endif

#if (STREAMS!=0)
	cudaStream_t s[STREAMS];
	for (int i=0; i<STREAMS; ++i)  cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);	
#endif
	__shared__ node_t child;
	__shared__ unsigned child_level;
	__shared__ unsigned num_grandchildren;
	
	unsigned node_level = levelArray[node];
	unsigned num_children = vertexArray[node+1]-vertexArray[node];
	
	for (unsigned childp = blockIdx.x; childp < num_children; childp+=gridDim.x){
		if (childp < num_children){
			if (threadIdx.x==0){
				child = edgeArray[vertexArray[node]+childp];
				num_grandchildren = 0; // by default, due not continue
				child_level = levelArray[child];
				if (child_level==UNDEFINED || child_level>(node_level+1)){
					unsigned old_level = atomicMin(&levelArray[child],node_level+1);
					if (old_level == child_level)
						num_grandchildren = vertexArray[child+1]-vertexArray[child];
				}
			}
			__syncthreads();
			if (num_grandchildren != 0){
				for (unsigned grandchild_p = threadIdx.x; grandchild_p < num_grandchildren; grandchild_p+=blockDim.x){
					if (grandchild_p < num_grandchildren){
						unsigned grandchild = edgeArray[vertexArray[child]+grandchild_p];
						unsigned grandchild_level = levelArray[grandchild];
						if (grandchild_level == UNDEFINED || grandchild_level > (node_level + 2)){
							unsigned old_level = atomicMin(&levelArray[grandchild],node_level+2);
							if (old_level == grandchild_level){
								unsigned num_grandgrandchildren = vertexArray[grandchild+1]-vertexArray[grandchild];
#if (STREAMS!=0)
								if (num_grandgrandchildren!=0) bfs_kernel_dp_hier<<<num_grandgrandchildren,THREADS_PER_BLOCK, 0, s[threadIdx.x%STREAMS]>>>(grandchild, vertexArray, edgeArray, levelArray);
#else 
								if (num_grandgrandchildren!=0) bfs_kernel_dp_hier<<<num_grandgrandchildren,THREADS_PER_BLOCK>>>(grandchild, vertexArray, edgeArray, levelArray);
#endif
							}
						}
					}
				}
			}
			__syncthreads();
		}
	}
}

void bfs_gpu(graph_t *graph, stats_t *stats){

	printf("bfs_gpu invoked: using %d streams\n",STREAMS);
	stats->streams=STREAMS;

	double time;

	//graph topology
	node_t *d_vertexArray;
        node_t *d_edgeArray;
        unsigned *d_levelArray;

	/* gpu allocation of vertex, edge and level array*/
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( (node_t**)&d_vertexArray, sizeof(node_t)*(graph->num_nodes+1) ) );
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( (node_t**)&d_edgeArray, sizeof(node_t)*(graph->num_edges) ) );
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( (unsigned**)&d_levelArray, sizeof(unsigned)*(graph->num_nodes) ) );
	printf("GPU allocation time = %.2f ms.\n", gettime_ms()-time);

	/* memory copy from CPU to GPU (vertex and edge array) */
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_vertexArray, graph->vertexArray, sizeof(node_t )*(graph->num_nodes+1), cudaMemcpyHostToDevice) );
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_edgeArray, graph->edgeArray, sizeof(node_t )*(graph->num_edges), cudaMemcpyHostToDevice) );
	printf("mem copy to GPU time = %.2f ms.\n", gettime_ms()-time);

	/* GPU computation */

	// ----------------------------------------------------------
	// version #1 - flat parallelism - level-based BFS traversal
	// ----------------------------------------------------------
#if (PROFILE_GPU!=0)
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif
	time = gettime_ms(); // start timing execution

	//copy the level array from CPU to GPU	
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph->levelArray_gpu, sizeof(unsigned )*(graph->num_nodes), cudaMemcpyHostToDevice) );
	
	//used for termination
	bool queue_empty = false;
	bool *d_queue_empty;
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_queue_empty, sizeof(bool)) );

	unsigned level = 0;	

	//level-based traversal
	while (!queue_empty){
		cudaCheckError(  __FILE__, __LINE__, cudaMemset( d_queue_empty, true, sizeof(bool)) );
		bfs_kernel_flat<<<NUM_BLOCKS_FLAT, THREADS_PER_BLOCK_FLAT>>>(level,graph->num_nodes, d_vertexArray, d_edgeArray, d_levelArray, d_queue_empty);
		cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
		cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( &queue_empty, d_queue_empty, sizeof(bool), cudaMemcpyDeviceToHost) );
		level++;
	}

	stats->gpu_time=gettime_ms()-time; // end timing execution
	printf("===> GPU #1 - flat parallelism: computation time = %.2f ms.\n", stats->gpu_time);
	
#if (PROFILE_GPU!=0)
	gpu_statistics<<<1,1>>>(1);
	cudaDeviceSynchronize();
#endif
	//copy the level array from GPU to CPU;
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( graph->levelArray_gpu, d_levelArray, sizeof(unsigned)*graph->num_nodes, cudaMemcpyDeviceToHost) );
	printf("mem copy to CPU time = %.2f ms.\n", gettime_ms()-time);

	// ----------------------------------------------------------
	// version #2 - dynamic parallelism - naive 
	// ----------------------------------------------------------
#if (PROFILE_GPU!=0)
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif
	
	time = gettime_ms(); // start timing execution
	
	//copy the level array from CPU to GPU	
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph->levelArray_gpu_np, sizeof(unsigned)*(graph->num_nodes), cudaMemcpyHostToDevice) );
	
	//recursive BFS traversal - naive
	unsigned children = graph->vertexArray[graph->source+1]-graph->vertexArray[graph->source];
	unsigned block_size = min (children, THREADS_PER_BLOCK);
	bfs_kernel_dp<<<1,block_size>>>(graph->source, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	
	stats->gpu_time_np=gettime_ms()-time; //end timing execution
	printf("===> GPU #2 - nested parallelism naive: computation time = %.2f ms.\n", stats->gpu_time_np);

#if (PROFILE_GPU!=0)
	gpu_statistics<<<1,1>>>(2);
	cudaDeviceSynchronize();
#endif
	
	//copy the level array from GPU to CPU;
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( graph->levelArray_gpu_np, d_levelArray, sizeof(unsigned)*graph->num_nodes, cudaMemcpyDeviceToHost) );
	printf("mem copy to CPU time = %.2f ms.\n", gettime_ms()-time);

	// ----------------------------------------------------------
	// version #3 - dynamic parallelism - hierarchical
	// ----------------------------------------------------------
#if (PROFILE_GPU!=0)
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif
	
	time = gettime_ms(); // start timing execution
	
	//copy the level array from CPU to GPU	
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph->levelArray_gpu_np_hier, sizeof(unsigned)*(graph->num_nodes), cudaMemcpyHostToDevice) );

	//recursive BFS traversal - hierarchical
	children = graph->vertexArray[graph->source+1]-graph->vertexArray[graph->source];
	bfs_kernel_dp_hier<<<children, THREADS_PER_BLOCK>>>(graph->source, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	
	stats->gpu_time_np_hier=gettime_ms()-time; //end timing execution
	printf("===> GPU #3 - nested parallelism hierarchical: computation time = %.2f ms.\n", stats->gpu_time_np_hier);

#if (PROFILE_GPU!=0)
	gpu_statistics<<<1,1>>>(3);
	cudaDeviceSynchronize();
#endif
	
	//copy the level array from GPU to CPU;
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( graph->levelArray_gpu_np_hier, d_levelArray, sizeof(unsigned)*graph->num_nodes, cudaMemcpyDeviceToHost) );
	printf("mem copy to CPU time = %.2f ms.\n", gettime_ms()-time);
	

	/* memory free on GPU */
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaFree( d_vertexArray ));
	cudaCheckError(  __FILE__, __LINE__, cudaFree( d_edgeArray ));
	cudaCheckError(  __FILE__, __LINE__, cudaFree( d_levelArray ));
	printf("mem free on GPU time = %.2f ms.\n", gettime_ms()-time);
}

