#include "bfs.h"
#include "stats.h"
#include "cuda_util.h"
#include "util.h"

#define PROFILE_GPU 1

#define THREADS_PER_BLOCK 32 
#define NUM_BLOCKS 32

#ifdef PROFILE_GPU
__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
	printf("====> GPU #%u - number of kernel calls:%u\n",solution, nested_calls);
}

__global__ void reset_gpu_statistics(){
	nested_calls = 0;
}
#endif

__global__ void bfs_kernel(unsigned level, node_t num_nodes, node_t *vertexArray, node_t *edgeArray, unsigned *levelArray, bool *queue_empty){
#ifdef PROFILE_GPU
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

__global__ void bfs_kernel_dp(node_t node, node_t *vertexArray, node_t *edgeArray, unsigned *levelArray){
#ifdef PROFILE_GPU
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nested_calls++;
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
				        if (block_size!=0) bfs_kernel_dp<<<1,block_size>>>(child, vertexArray, edgeArray, levelArray);
				}
			}
		}
	}
}

__global__ void bfs_kernel_dp_hier(node_t node, node_t *vertexArray, node_t *edgeArray, unsigned *levelArray){
#ifdef PROFILE_GPU
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nested_calls++;
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
								if (num_grandgrandchildren!=0) bfs_kernel_dp_hier<<<num_grandgrandchildren,THREADS_PER_BLOCK>>>(grandchild, vertexArray, edgeArray, levelArray);
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
#ifdef PROFILE_GPU
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
		bfs_kernel<<<256, 256>>>(level,graph->num_nodes, d_vertexArray, d_edgeArray, d_levelArray, d_queue_empty);
		cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
		cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( &queue_empty, d_queue_empty, sizeof(bool), cudaMemcpyDeviceToHost) );
		level++;
	}

	stats->gpu_time=gettime_ms()-time; // end timing execution
	printf("===> GPU #1 - flat parallelism: computation time = %.2f ms.\n", stats->gpu_time);
	
#ifdef PROFILE_GPU
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
#ifdef PROFILE_GPU
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif
	
	time = gettime_ms(); // start timing execution
	
	//copy the level array from CPU to GPU	
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph->levelArray_gpu_np, sizeof(unsigned)*(graph->num_nodes), cudaMemcpyHostToDevice) );
	
	//recursive BFS traversal - naive
	unsigned children = graph->vertexArray[graph->source+1]-graph->vertexArray[graph->source];

	bfs_kernel_dp<<<1,children>>>(graph->source, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	
	stats->gpu_time_np=gettime_ms()-time; //end timing execution
	printf("===> GPU #2 - nested parallelism naive: computation time = %.2f ms.\n", stats->gpu_time_np);

#ifdef PROFILE_GPU
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
#ifdef PROFILE_GPU
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

#ifdef PROFILE_GPU
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

