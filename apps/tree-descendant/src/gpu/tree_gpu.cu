#include "tree.h"
#include "stats.h"
#include "cuda_util.h"
#include "util.h"

#define PROFILE_GPU 1

#ifdef PROFILE_GPU
__device__ unsigned nested_calls = 0;

__global__ void gpu_statistics(unsigned solution){
	printf("====> GPU #%u - number of nested kernel calls:%u\n",solution, nested_calls);
}

__global__ void reset_gpu_statistics(){
	nested_calls = 0;
}
#endif

__global__ void descendants_kernel(node_t num_nodes, node_t *vertexArray, node_t *parentArray, node_t *edgeArray, node_t *descendantArray){
	unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
	for (node_t node = tid; node < num_nodes; node +=blockDim.x * gridDim.x){
		for (node_t parent = parentArray[node]; parent != (node_t) -1; parent = parentArray[parent]){
			atomicAdd(&descendantArray[parent],1);
		}
	}	
	
}

__global__ void descendants_kernel_dp(node_t node, node_t *vertexArray, node_t *parentArray, node_t *edgeArray, node_t *descendantArray){
#ifdef PROFILE_GPU
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nested_calls++;
#endif
	unsigned num_children = vertexArray[node+1]-vertexArray[node];
	for (unsigned childp = threadIdx.x; childp < num_children; childp+=blockDim.x){
		if (childp < num_children){
			node_t child = edgeArray[vertexArray[node]+childp];
			unsigned num_grandchildren = vertexArray[child+1]-vertexArray[child];
			if (num_grandchildren!=0){ 
				descendants_kernel_dp<<<1,num_grandchildren>>>(child, vertexArray, parentArray, edgeArray, descendantArray);
				cudaDeviceSynchronize();
			}
			__syncthreads();
			atomicAdd(&descendantArray[node], descendantArray[child]);
		}
	}
}

__global__ void descendants_kernel_dp_hier(node_t node, node_t *vertexArray, node_t *parentArray, node_t *edgeArray, node_t *descendantArray){
#ifdef PROFILE_GPU
	if (threadIdx.x+blockDim.x*blockIdx.x==0) nested_calls++;
#endif
	unsigned num_children = vertexArray[node+1]-vertexArray[node];
	__shared__ node_t child;
	__shared__ unsigned num_grandchildren;
	__shared__ bool recurse;
	for (unsigned childp = blockIdx.x; childp < num_children; childp+=gridDim.x){
		if (childp < num_children){
			if (threadIdx.x==0){
				child = edgeArray[vertexArray[node]+childp];
				num_grandchildren = vertexArray[child+1]-vertexArray[child];
				recurse = false;
			}
			__syncthreads();
			if (num_grandchildren != 0){
				for (unsigned grandchild_p = threadIdx.x; grandchild_p < num_grandchildren; grandchild_p+=blockDim.x){
					if (grandchild_p < num_grandchildren){
						unsigned grandchild = edgeArray[vertexArray[child]+grandchild_p];
						unsigned num_grandgrandchildren = vertexArray[grandchild+1]-vertexArray[grandchild];
						if (num_grandgrandchildren!=0) recurse = true;
					}
				}
				if (threadIdx.x==0){
					if (recurse){
						descendants_kernel_dp_hier<<<64, 64>>> (child, vertexArray, parentArray, edgeArray, descendantArray);
						cudaDeviceSynchronize();
					}else descendantArray[child]+=num_grandchildren;
				}
			}
			__syncthreads();
			if (threadIdx.x==0)atomicAdd(&descendantArray[node], descendantArray[child]);
		}
	}
}

void descendants_gpu(tree_t *tree, stats_t *stats){

	double time;

	node_t *d_vertexArray;
        node_t *d_parentArray;
        node_t *d_edgeArray;
        node_t *d_descendantArray;

	/* gpu allocation */
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( (node_t**)&d_vertexArray, sizeof(node_t)*(tree->num_nodes+1) ) );
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( (node_t**)&d_parentArray, sizeof(node_t)*(tree->num_nodes) ) );
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( (node_t**)&d_edgeArray, sizeof(node_t)*(tree->num_nodes) ) );
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( (node_t**)&d_descendantArray, sizeof(node_t)*(tree->num_nodes) ) );
	printf("GPU allocation time = %.2f ms.\n", gettime_ms()-time);

	/* memory copy from CPU to GPU */
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_vertexArray, tree->vertexArray, sizeof(node_t )*(tree->num_nodes+1), cudaMemcpyHostToDevice) );
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_parentArray, tree->parentArray, sizeof(node_t )*(tree->num_nodes), cudaMemcpyHostToDevice) );
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_edgeArray, tree->edgeArray, sizeof(node_t )*(tree->num_nodes), cudaMemcpyHostToDevice) );
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_descendantArray, tree->descendantArray_gpu, sizeof(node_t )*(tree->num_nodes), cudaMemcpyHostToDevice) );
	printf("mem copy to GPU time = %.2f ms.\n", gettime_ms()-time);

	/* GPU computation */

	//version #1 - no dynamic parallelism
	time = gettime_ms();
	descendants_kernel<<<256, 256>>>(tree->num_nodes, d_vertexArray, d_parentArray, d_edgeArray, d_descendantArray);
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	stats->gpu_time=gettime_ms()-time;
	printf("===> GPU #1 - no nested parallelism: computation time = %.2f ms.\n", stats->gpu_time);
	
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( tree->descendantArray_gpu, d_descendantArray, sizeof(int)*tree->num_nodes, cudaMemcpyDeviceToHost) );
	printf("mem copy to CPU time = %.2f ms.\n", gettime_ms()-time);

	//version #2 - dynamic parallelism
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 8);
	unsigned children = tree->vertexArray[1]-tree->vertexArray[0];
	
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_descendantArray, tree->descendantArray_gpu_np, sizeof(node_t )*(tree->num_nodes), cudaMemcpyHostToDevice) );
	cudaDeviceSynchronize();

	time = gettime_ms();
	descendants_kernel_dp<<<1,children>>>(0, d_vertexArray, d_parentArray, d_edgeArray, d_descendantArray);
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	stats->gpu_time_np=gettime_ms()-time;
	printf("===> GPU #2 - nested parallelism: computation time = %.2f ms.\n", stats->gpu_time_np);

#ifdef PROFILE_GPU
	gpu_statistics<<<1,1>>>(2);
	cudaDeviceSynchronize();
#endif
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( tree->descendantArray_gpu_np, d_descendantArray, sizeof(int)*tree->num_nodes, cudaMemcpyDeviceToHost) );
	printf("mem copy to CPU time = %.2f ms.\n", gettime_ms()-time);

	//version #3 - dynamic parallelism with hierarchy
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_descendantArray, tree->descendantArray_gpu_np_hier, sizeof(node_t )*(tree->num_nodes), cudaMemcpyHostToDevice) );
	cudaDeviceSynchronize();
#ifdef PROFILE_GPU
	reset_gpu_statistics<<<1,1>>>();
#endif
	cudaDeviceSynchronize();

	time = gettime_ms();
	descendants_kernel_dp_hier<<<children, children>>>(0, d_vertexArray, d_parentArray, d_edgeArray, d_descendantArray);
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	stats->gpu_time_np_hier=gettime_ms()-time;
	printf("===> GPU #3 - hierarchical nested parallelism: computation time = %.2f ms.\n", stats->gpu_time_np_hier);

#ifdef PROFILE_GPU
	gpu_statistics<<<1,1>>>(3);
	cudaDeviceSynchronize();
#endif
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( tree->descendantArray_gpu_np_hier, d_descendantArray, sizeof(int)*tree->num_nodes, cudaMemcpyDeviceToHost) );
	printf("mem copy to CPU time = %.2f ms.\n", gettime_ms()-time);

	//version #4 - dynamic parallelism with consolidation
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_descendantArray, tree->descendantArray_gpu_np_hier, sizeof(node_t )*(tree->num_nodes), cudaMemcpyHostToDevice) );
	cudaDeviceSynchronize();
#ifdef PROFILE_GPU
	reset_gpu_statistics<<<1,1>>>();
#endif
	cudaDeviceSynchronize();

	time = gettime_ms();
	descendants_kernel_dp_hier<<<children, children>>>(0, d_vertexArray, d_parentArray, d_edgeArray, d_descendantArray);
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	stats->gpu_time_np_hier=gettime_ms()-time;
	printf("===> GPU #3 - hierarchical nested parallelism: computation time = %.2f ms.\n", stats->gpu_time_np_hier);

#ifdef PROFILE_GPU
	gpu_statistics<<<1,1>>>(3);
	cudaDeviceSynchronize();
#endif
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( tree->descendantArray_gpu_np_hier, d_descendantArray, sizeof(int)*tree->num_nodes, cudaMemcpyDeviceToHost) );
	printf("mem copy to CPU time = %.2f ms.\n", gettime_ms()-time);

	/* memory free on GPU */
	time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaFree( d_vertexArray ));
	cudaCheckError(  __FILE__, __LINE__, cudaFree( d_parentArray ));
	cudaCheckError(  __FILE__, __LINE__, cudaFree( d_edgeArray ));
	cudaCheckError(  __FILE__, __LINE__, cudaFree( d_descendantArray ));
	printf("mem free on GPU time = %.2f ms.\n", gettime_ms()-time);
}

