#include <stdio.h>
#include "tree.h"
#include "util.h"
#include "stats.h"
#include "cuda_util.h"

static stats_t stats;

void validateArrays(node_t n, node_t *array1, node_t *array2, const char *message){
	for (node_t node=0; node<n;node++){
		if (array1[node]!=array2[node]){
			printf("ERROR: validation error at %llu: %s !\n", node, message);
			break;
		}
	}
}

int main(int argc, char *argv[])
{
    	init_device(DEVICE);
	double time;
	tree_t tree;
	if (argc < 4){
		printf ("usage: ./run_tree NUM_LEVELS, OUTDEGRE_MIN, OUTDEGREE_MAX\n");
		return(-1);
	}
	unsigned num_levels = atoi(argv[1]);
	unsigned outdegree_min  =  atoi(argv[2]);
	unsigned outdegree_max  =  atoi(argv[3]);
	char *filename = "test/tree.dot";	
	
	printf("\n===MAIN=== :: [levels,outdegree_min,outdegree_max] = %u, %u, %u\n", num_levels, outdegree_min, outdegree_max);

	gen_random_tree(&tree, num_levels, outdegree_min, outdegree_max);

	printf("nodes=%llu\n",tree.num_nodes);

	stats.levels=num_levels;
	stats.outdegree_min=outdegree_min;
	stats.outdegree_max=outdegree_max;
	stats.num_nodes=tree.num_nodes;	

	//compute descendants iteratively
	time = gettime_ms();
	if (tree.num_nodes!=0) descendants(&tree);
	stats.cpu_time_it=gettime_ms()-time;
	printf("===> CPU #1 time to compute descendants = %.2f ms.\n",gettime_ms()-time);
	
	//compute descendants recursively
	time = gettime_ms();
	if (tree.num_nodes!=0) descendants_rec(&tree);
	stats.cpu_time_rec=gettime_ms()-time;
	printf("===> CPU #2 time to compute descendants recursively = %.2f ms.\n",gettime_ms()-time);

	validateArrays(tree.num_nodes, tree.descendantArray, tree.descendantArray_rec, "CPU descendants rec");

	//compute descendants on GPU 
	if (tree.num_nodes!=0) descendants_gpu(&tree, &stats);

	validateArrays(tree.num_nodes, tree.descendantArray, tree.descendantArray_gpu, "GPU #1 descendants");
	validateArrays(tree.num_nodes, tree.descendantArray, tree.descendantArray_gpu_np, "GPU #2 descendants np");
	validateArrays(tree.num_nodes, tree.descendantArray, tree.descendantArray_gpu_np_hier, "GPU #3 descendants np hier");
	
	if (tree.num_nodes < 200){
		FILE *file = fopen(filename, "w");
		tree_to_dot(tree, file);
	}

	FILE *file = fopen("stats_r.txt", "w+");
	print_stats_r(file, stats);
	fclose(file);
	
	return(0);
}
