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
	if (argc < 5){
		printf ("usage: ./run_tree DEPTH_MIN, DEPTH_MAX, OUTDEGRE_MIN, OUTDEGREE_MAX, POSSIBILITY of descendants\n");
		return(-1);
	}
	unsigned level_min  = atoi(argv[1]);
        unsigned level_max  = atoi(argv[2]);
        unsigned outdegree_min  =  atoi(argv[3]);
        unsigned outdegree_max  =  atoi(argv[4]);
        char *filename = "test/tree.dot";
        unsigned possi=3;
        if(6==argc) possi = atoi(argv[5]);
	
        printf("\n===MAIN=== :: [depth_min,depth_max,outdegree_min,outdegree_max,possibility] = %u, %u, %u, %u, %u\n", level_min,level_max, outdegree_min, outdegree_max, possi);

        gen_random_tree(&tree, level_min, level_max, outdegree_min, outdegree_max, possi);

	printf("nodes=%llu\n",tree.num_nodes);

	stats.levels=level_max;
        stats.level_min=level_min;
        stats.level_max=level_max;
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
