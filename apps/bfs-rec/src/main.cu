#include <stdio.h>
#include "bfs.h"
#include "util.h"
#include "stats.h"
#include "cuda_util.h"
#include "graph_util.h"

#define ENABLE_CPU_RECURSIVE

static stats_t stats;

CONF config;

void usage() {
	fprintf(stderr,"\n");
	fprintf(stderr,"Usage:  gpu-bfs-rec [option]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "    --help,-h      print this message\n");
	fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
	fprintf(stderr, "    --debug,-d     enhanced verbosity level\n");
	fprintf(stderr, "\nOther:\n");
	fprintf(stderr, "    --import,-i <graph_file>           import graph file\n");
	fprintf(stderr, "    --format,-f <number>               specify the input format\n");
 	fprintf(stderr, "                 0 - DIMACS9\n");
	fprintf(stderr, "                 1 - DIMACS10\n");
	fprintf(stderr, "                 2 - SLNDC\n");
	fprintf(stderr, "    --solution,-s <number>             specify the solution\n");
	fprintf(stderr, "                 0 - thread pull (nopruning)\n");
	fprintf(stderr, "                 1 - dual queue\n");
	fprintf(stderr, "                 2 - shared delayed buffer\n");
	fprintf(stderr, "                 3 - global delayed buffer\n");
	fprintf(stderr, "                 4 - multi-dp launch per block\n");
	fprintf(stderr, "                 5 - single dp per block\n");
	fprintf(stderr, "                 6 - workload consolidation for dynamic parallelism\n");
	fprintf(stderr, "    --device,-e <number>               select the device\n");
}

void print_conf() { 
	fprintf(stderr, "\nCONFIGURATION:\n");
	if (config.graph_file) {
		fprintf(stderr, "- Graph file: %s\n", config.graph_file);
		fprintf(stderr, "- Graph format: %d\n", config.data_set_format);
 	}
	fprintf(stderr, "- Solution: %d\n", config.solution);
	if (config.verbose && config.debug) fprintf(stderr, "- verbose mode\n");
 	if (config.debug) fprintf(stderr, "- debug mode\n");
}

void init_conf() {
	config.data_set_format = 0;
	config.solution = 0;
	config.device_num = 0;
	config.verbose = false;
	config.debug = false;
}

int parse_arguments(int argc, char** argv) {
	int i = 1;
	if ( argc<2 ) {
		usage();
		return 0;
	}
	while ( i<argc ) {
/*		if ( strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0 ) {
			usage();
			return 0;
		}
		else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) {
			VERBOSE = config.verbose = 1;
		}
		else if ( strcmp(argv[i], "-d")==0 || strcmp(argv[i], "--debug")==0 ) {
			DEBUG = config.debug = 1;
		}
		else if ( strcmp(argv[i], "-f")==0 || strcmp(argv[i], "--format")==0 ) {
			++i;
			config.data_set_format = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-s")==0 || strcmp(argv[i], "--solution")==0 ) {
			++i;
			config.solution = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-e")==0 || strcmp(argv[i], "--device")==0 ) {
			++i;
			config.device_num = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-i")==0 || strcmp(argv[i], "--import")==0 ) {
			++i;
			if (i==argc) {
				fprintf(stderr, "Graph file name missing.\n");
			}
			config.graph_file = argv[i];
		}
		++i;*/
	}

	return 1;
}

void validateArrays(node_t n, unsigned int *array1, unsigned int *array2, const char *message){
	for (node_t node=0; node<n;node++){
		if (array1[node]!=array2[node]){
			printf("ERROR: validation error at %llu: %s !\n", node, message);
			break;
		}
	}
}

unsigned get_num_levels(graph_t graph){
	unsigned level = 0;
	for (node_t n=0; n < graph.num_nodes; n++){
		if (graph.levelArray[n]!=UNDEFINED) level = max(level,graph.levelArray[n]);
	}
	return (level+1);
}

int main(int argc, char *argv[])
{
	//init_device(DEVICE);
	double time;
	graph_t graph;
	char *filename = "test/graph.dot"; //unused at the moment...	
	unsigned dataset_num = 0;

	//process the input parameters
	if ( argc==2 ) {
		dataset_num = atoi(argv[1]);
	}
	else {
		printf("Usage: run_bfs [dataset] < [path to graph data file]\n");
		printf("dataset: 0 - DIMACS9\n");
		printf("         1 - DIMACS10\n");
		printf("         2 - SLNDC\n");
		exit(0);
	}

	//read graph datafile and convert it to CSR
	switch(dataset_num){
		case 0: readInputDIMACS9(&graph); break;
		case 1: readInputDIMACS10(&graph); break;
		case 2: readInputSLNDC(&graph); break;
		default: printf("Wrong code for dataset\n"); break;
	}

	//starts execution
	printf("\n===MAIN=== :: [num_nodes,num_edges] = %u, %u\n", graph.num_nodes, graph.num_edges);

	stats.num_nodes = graph.num_nodes;
	stats.num_edges = graph.num_edges;

	//compute bfs on CPU iteratively
	time = gettime_ms();
	if (graph.num_nodes!=0) bfs(&graph);
	stats.cpu_time_it=gettime_ms()-time;
	printf("===> CPU #1 time to compute bfs = %.2f ms.\n",gettime_ms()-time);

	stats.levels = get_num_levels(graph);

#ifdef ENABLE_CPU_RECURSIVE
	//compute bfs on CPU recursively
	time = gettime_ms();
	if (graph.num_nodes!=0) bfs_rec(&graph);
	stats.cpu_time_rec=gettime_ms()-time;
	printf("===> CPU #2 time to compute bfs recursively = %.2f ms.\n",gettime_ms()-time);

	validateArrays(graph.num_nodes, graph.levelArray, graph.levelArray_rec, "CPU bfs rec");
#endif

	//compute bfs on GPU 
	if (graph.num_nodes!=0) bfs_gpu(&graph, &stats);

	validateArrays(graph.num_nodes, graph.levelArray, graph.levelArray_gpu, "GPU #1 bfs");
	validateArrays(graph.num_nodes, graph.levelArray, graph.levelArray_gpu_np, "GPU #2 bfs np");
	validateArrays(graph.num_nodes, graph.levelArray, graph.levelArray_gpu_np_hier, "GPU #3 bfs np hier");

	//write stats file
	FILE *file = fopen("stats.txt", "w+");
	print_stats(file, stats);
	fclose(file);
	
	delete_graph(&graph);

	return(0);
}
