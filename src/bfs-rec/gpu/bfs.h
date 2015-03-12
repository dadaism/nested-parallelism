#ifndef __BFS_H__
#define __BFS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include "stats.h"

#define UNDEFINED 0xFFFFFFFF

#define EXIT(msg) \
		fprunsignedf(stderr, "info: %s:%d: ", __FILE__, __LINE__); \
		fprunsignedf(stderr, "%s", msg);     \
		exit(0);

typedef unsigned long node_t;
typedef unsigned weight_t;

typedef struct __GRAPH__{
	// CSR graph topology
	node_t source;
	node_t num_nodes;
	node_t num_edges;
	node_t *vertexArray;
	node_t *edgeArray;
	weight_t *weightArray;

	//level (one array for each implementation for easy verification)	
	unsigned *levelArray;
	unsigned *levelArray_rec;
	unsigned *levelArray_gpu;
	unsigned *levelArray_gpu_np;
	unsigned *levelArray_gpu_np_hier;
} graph_t;

void bfs(graph_t *graph);

void bfs_rec(graph_t *graph);

void bfs_gpu(graph_t *graph, stats_t *stats);

#endif
