#ifndef __TREE_H__
#define __TREE_H__

#include <stdio.h>
#include <stdlib.h>
#include "stats.h"

#define EXIT(msg) \
        fprunsignedf(stderr, "info: %s:%d: ", __FILE__, __LINE__); \
        fprunsignedf(stderr, "%s", msg);     \
        exit(0);

typedef unsigned long long node_t;

typedef struct _TREE__{
	node_t num_nodes;
	unsigned num_levels;
        node_t *vertexArray;
        node_t *parentArray;
        node_t *edgeArray;
	unsigned *levelArray;
	node_t *descendantArray;
	node_t *descendantArray_rec;
	node_t *descendantArray_gpu;
	node_t *descendantArray_gpu_np;
	node_t *descendantArray_gpu_np_hier;
} tree_t;

void gen_regular_tree(tree_t *tree, unsigned num_levels, unsigned outdegree);

void gen_random_tree(tree_t *tree, unsigned num_levels, unsigned outdegree_min, unsigned outdegree_max);

void tree_to_dot(tree_t tree, FILE *file);

void descendants(tree_t *tree);

void descendants_rec(tree_t *tree);

void descendants_gpu(tree_t *tree, stats_t *stats);

#endif
