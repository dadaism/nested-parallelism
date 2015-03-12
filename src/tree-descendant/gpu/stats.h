#ifndef __STATS_H__
#define __STATS_H__

#include <stdio.h>

typedef struct __STATS__ {
	unsigned levels;
	unsigned outdegree;
	unsigned outdegree_min;
	unsigned outdegree_max;
	unsigned long long num_nodes;
	double cpu_time_it;
	double cpu_time_rec;
	double gpu_time;
	double gpu_time_np;
	double gpu_time_np_hier;
	//unsigned gpu_np_calls;
	//unsigned gpu_np_hier_calls;

} stats_t;

inline void print_stats(FILE *file, stats_t stats){
	fprintf(file, "%u\t%u\t%llu\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",stats.levels, stats.outdegree, stats.num_nodes, stats.cpu_time_it, stats.cpu_time_rec, stats.gpu_time, stats.gpu_time_np, stats.gpu_time_np_hier);
}

inline void print_stats_r(FILE *file, stats_t stats){
	fprintf(file, "%u\t%u\t%u\t%llu\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",stats.levels, stats.outdegree_min, stats.outdegree_max, stats.num_nodes, stats.cpu_time_it, stats.cpu_time_rec, stats.gpu_time, stats.gpu_time_np, stats.gpu_time_np_hier);
}

#endif
