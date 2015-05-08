#ifndef __STATS_H__
#define __STATS_H__

#include <stdio.h>

typedef struct __STATS__ {
	unsigned streams;
	unsigned long long num_nodes;
	unsigned long long num_edges;
	unsigned levels;
	double cpu_time_it;
	double cpu_time_rec;
	double gpu_time;
	double gpu_time_np;
	double gpu_time_np_hier;
	//unsigned gpu_np_calls;
	//unsigned gpu_np_hier_calls;

} stats_t;

inline void print_stats(FILE *file, stats_t stats){
	fprintf(file, "%u\t%llu\t%llu\t%u\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",stats.streams, stats.num_nodes, stats.num_edges, stats.levels, stats.cpu_time_it, stats.cpu_time_rec, stats.gpu_time, stats.gpu_time_np, stats.gpu_time_np_hier);
}

#endif
