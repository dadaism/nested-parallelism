#include "join.h"

#define N 10

using namespace std;

CONF config;
TABLE table1;
TABLE table2;

double start_time = 0;
double end_time = 0;
double init_time = 0;
double d_malloc_time = 0;
double h2d_memcpy_time = 0;
double ker_exe_time = 0;
double d2h_memcpy_time = 0;

void usage() {
	fprintf(stderr,"\n");
	fprintf(stderr,"Usage:  gpu-join [option]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "    --help,-h      print this message\n");
	fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
	fprintf(stderr, "    --debug,-d     enhanced verbosity level\n");
	fprintf(stderr, "\nOther:\n");
	fprintf(stderr, "    --import,-i <data file>           import graph file\n");
	fprintf(stderr, "    --solution,-s <number>             specify the solution\n");
	fprintf(stderr, "                 0 - baseline\n");
	fprintf(stderr, "                 1 - dual queue\n");
	fprintf(stderr, "                 2 - shared delayed buffer\n");
	fprintf(stderr, "                 3 - global delayed buffer\n");
	fprintf(stderr, "                 4 - multiple dp per block\n");
	fprintf(stderr, "                 5 - single dp per block\n");
	fprintf(stderr, "                 6 - workload consolidation\n");
	fprintf(stderr, "    --device,-e <number>               select the device\n");
}

void print_conf() { 
	fprintf(stderr, "\nCONFIGURATION:\n");
	if (config.data_file) {
		fprintf(stderr, "- Data file: %s\n", config.data_file);
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
		if ( strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0 ) {
			usage();
			return 0;
		}
		else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) {
			VERBOSE = config.verbose = 1;
		}
		else if ( strcmp(argv[i], "-d")==0 || strcmp(argv[i], "--debug")==0 ) {
			DEBUG = config.debug = 1;
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
				fprintf(stderr, "Data file name missing.\n");
			}
			config.data_file = argv[i];
		}
		++i;
	}
	
	return 1;
}

void import_data()
{
	fscanf(fp, "%d", &table1.size );
	for (int i=0; i<table1.size; ++i) {
		fscanf(fp, "%d %d\n", &table1.key[i], &table1.value[i]);
	}
	fscanf(fp, "%d", &table2.size );
	for (int i=0; i<table2.size; ++i) {
		fscanf(fp, "%d %d\n", &table2.key[i], &table2.value[i]);
	}
}

void clean()
{


}

int main(int argc, char* argv[])
{
	init_conf();
	if ( !parse_arguments(argc, argv) ) return 0;
	
	print_conf();
	
	if (config.data_file!=NULL) {
		fp = fopen(config.data_file, "r");
		if ( fp==NULL ) {
			fprintf(stderr, "Error: NULL file pointer.\n");
			return 1;
		}
	}
	else
		return 0;

	double time, end_time;
	
	time = gettime();
	import_data();
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "Import graph:\t\t%lf\n",end_time-time);
	
	/* SpMV on GPU */
	for (int i=0; i<N; ++i) {
		JOIN_GPU();
	}

	if (VERBOSE) {
		fprintf(stdout, "CUDA runtime initialization:\t\t%lf\n", init_time/N);
		fprintf(stdout, "CUDA cudaMalloc:\t\t%lf\n", d_malloc_time/N);
		fprintf(stdout, "CUDA H2D cudaMemcpy:\t\t%lf\n", h2d_memcpy_time/N);
		fprintf(stdout, "CUDA kernel execution:\t\t%lf\n", ker_exe_time/N);
		fprintf(stdout, "CUDA D2H cudaMemcpy:\t\t%lf\n", d2h_memcpy_time/N);
	}
	if (DEBUG) {
		for (int i=0; i<noNodeTotal; ++i) {
			//fprintf(stderr, "x[%d] = %f\n", i, x[i]);
			fprintf(stderr, "y[%d] = %f\n", i, y[i]);
		}
	}
	clear();
	return 0;
}
