#include "np_synth.h"

#define N 10

using namespace std;

CONF config;

double init_time = 0;
double d_malloc_time = 0;
double h2d_memcpy_time = 0;
double ker_exe_time = 0;
double d2h_memcpy_time = 0;

int data_size;
int *iter;
FLOAT_T *rst;
FLOAT_T *data;
int VERBOSE;
int DEBUG;
FILE *fp = NULL;

double gettime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}

void usage() {
	fprintf(stderr,"\n");
	fprintf(stderr,"Usage:  gpu-ir-synth [option]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "    --help,-h      print this message\n");
	fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
	fprintf(stderr, "    --debug,-d     enhanced verbosity level\n");
	fprintf(stderr, "\nOther:\n");
	fprintf(stderr, "    --import,-i <data_file>           import data file\n");
	fprintf(stderr, "    --solution,-s <number>             specify the solution\n");
	fprintf(stderr, "                 0 - baseline\n");
	fprintf(stderr, "                 1 - dual queue\n");
	fprintf(stderr, "                 2 - shared delayed buffer\n");
	fprintf(stderr, "                 3 - global delayed buffer\n");
	fprintf(stderr, "                 4 - multiple dp per block\n");
	fprintf(stderr, "                 5 - single dp per block\n");
	fprintf(stderr, "    --device,-e <number>               select the device\n");
}

void print_conf() { 
	fprintf(stderr, "\nCONFIGURATION:\n");
	if (config.data_file) {
		fprintf(stderr, "- Data file: %s\n", config.data_file);
 	}
	if (config.verbose && config.debug) fprintf(stderr, "- verbose mode\n");
 	if (config.debug) fprintf(stderr, "- debug mode\n");
}

void init_conf() {
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
	fscanf(fp, "%d", &data_size);
	iter = (int*)malloc( (data_size+1)*sizeof(int) );
	
	for (int i=0; i<data_size; ++i) {
		fscanf(fp, "%d", &iter[i]);		
	}
	
	rst = (FLOAT_T*)malloc( data_size*sizeof(FLOAT_T) );
	data = (FLOAT_T*)malloc( data_size*sizeof(FLOAT_T) );
}

void clean()
{
	free(iter);
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

	double start_time, end_time;
	
	start_time = gettime();
	import_data();
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "Import graph:\t\t%lf\n",end_time-start_time);

	/* NP-SYNTH on GPU */
	for (int i=0; i<N; ++i)
		NP_SYNTH_GPU();

	if (VERBOSE) {
		fprintf(stdout, "CUDA runtime initialization:\t\t%lf\n", init_time/N);
		fprintf(stdout, "CUDA cudaNalloc:\t\t%lf\n", d_malloc_time/N);
		fprintf(stdout, "CUDA H2D cudaNemcpy:\t\t%lf\n", h2d_memcpy_time/N);
		fprintf(stdout, "CUDA kernel execution:\t\t%lf\n", ker_exe_time/N);
		fprintf(stdout, "CUDA D2H cudaNemcpy:\t\t%lf\n", d2h_memcpy_time/N);
	}

	if (DEBUG) {
		for (int i=0; i<data_size; ++i) {
			fprintf(stderr, "rst[%d] = %f\n", i, rst[i]);
		}
	}
	return 0;
}
