#include "bc.h"

using namespace std;

CONF config;

void usage() {
	fprintf(stderr,"\n");
	fprintf(stderr,"Usage:  gpu-bc [option]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "    --help,-h      print this message\n");
	fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
	fprintf(stderr, "    --debug,-d     enhanced verbosity level\n");
	fprintf(stderr, "\nOther:\n");
	fprintf(stderr, "    --import,-i <graph_file>           import graph file\n");
	fprintf(stderr, "    --thread,-t <number of threads>    specify number of threads\n");
	fprintf(stderr, "    --format,-f <number>               specify the input format\n");
 	fprintf(stderr, "                 0 - DIMACS9\n");
	fprintf(stderr, "                 1 - DIMACS10\n");
	fprintf(stderr, "                 2 - SLNDC\n");
	fprintf(stderr, "    --solution,-s <number>             specify the solution\n");
	fprintf(stderr, "                 0 - thread bitmap\n");
	fprintf(stderr, "                 1 - thread queue\n");
	fprintf(stderr, "                 2 - dual queue\n");
	fprintf(stderr, "                 3 - shared delayed buffer\n");
	fprintf(stderr, "                 4 - global delayed buffer\n");
	fprintf(stderr, "                 5 - naive dp\n");
	fprintf(stderr, "                 6 - optimized dp\n");
	fprintf(stderr, "                 7 - workload consolidation for dynamic parallelism\n");
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
		++i;
	}
	
	return 1;
}

int main(int argc, char* argv[])
{
	init_conf();
	if ( !parse_arguments(argc, argv) ) return 0;
	
	print_conf();
	
	if (config.graph_file!=NULL) {
		fp = fopen(config.graph_file, "r");
		if ( fp==NULL ) {
			fprintf(stderr, "Error: NULL file pointer.\n");
			return 1;
		}
	}
	else
		return 0;

	double time, end_time;
	
	time = gettime();
	switch(config.data_set_format) {
		case 0: readInputDIMACS9(); break;
		case 1: readInputDIMACS10(); break;
		case 2: readInputSLNDC(); break;
		default: fprintf(stderr, "Wrong code for dataset\n"); break;
	}
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "Import graph:\t\t%lf\n",end_time-time);
	
	time = gettime();
	convertCSR();
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "AdjList to CSR:\t\t%lf\n",end_time-time);

	bc = new float [noNodeTotal];
	
	/* BC on GPU */
	BC_GPU();
	
	if (DEBUG)
		outputBC(stdout);
	
	delete [] bc;
	clear();
	return 0;
}
