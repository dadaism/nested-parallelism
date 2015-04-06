#include "bc.h"
#include "bc_kernel.cpp"

using namespace std;

struct conf {
	bool verbose;
	bool debug;
	int data_set_format;
	int num_thread;
	char *graph_file;
} config;

void usage() {
	fprintf(stderr,"\n");
	fprintf(stderr,"Usage: cpu-bc [options]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "    --help,-h		print this message\n");
	fprintf(stderr, "    --verbose,-v	basic verbosity level\n");
	fprintf(stderr, "    --debug,-d		enhanced verbosity level\n");
	fprintf(stderr, "\nOther:\n");
	fprintf(stderr, "    --import,-i <graph_file>           import graph file\n");
	fprintf(stderr, "    --thread,-t <number of threads>    specify number of threads\n");
	fprintf(stderr, "    --format,-f <number>               specify the input format\n");
	fprintf(stderr, "                 0 - DIMACS9\n");
	fprintf(stderr, "                 1 - DIMACS10\n");
	fprintf(stderr, "                 2 - SLNDC\n");
}

void print_conf() {
	fprintf(stderr, "\nCONFIGURATION:\n");
	if (config.graph_file) {
		fprintf(stderr, "- Graph file: %s\n", config.graph_file);
		fprintf(stderr, "- Graph format: %d\n", config.data_set_format);
	}
	fprintf(stderr, "- Number of threads: %d\n", config.num_thread);
	if (config.verbose && config.debug)	fprintf(stderr, "- verbose mode\n");
	if (config.debug) fprintf(stderr, "- debug mode\n");
}

void init_conf() {
	config.data_set_format = 0;
	config.num_thread = 1;
	config.verbose = false;
	config.debug = false;
	config.graph_file = NULL;
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
		else if ( strcmp(argv[i], "-t")==0 || strcmp(argv[i], "--thread")==0 ) {
			++i;
			config.num_thread = atoi(argv[i]);
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

void output_bc()
{
	for (int i=0; i<noNodeTotal; ++i) {
		fprintf(stderr, "%d: %.8f\n", i, bc[i]);
		//fprintf(stderr, "%d: %d\n", i, sigma[i]);
		//fprintf(stderr, "%d: %lf\n", i, delta[i]);
	}
}

int main(int argc, char** argv)
{
	init_conf();
	if ( !parse_arguments(argc, argv) )	return 0;
	
	print_conf();
	
	if (config.graph_file!=NULL) {
		fp = fopen(config.graph_file, "r");
		if ( fp==NULL )	{
			fprintf(stderr, "Error: NULL file pointer.\n");
			return 1;
		}
	}
	else
		return 0;
	
	omp_set_dynamic(0);
	omp_set_num_threads(config.num_thread);
	
	double time, end_time;
	
	time = gettime();
	switch(config.data_set_format) {
		case 0: readInputDIMACS9();	break;
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
		fprintf(stderr,"AdjList to CSR:\t\t%lf\n",end_time-time);
	
	time = gettime();
	bc_cpu();
		
	end_time = gettime();
	printf("Execution time:\t\t%lf\n",end_time-time);
	if (DEBUG)
		outputBC(stdout);
	delete [] bc;
	delete [] delta;
	delete [] sigma;
	delete [] p;

	clear();
	return 0;
}

void bc_cpu()
{
	double time, end_time;
	bc = new float [noNodeTotal];
	delta = new float [noNodeTotal];
	sigma = new int [noNodeTotal];
	p = new char [noNodeTotal*noNodeTotal];
	
	p_idx = new int [noNodeTotal+1];
	p_data = new int [noEdgeTotal];

	memset(bc, 0, sizeof(float)*noNodeTotal);
	#pragma omp parallel for
	for (int i=0; i<noNodeTotal; ++i) {
		if (VERBOSE) {
 			if (i%1000==0)	fprintf(stderr, "Processing node %d...\n", i);
		}
	
		/* forward BFS */
		int dist = 0;
		forward_bfs_init(i);
		while (cont) {
			cont = false;
			forward_bfs_bitmap(dist);
			bitmap_workset();
			dist++;
		}
		/* backward propagation */
		/* init backward propagation */
		backward_init();
		while (dist>1) {
			backward_propagation(dist);
			backward_sum(i, dist);
			dist--;
		}
	}

	return;
}
