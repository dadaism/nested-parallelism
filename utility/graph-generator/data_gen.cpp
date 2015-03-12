#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <set>

typedef struct conf {
    bool verbose;
	int low;
	int high;
    int dist;
	long num;
	float gamma;
    char *data_file;
} CONF;

CONF config;
bool VERBOSE = false;
FILE *fp = NULL;

double gettime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int gettime_usec() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec*1e+6+t.tv_usec;
}

void usage() {
    fprintf(stderr,"\n");
    fprintf(stderr,"Usage:  np-data-gen [option]\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "    --help,-h      print this message\n");
    fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
    fprintf(stderr, "\nOther:\n");
    fprintf(stderr, "    --export,-e <data_file>           export data file\n");
    fprintf(stderr, "    --low,-l <number>             specify the minimum value (default: 10)\n");
    fprintf(stderr, "    --high,-m <number>             specify the maximum value (default: 1000)\n");
    fprintf(stderr, "    --num,-n <number>             specify the number of data points (default: 10)\n");
    fprintf(stderr, "    --gamma,-g <number>            specify the gamma value for power law dist (default: 0.2)\n");
    fprintf(stderr, "    --dist,-d <number>             specify the distribution (default: 0)\n");
    fprintf(stderr, "                0 - unified distribution\n");
    fprintf(stderr, "                1 - power law distribution\n");
}

void print_conf() { 
    fprintf(stderr, "\nCONFIGURATION:\n");
    if (config.data_file) {
        fprintf(stderr, "- Data file: %s\n", config.data_file);
    }   
    if (config.verbose) fprintf(stderr, "- verbose mode\n");
}

void init_conf() {
	config.num = 10;
    config.low = 10;
    config.high = 1000;
	config.dist = 0;
	config.gamma = 0.2;
    config.verbose = false;
	config.data_file = NULL;
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
        else if ( strcmp(argv[i], "-l")==0 || strcmp(argv[i], "--low")==0 ) { 
            ++i;
            config.low = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-m")==0 || strcmp(argv[i], "--high")==0 ) { 
            ++i;
            config.high = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) { 
            VERBOSE = config.verbose = 1;
        }   
        else if ( strcmp(argv[i], "-d")==0 || strcmp(argv[i], "--dist")==0 ) { 
            ++i;
            config.dist = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-n")==0 || strcmp(argv[i], "--num")==0 ) { 
            ++i;
            config.num = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-g")==0 || strcmp(argv[i], "--gamma")==0 ) { 
            ++i;
            config.gamma = atof(argv[i]);
        }   
        else if ( strcmp(argv[i], "-e")==0 || strcmp(argv[i], "--export")==0 ) { 
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

inline int unified_dist(int l, int h) {
	if (l>h)
		return -1;
	else if (l==h)
		return h;
	else {
		srand( gettime_usec() );
		return rand() % (h-l) + l;
	}
}

inline int power_law(int l, int h, float k) {
	if (l>h)
		return -1;
	else if (l==h)
		return h;
	else {
		srand( gettime_usec() );
		double value = 	( pow(h, k+1) - pow(l, k+1) ) * ( (double)rand() / RAND_MAX) + pow(l, k+1);
		return pow(value, 1.0/(k+1));
	}
}

void gen_graph_data() {
	int y = 0;
	long total_neighbors = 0;
	int *neighbor_array = (int *)malloc(config.num*sizeof(int));
	
	int *shuffle = (int *)malloc(config.num*sizeof(int));	
	for (int i=0; i<config.num; ++i)
		shuffle[i] = i;

	for (int i=0; i<config.num; ++i) {
		int num_neighbors = 0;
		switch ( config.dist ) {
			case 0: num_neighbors = unified_dist( config.low, config.high ); 
				break;
			case 1: num_neighbors = power_law( config.low, config.high, -config.gamma );
				break;
			default:
				break;
		}
		if ( num_neighbors>config.num ) {
			num_neighbors = config.num;
		}
		neighbor_array[i] = num_neighbors;
		total_neighbors += num_neighbors;
	}

	fprintf(fp, "# Directed graph (each unordered pair of nodes is saved once): %s\n", config.data_file );
	fprintf(fp, "# Directed scale-free graph network from Oct 5 2014\n" );
	fprintf(fp, "# Nodes: %d Edges: %d\n", config.num, total_neighbors);
	fprintf(fp, "# FromNodeId    ToNodeId\n");
	for (int i=0; i<config.num; ++i) {
		int num_neighbors = neighbor_array[i];	
		// Method 1, shuffle num_neighbor times and pick up the first num_neighbors
/*		for (int j=0; j<num_neighbors; ++j) {

		}
*/
		// Method 2, set (very slow)	
		std::set<int> s;
		std::set<int>::iterator it;
		//fprintf(stdout, "Generate %d neighbors\n", num_neighbors);
		for (int j=0; j<num_neighbors; ++j) {
			int node_id = 0;
			s.insert(j);
			do {
				node_id = unified_dist( 0, config.num );
				it = s.find(node_id);
			} while (it!=s.end());
			fprintf(fp, "%d %d\n", i, node_id );
			s.insert(node_id);
		}
		s.clear();
	}
	fprintf(fp, "\n");
}

int main(int argc, char *argv[])
{
	init_conf();
    if ( !parse_arguments(argc, argv) ) return 0;
    
	if (VERBOSE)
    	print_conf();
    
    if ( config.data_file!=NULL ) {
        fp = fopen(config.data_file, "w");
        if ( fp==NULL ) { 
            fprintf(stderr, "Error: NULL file pointer.\n");
            return 1;
        }   
    }   
    else
		fp = stdout;

    double start_time, end_time;
    
    start_time = gettime();
    
    gen_graph_data();
    
    end_time = gettime();
    if (VERBOSE)
        fprintf(stderr, "Import graph:\t\t%lf\n",end_time-start_time);

    return 0;
}
