#include "analyzer.h"

using namespace std;

CONF config;

struct _INFO_ info;

void usage(char** argv) {
    fprintf(stderr,"\n");
    fprintf(stderr,"Usage:  %s [option]\n", argv[0]);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "    --help,-h      print this message\n");
    fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
    fprintf(stderr, "\nOther:\n");
    fprintf(stderr, "    --import,-i <graph_file>           import graph file\n");
    fprintf(stderr, "    --dist, -d                         output distribution\n");
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
    fprintf(stderr, "- Solution: %d\n", config.solution);
    if (config.verbose) fprintf(stderr, "- verbose mode\n");
    if (config.dist) fprintf(stderr, "- output distribution\n");
}

void init_conf() {
    config.data_set_format = 0;
    config.solution = 0;
    config.verbose = false;
    config.dist = false;
}

int parse_arguments(int argc, char** argv) {
	int i = 1;
	if ( argc<2 ) {
		usage(argv);
		return 0;
	}
	while ( i<argc ) { 
		if ( strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0 ) { 
			usage(argv);
            return 0;
        }   
        else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) { 
            VERBOSE = config.verbose = 1;
        }   
        else if ( strcmp(argv[i], "-d")==0 || strcmp(argv[i], "--dist")==0 ) { 
            DEBUG = config.dist = 1;
        }   
        else if ( strcmp(argv[i], "-f")==0 || strcmp(argv[i], "--format")==0 ) { 
            ++i;
            config.data_set_format = atoi(argv[i]);
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

void getStatInfo()
{
	// total number of nodes and edges
	info.noNodeTotal = noNodeTotal;
	info.noEdgeTotal = noEdgeTotal;

	int maxDegreeNodeId;
	int minDegreeNodeId;
	int *nodeDegree = new int [noNodeTotal];
	info.avgDegree = 0;
	info.minDegree = 999999;
	info.maxDegree = 0;
	/* get "avg" "min" "max" of degree */
	for (int i=0; i<noNodeTotal; ++i){
		int degree = 0;
		//printf("Size of node %d is %d\n", i, adjacencyNodeList[i].size() );
		while ( adjacencyNodeList[i].empty()!=true &&
				adjacencyWeightList[i].empty()!=true){
			//adjacencyNodeList[i].back();
			//adjacencyWeightList[i].back();
			degree++;
			adjacencyNodeList[i].pop_back();
			adjacencyWeightList[i].pop_back();
		}
		//if (degree==0){
		//	printf("For node %d, degree is %d\n", i, degree);
		//}
		nodeDegree[i] = degree;
		info.totalDegree += degree;
		if ( degree>info.maxDegree ){
			info.maxDegree = degree;
			maxDegreeNodeId = i;		
		}
		if ( degree<info.minDegree ){
			info.minDegree = degree;
			minDegreeNodeId = i;		
		}
	}
	info.avgDegree = (float)info.totalDegree / info.noNodeTotal;
	printf("Node:%d\n", info.noNodeTotal);
	printf("Edge by degree:%ld\n", info.totalDegree/2);
	printf("Edge:%d\n", info.noEdgeTotal);
	printf("Avg:\t%f\nMax:\t%d\nMin:\t%d\n",info.avgDegree, info.maxDegree, info.minDegree);
	printf("Node ID with max degree:%d\n", maxDegreeNodeId);
	printf("Node ID with min degree:%d\n", minDegreeNodeId);

	if ( config.dist ) {
		/* get distribution of degree */
		info.distDegree = new long [ info.maxDegree + 1];
		for (int i=info.minDegree; i<=info.maxDegree; ++i)
			info.distDegree[i] = 0;

		for (int i=0; i<noNodeTotal; ++i){
			info.distDegree[ nodeDegree[i] ]++;
		}
	
		/* print distribution of degree */
		for (int i=info.minDegree; i<=info.maxDegree; ++i){
			if ( info.distDegree[i]!=0 )
				printf("%d %d\n", i, info.distDegree[i]);
		}
		delete [] nodeDegree;
	}
}

int main(int argc, char** argv)
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

    switch(config.data_set_format) {
        case 0: readInputDIMACS9(); break;
        case 1: readInputDIMACS10(); break;
        case 2: readInputSLNDC(); break;
        default: fprintf(stderr, "Wrong code for dataset\n"); break;
    }

	getStatInfo();
	return 0;
}

