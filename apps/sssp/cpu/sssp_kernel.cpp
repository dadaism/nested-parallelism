#include "usssp.h"

using namespace std;


bool compareInt(int a, int b)
{
	if ( a <= b )	
		return true;
	else 
		return false;
}

void usssp_cpu()
{
	char *bitmap = new char [noNodeTotal] ();
	char *update = new char [noNodeTotal] ();
	char *tmp = NULL;
	for (int i=0; i<noNodeTotal; ++i){
		graph.costArray[i] = INF;
	}

	graph.costArray[ source ] = 0;
	bitmap[source] = 1;
	int iteration = 1;
	int stop = 0;
	while (!stop) {
		stop = 1;
		iteration++;
		#pragma omp parallel for 
		for (int i=0; i<noNodeTotal; ++i) {
			int curr = i;
			if ( bitmap[curr]==1 ) {
				bitmap[curr] = 0;

				/* For each neighbour of curr */
				int start = graph.vertexArray[curr];
				int end = graph.vertexArray[curr+1];
				for (int i=start; i<end; ++i){
					int nid = graph.edgeArray[i];	// nid is neighbour id
					int altCost = graph.costArray[curr] + graph.weightArray[i];
					if ( graph.costArray[nid] > altCost ){
						graph.costArray[nid] = altCost;
						update[nid] = 1; stop = 0;
					}
				}
			}
		}
		tmp = bitmap;
		bitmap = update;
		update = tmp;
	}
	delete [] bitmap;
	delete [] update;
}

