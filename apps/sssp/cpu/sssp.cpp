#include "sssp.h"

using namespace std;

bool compareInt(int a, int b)
{
	if ( a <= b )	
		return true;
	else 
		return false;
}

int SSSP_CPU()
{
	Heap *myheap = new Heap();
	for (int i=0; i<noNodeTotal; ++i){
		graph.costArray[i] = INF;
	}

	//graph.costArray[ source ] = 0;
	myheap->insert(source, 0);
	
	int iteration = 1;

	while ( !myheap->empty() ){
		item it = myheap->remove();
		int curr = it.id;

		if ( INF == graph.costArray[ curr ] ){
			graph.costArray[curr] = it.key;
	
			/* For each neighbour of curr */
			int start = graph.vertexArray[curr];
			int end = graph.vertexArray[curr+1];
			for (int i=start; i<end; ++i){
				int nid = graph.edgeArray[i];	// nid is neighbour id
				if ( INF == graph.costArray[ nid ] ){
					int altCost = graph.costArray[curr] + graph.weightArray[i];
					myheap->insert(nid, altCost);
				}
			}
		}
		iteration++;
	}
	printf("Iteration number: %d\n", iteration);
	return 0;
}

int main()
{
	double time, end_time;
	
	time = gettime();

	//readInputDIMACS9();
	readInputDIMACS10();
	//readInputSLNDC();

	printf("Source node is: %d\n", source);
	end_time = gettime();
	printf("Read data:\t\t%lf\n",end_time-time);
	
	time = gettime();
	convertCSR();
	end_time = gettime();
	printf("AdjList to CSR:\t\t%lf\n",end_time-time);
	
	time = gettime();
	SSSP_CPU();
		
	end_time = gettime();
	printf("SSSP iteration:\t\t%lf\n",end_time-time);

	FILE *log = fopen("log/costs_cpu","w+");
	outputCost(log);
	fclose(log);
	//clear();
	return 0;

}

