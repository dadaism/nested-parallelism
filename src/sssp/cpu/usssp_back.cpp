#include "usssp.h"

using namespace std;


bool compareInt(int a, int b)
{
	if ( a <= b )	
		return true;
	else 
		return false;
}

int USSSP_CPU()
{
	char *bitmap = new char [noNodeTotal] ();
	queue<int> myqueue;
	for (int i=0; i<noNodeTotal; ++i){
		graph.costArray[i] = INF;
	}

	graph.costArray[ source ] = 0;
	bitmap[source] = 1;
	myqueue.push(source);
	int iteration = 1;
	
	while ( !myqueue.empty() ){
		int curr = myqueue.front();
		myqueue.pop();
		bitmap[curr] = 0;

		/* For each neighbour of curr */
		int start = graph.vertexArray[curr];
		int end = graph.vertexArray[curr+1];
		for (int i=start; i<end; ++i){
			int nid = graph.edgeArray[i];	// nid is neighbour id
			int altCost = graph.costArray[curr] + graph.weightArray[i];
			if ( graph.costArray[nid] > altCost ){
				graph.costArray[nid] = altCost;
				if ( bitmap[nid]==0 ){
					bitmap[nid] = 1;
					myqueue.push(nid);
				}
			}
		}
	}
	delete [] bitmap;
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
	USSSP_CPU();
		
	end_time = gettime();
	printf("SSSP iteration:\t\t%lf\n",end_time-time);

	FILE *log = fopen("log/costs_cpu_unordered","w+");
	outputCost(log);
	fclose(log);
	//clear();
	return 0;

}

