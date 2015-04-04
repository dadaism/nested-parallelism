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
	list<int> mylist;	// ordered list
	int *previous = new int [ noNodeTotal ];
	for (int i=0; i<noNodeTotal; ++i){
		graph.costArray[i] = INF;
		previous[i] = -1;
		mylist.push_back(i);	// In list, there are node id
								// So using "remove" is safe
	}

	graph.costArray[ source ] = 0;
	mylist.remove( source );
	mylist.push_front( source );
	int iteration = 1;
	while ( !mylist.empty() ){
		int curr = mylist.front();
		if ( INF == graph.costArray[ curr ] ){
			break;
		}
		mylist.pop_front();
		//if (iteration==40000)
		//	break;
		//printf("Commit node: %d\n", curr+1);
		//if ( iteration%1000==0 )
		//	printf("List size %d\n", mylist.size() );

		/* For each neighbour of curr */
		int start = graph.vertexArray[curr];
		int end = graph.vertexArray[curr+1];
		for (int i=start; i<end; ++i){
			int nid = graph.edgeArray[i];	// nid is neighbour id
			int altCost = graph.costArray[curr] + graph.weightArray[i];
			if ( altCost < graph.costArray[nid] ){
				//printf("From %d to %d, altCost: %d\n", curr+1, nid+1, altCost);
				graph.costArray[nid] = altCost;
				previous[nid] = curr;
				/* reorder nid in list, complexity: ( O(n^2) ) */ 
				list<int>::iterator it = mylist.begin();
				while( graph.costArray[*it] < altCost )
					++it;
				if ( *it!=nid ){
					mylist.remove(nid);
					mylist.insert(it, nid);
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
	//readInputDIMACS10();
	readInputSLNDC();
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

	outputCost();
	//clear();
	return 0;

}

