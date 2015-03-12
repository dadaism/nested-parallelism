#include "heap.h"

using namespace std;

void testHeap()
{
	Heap* myheap = new Heap();
	myheap->insert(1, 90);
	myheap->insert(1, 80);
	myheap->insert(1, 70);
	myheap->insert(1, 60);
	myheap->insert(1, 50);
	myheap->insert(1, 40);
	myheap->insert(1, 30);
	myheap->insert(2, 30);
	myheap->insert(2, 30);
	myheap->insert(2, 40);
	myheap->print();	
	printf("Extract min %d from heap\n\n", myheap->remove());
	myheap->print();
	printf("Insert 10\n");
	myheap->insert(1,10);
	myheap->print();
	printf("Extract min %d from heap\n\n", myheap->remove());
	myheap->print();
	printf("Extract min %d from heap\n\n", myheap->remove());
	myheap->print();
}

int main()
{
	testHeap();

	return 0;

}

