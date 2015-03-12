#include "heap.h"

Heap::Heap()
{
	N = 102400000;
	n = 0;
	bitmask = new char [N];
	for (int i=0; i<N; ++i)
		bitmask[i] = 0;
}


Heap::~Heap()
{
}

int Heap::size()
{
	return n;
}

bool Heap::empty()
{
	if ( 0==n )
		return true;
	return false;
}

void Heap::insert(int id, int key)
{
	item tmp;
	if ( n==N-1)
	{
		printf("heap overflow!\n");
		exit(0);
	}
	tmp.id = id; tmp.key = key;
	bitmask[id] = 1;
	heap.push_back(tmp);
	siftup(heap.size() - 1);
	n++;
};

item Heap::peek()
{
	item min = heap.front();
	return min;
}

item Heap::remove()
{
	item min = heap.front();
	heap[0] = heap.at(heap.size() - 1);	
	heap.pop_back();
	siftdown(0);
	n--;
	bitmask[min.id] = 0;
	return min;
}
bool Heap::member(int id)
{
	if ( bitmask[id]==1 )
		return true;
	return false;
}

void Heap::changeKey(int id, int key)
{
	for (int i=0; i<n; ++i)
	{
		item it = heap[i];
		if (heap[i].id==id){
			heap[i].key = heap[i].key>key? key : heap[i].key;
			siftup(i);
			break;
		}
	}
}

/* sift element in index to restore heap order */
void Heap::siftup(int index)
{
	while ( (index>0) && ( getParent(index)>=0 ) &&
			heap[getParent(index)].key > heap[index].key )
	{
		item tmp = heap[getParent(index)];
		heap[getParent(index)] = heap[index];
		heap[index] = tmp; 
		index = getParent(index);
	}
}

void Heap::siftdown(int index)
{
	int child = getLeft(index);
	int right = getRight(index);
	if ( (child>0) && (right>0) &&
		 ( heap[child].key > heap[right].key ) )
	{
		child = right;
	}
	if (child>0 && heap[child].key<heap[index].key)
	{
		item tmp = heap[child];
		heap[child] = heap[index];
		heap[index] = tmp; 
		siftdown(child);
	}
}

int Heap::getParent(int child)
{
	if ( 0!=child )
	{
		int i = (child-1) >> 1;
		return i;
	}
	return -1;
}

int Heap::getLeft(int parent)
{
	int i = ( parent<<1 ) + 1;
	return (i<heap.size())?i:-1;
}

int Heap::getRight(int parent)
{
	int i = ( parent<<1 ) + 2;
	return (i<heap.size())?i:-1;

}

void Heap::print()
{
	printf("Heap = ");
	for (int i=0; i<n; ++i)
	{
		printf("Id:%d Key:%d\t", heap[i].id, heap[i].key);
	}
	printf("\n");
}













