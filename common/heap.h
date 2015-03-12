#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iterator>

using namespace std;

typedef struct _ITEM_{
	int id;
	int key;
}item;

class Heap{
public:
	Heap();
	~Heap();
	void insert(int id, int key);
	item peek();	
	item remove();
	bool member(int id);
	void changeKey(int id, int key);
	bool empty();
	int size();
	void siftup(int index);
	void siftdown(int index);
	int getParent(int child);
	int getLeft(int parent);
	int getRight(int parent);
	void print();
private:
	int N;
	int n;
	char *bitmask;
	vector<item> heap;
};
