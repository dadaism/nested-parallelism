CC = g++
FLAGS = -fopenmp -O
INC = -I./ -I../../../../common

all: global.o cpu-pg

global.o: ../../../../common/global.cpp
	$(CC) $(FLAGS) $(INC) -c $^ -o $@

cpu-pg: global.o pagerank.cpp pagerank.h pagerank_kernel.cpp
	$(CC) $(FLAGS) $(INC) global.o pagerank_kernel.cpp pagerank.cpp -o $@

clean:
	rm *.o cpu-pg
