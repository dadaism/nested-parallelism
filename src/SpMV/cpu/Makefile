CC = g++
FLAGS = -fopenmp -O
INC = -I./ -I../../../common

all: global.o cpu-SpMV

global.o: ../../../common/global.cpp
	$(CC) $(FLAGS) $(INC) -c $^ -o $@

cpu-SpMV: global.o SpMV.cpp SpMV.h SpMV_kernel.cpp
	$(CC) $(FLAGS) $(INC) global.o SpMV_kernel.cpp SpMV.cpp -o $@

clean:
	rm *.o cpu-SpMV
