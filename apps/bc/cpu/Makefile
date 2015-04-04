CC = g++
FLAGS = -fopenmp -O

COMMON = ../../../common
INC = -I./ -I../../../common

all:	global.o cpu-bc

global.o: ../../../common/global.cpp
	${CC} ${FLAGS} $(INC) -c $^ -o $@
	
cpu-bc:	global.o bc.cpp bc.h bc_kernel.cpp
	${CC} ${FLAGS} $(INC) global.o bc.cpp -o $@
clean:
	rm *.o cpu-bc
