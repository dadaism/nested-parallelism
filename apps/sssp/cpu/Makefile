CC = g++
FLAGS = -fopenmp -O

COMMON = ../../../common
INC = -I./ -I../../../common

all:	global.o cpu-sssp cpu-usssp

global.o: ../../../common/global.cpp
	${CC} ${FLAGS} ${INC} -c $^ -o $@
cpu-sssp:	sssp.cpp sssp.h  $(COMMON)/heap.cpp  $(COMMON)/heap.h
	${CC} ${FLAGS} $(INC) global.o sssp.cpp sssp.h  $(COMMON)/heap.cpp -o $@
cpu-usssp:	usssp.cpp usssp.h sssp_wrapper.cpp sssp_kernel.cpp
	${CC} ${FLAGS} $(INC) global.o sssp_wrapper.cpp usssp.cpp usssp.h -o $@
clean:
	rm *.o cpu-sssp cpu-usssp
