CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 
INC = -I./ -I../../../common

all: global.o gpu-join

global.o: ../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

gpu-join: global.o join.cpp join.h join_wrapper.cu join_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o join_wrapper.cu join.cpp -o $@

clean:
	rm *.o gpu-join
