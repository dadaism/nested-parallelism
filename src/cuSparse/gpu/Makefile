CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 
INC = -I./ -I../../../common

all: global.o cuSparse

global.o: ../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

cuSparse: global.o cuSparse_wrapper.cu cuSparse.cu cuSparse.h
	$(CC) $(CC_FLAGS) -lcusparse $(INC) global.o cuSparse_wrapper.cu cuSparse.cu -o $@

#gpu-SpMV: global.o SpMV.cpp SpMV.h SpMV_wrapper.cu SpMV_kernel.cu
#	$(CC) $(CC_FLAGS) $(INC) global.o SpMV_wrapper.cu SpMV.cpp -o $@

example:
	nvcc cusparse_example.cu -lcusparse -o example

clean:
	rm *.o cuSparse
