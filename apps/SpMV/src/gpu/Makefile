CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 
INC = -I./ -I../../../../common

all: global.o gpu-SpMV

global.o: ../../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

gpu-SpMV: global.o SpMV.cpp SpMV.h SpMV_wrapper.cu SpMV_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o SpMV_wrapper.cu SpMV.cpp -o $@

clean:
	rm *.o gpu-SpMV
