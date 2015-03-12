CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 
INC = -I./ -I../../../common

all: global.o gpu-bc

global.o: ../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

gpu-bc: global.o bc.cpp bc.h bc_wrapper.cu bc_kernel.cu bc_fp_kernel.cu bc_bp_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o bc_wrapper.cu bc.cpp -o $@

clean:
	rm *.o gpu-bc
