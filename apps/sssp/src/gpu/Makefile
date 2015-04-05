CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2

#CC_FLAGS += --ptxas-options=-v

#CC_FLAGS += -G -g

#CC_FLAGS += -DDEBUG

#CC_FLAGS += -DGPU_PROFILE 

#CC_FLAGS += -DCPU_PROFILE

#CC_FLAGS += -DTHREASHOLD=400

INC = -I./ -I../../../../common

all: global.o gpu-sssp

global.o: ../../../../common/global.cpp
	${CC} ${FLAGS} ${INC} -c $^ -o $@
gpu-sssp: global.o sssp.cpp sssp_wrapper.cu sssp_kernel.cu
	$(CC) -D'CONSOLIDATE_LEVEL=1' $(CC_FLAGS) ${INC} global.o sssp.cpp sssp_wrapper.cu -o $@

clean:
	rm *.o gpu-sssp*
