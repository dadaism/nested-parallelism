CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 
INC = -I./ -I../../../common

all: gpu-np-synth

gpu-np-synth: np_synth.cpp np_synth.h np_synth_wrapper.cu np_synth_kernel.cu work_device.cu
	$(CC) $(CC_FLAGS) $(INC) np_synth_wrapper.cu np_synth.cpp -o $@

clean:
	rm gpu-np-synth
