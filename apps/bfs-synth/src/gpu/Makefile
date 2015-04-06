CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 
INC = -I./ -I../../../../common

all: global.o gpu-bfs-synth

global.o: ../../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

#myqueue.o: ../../../common/myqueue.cpp
#	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

#con_queue.o: ../../../common/con_queue.cpp
#	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

#bfs: myqueue.o con_queue.o global.o bfs.cpp bfs_wrapper.cu
#	$(CC) $(CC_FLAGS) $(INC) $^ -o bfs

gpu-bfs-synth: global.o bfs_synth.cpp bfs_synth.h bfs_synth_wrapper.cu bfs_synth_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o bfs_synth_wrapper.cu bfs_synth.cpp -o $@

clean:
	rm *.o gpu-bfs-synth
