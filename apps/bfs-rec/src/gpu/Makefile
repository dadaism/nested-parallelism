CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20 
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v 
INC = -I./ -I../../../../common

all: global.o gpu-bfs-rec

global.o: ../../../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

#myqueue.o: ../../../common/myqueue.cpp
#	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

#con_queue.o: ../../../common/con_queue.cpp
#	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

#bfs: myqueue.o con_queue.o global.o bfs.cpp bfs_wrapper.cu
#	$(CC) $(CC_FLAGS) $(INC) $^ -o bfs

gpu-bfs-rec: global.o bfs_rec.cpp bfs_rec.h bfs_rec_wrapper.cu bfs_rec_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o bfs_rec_wrapper.cu bfs_rec.cpp -o $@

clean:
	rm *.o gpu-bfs-rec
