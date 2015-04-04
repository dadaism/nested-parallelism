CC = /usr/bin/g++
NVCC = /usr/local/cuda/bin/nvcc
#CFLAGS = -g -pg -w -O4 -I..
CFLAGS = -O2 -I..
NVCCFLAGS = -Xcompiler -fopenmp -rdc=true -arch sm_35 -w -O2 -I.. 
LINK_OPT = -L/usr/local/cuda/lib64 -lcuda -lcudart

all:	main_0streams main_1streams main_2streams main_4streams main_8streams  

clean:
	-rm *.o run*

.c.o:
	${CC} ${CFLAGS} -c $*.c 

bfs.o : bfs.h

bfs_gpu0.o : bfs.h stats.h
	${NVCC} ${NVCCFLAGS} -D'STREAMS=0' ${LINK_OPT} -dc bfs_gpu.cu -o bfs_gpu0.o

bfs_gpu1.o : bfs.h stats.h
	${NVCC} ${NVCCFLAGS} -D'STREAMS=1' ${LINK_OPT} -dc bfs_gpu.cu -o bfs_gpu1.o

bfs_gpu2.o : bfs.h stats.h
	${NVCC} ${NVCCFLAGS} -D'STREAMS=2' ${LINK_OPT} -dc bfs_gpu.cu -o bfs_gpu2.o

bfs_gpu4.o : bfs.h stats.h
	${NVCC} ${NVCCFLAGS} -D'STREAMS=4' ${LINK_OPT} -dc bfs_gpu.cu -o bfs_gpu4.o

bfs_gpu8.o : bfs.h stats.h
	${NVCC} ${NVCCFLAGS} -D'STREAMS=8' ${LINK_OPT} -dc bfs_gpu.cu -o bfs_gpu8.o

main.o : bfs.h stats.h
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} -dc main.cu

main_irr.o : bfs.h stats.h
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} -dc main_irr.cu

main_0streams : bfs.o bfs_gpu0.o main.o 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} bfs.o bfs_gpu0.o main.o -o run_bfs_0streams

main_1streams : bfs.o bfs_gpu1.o main.o 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} bfs.o bfs_gpu1.o main.o -o run_bfs_1streams

main_2streams : bfs.o bfs_gpu2.o main.o 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} bfs.o bfs_gpu2.o main.o -o run_bfs_2streams 

main_4streams : bfs.o bfs_gpu4.o main.o 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} bfs.o bfs_gpu4.o main.o -o run_bfs_4streams 

main_8streams : bfs.o bfs_gpu8.o main.o 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} bfs.o bfs_gpu8.o main.o -o run_bfs_8streams 

exe:
	-chmod a+x run*
