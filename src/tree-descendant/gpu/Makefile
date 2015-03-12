CC = /usr/bin/g++
NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -g -pg -w -O4 -I.. 
NVCCFLAGS = -Xcompiler -fopenmp -arch sm_35 -g -w -O4 -I.. 
LINK_OPT = -L/usr/local/cuda/lib64 -lcuda -lcudart

all:	main main_irr

clean:
	-rm *.o run*

.c.o:
	${CC} ${CFLAGS} -c $*.c 

tree.o : tree.h

tree_gpu.o : tree.h stats.h
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} -dc tree_gpu.cu

main.o : tree.h stats.h
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} -dc main.cu

main_irr.o : tree.h stats.h
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} -dc main_irr.cu

main : tree.o tree_gpu.o main.o 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} tree.o tree_gpu.o main.o -o run_tree 

main_irr : tree.o tree_gpu.o main_irr.o 
	${NVCC} ${NVCCFLAGS} ${LINK_OPT} tree.o tree_gpu.o main_irr.o -o run_tree_irr

exe:
	-chmod a+x run*
