GCC = g++
CC = nvcc
INC = -I./ -I../../../common

all: np-data-gen

np-data-gen: data_gen.cpp
	$(GCC) $(INC) $^ -o $@

clean:
	rm np-data-gen
