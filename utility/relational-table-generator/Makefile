GCC = g++
CC = nvcc
INC = -I./ -I../../../common

all: table-data-gen

table-data-gen: table_data_gen.cpp
	$(GCC) $(INC) $^ -o $@

clean:
	rm table-data-gen
