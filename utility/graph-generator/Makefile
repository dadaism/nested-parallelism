CC = g++

all: graph-gen

graph-gen: data_gen.cpp
	$(CC) ${INC} $^ -o $@

clean:
	rm *.o graph-gen
