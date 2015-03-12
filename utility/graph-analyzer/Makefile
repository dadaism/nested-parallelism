CC=g++

FLAGS= -O3

INC=-I./ -I../../common

all: global.o analyzer

global.o: ../../common/global.cpp
	${CC} ${FLAGS} ${INC} -c $^ -o $@

analyzer:	global.o analyzer.cpp analyzer.h
	${CC} ${FLAGS} ${INC} global.o analyzer.cpp -o $@

clean:
	rm *.o analyzer
