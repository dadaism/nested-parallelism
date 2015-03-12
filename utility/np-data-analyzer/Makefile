GCC = g++
INC = -I./ -I../../../common

all: np-data-analyzer

np-data-analyzer: data_analyzer.cpp
	$(GCC) $(INC) $^ -o $@

clean:
	rm np-data-analyzer
