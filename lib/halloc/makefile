PREFIX=~/Documents/Working/nested-parallelism
NAME=libhalloc.a
HEADER=halloc.h
SRC_C=*.cu
SRC_H=*.h *.cuh
SRC=$(SRC_C) $(SRC_H)
TGT=./$(NAME)
ARCH= -gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35
#TEST_TGT=tst/corr/bin/test

TMP=*~ \\\#* ./*~ ./\\\#* $(TGT) $(TEST_TGT)

# be careful: using cs modifier can lead to errors maxrregcount should be 44-64,
# this allows both enough threads and enough storage (values of 44 and 54 give
# good results in phase test, while 60 and 64 provide somewhat better spree
# throughput); 39-42 (39 tested) are good when operating in L1-preferred mode
build: $(TGT)
$(TGT):	$(SRC) makefile
	nvcc $(ARCH) -lineinfo -O3 -lib -rdc=true -Xptxas -dlcm=cg -Xptxas -dscm=wb \
	-Xptxas -maxrregcount=64 -o $(TGT) $(SRC_C)
#	-Xptxas -maxrregcount=42 -o $(TGT) $(SRC_C)
#	nvcc $(ARCH) -O3 -lib -rdc=true -Xptxas -dlcm=cs -Xptxas -dscm=cs -o $(TGT) $(SRC_C)
#	nvcc $(ARCH) -O3 -lib -rdc=true -o $(TGT) $(SRC_C)

clean:
	rm -f $(TMP)

install: $(HEADER) $(TGT)
	cp $(HEADER) $(PREFIX)/common/halloc.h
	cp $(TGT) $(PREFIX)/lib/libhalloc.a

uninstall:
	rm -f $(PREFIX)/include/halloc.h $(PREFIX)/lib/libhalloc.a
