CUDA_HOME   = /Soft/cuda/12.2.2/

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -gencode arch=compute_86,code=sm_86 --ptxas-options=-v -I$(CUDA_HOME)/lib
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib -I$(CUDA_HOME)/lib

PROG_FLAGS  = -DSIZE=16


MERGESORTEXE    = mergesort.exe

MERGESORTO    = mergesort.o

default: $(MERGESORTEXE)

mergesort.o: mergesort.cu
	        $(NVCC) -c -o $@ mergesort.cu $(NVCC_FLAGS) $(PROG_FLAGS)


$(MERGESORTEXE): $(MERGESORTO)
	        $(NVCC) $(MERGESORTO) -o $(MERGESORTEXE) $(LD_FLAGS)


all:    $(MERGESORTEXE)

clean:
	        rm -rf *.o mergesort*.exe submit* report*
