# System specific configuration is in make.inc
include make.inc

# General parts:
SOURCES      = $(wildcard src/*.cc) $(wildcard src/*/*.cc)
OBJECTS      = $(patsubst src/%.cc, obj/%.o, $(SOURCES))
CUDA_SOURCES = $(wildcard src/*.cu) $(wildcard src/*/*.cu)
CUDA_OBJECTS = $(patsubst src/%.cu, obj/%.o, $(CUDA_SOURCES))
CUDA_DEVICE_OBJECTS = $(patsubst src/%.cu, obj/%.o_dlinked.o, $(CUDA_SOURCES))

STATIC       = lib/liblinalg.a
DYNAMIC      = lib/linlinalg.so

.PHONY: clean doc

# Main targets
#all: $(STATIC) $(DYNAMIC)
all: $(STATIC)

doc:
	doxygen doc/doxygen.conf

$(STATIC): $(OBJECTS) $(CUDA_OBJECTS)
	ar rcs lib/liblinalg.a \
	    $(OBJECTS) $(CUDA_OBJECTS) $(CUDA_DEVICE_OBJECTS)

$(DYNAMIC): $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) -shared -o lib/liblinalg.so \
	    $(OBJECTS) $(CUDA_OBJECTS) \
	    $(CXX_LINKER_FLAGS) \
	    $(MPI_LIBS) \
	    $(BLAS_LIBS) \
	    $(MKL_LIBS) \
	    $(CUDA_LIBS) \
	    $(MAGMA_LIBS) \
	    $(OTHER_LIBS)

$(OBJECTS): obj/%.o : src/%.cc
	$(CXX) -c $< -o $@ -I./include $(LINALG_FLAGS) \
	    $(CXX_FLAGS) \
	    $(MPI_FLAGS) $(MPI_INCLUDE) \
	    $(MKL_FLAGS) $(MKL_INCLUDE) \
	    $(CUDA_FLAGS) $(CUDA_INCLUDE) \
	    $(MAGMA_FLAGS) $(MAGMA_INCLUDE) \
	    $(OTHER_FLAGS) $(OTHER_INCLUDE)

$(CUDA_OBJECTS): obj/%.o : src/%.cu
	$(NVCC) --relocatable-device-code=true -c $< -o $@ -I./include $(LINALG_FLAGS) \
	    $(NVCC_FLAGS) \
	    $(CUDA_FLAGS) $(CUDA_INCLUDE)
	$(NVCC) --device-link $@ -o $@_dlinked.o $(NVCC_FLAGS) $(CUDA_FLAGS)

clean:
	rm -rf obj/*.o obj/*/*.o lib/liblinalg.a lib/liblinalg.so doc/html/doxygen/* doc/latex/*
