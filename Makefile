# System specific configuration is in make.inc
include make.inc

# General parts:
SOURCES = $(wildcard src/*.cc) $(wildcard src/*/*.cc)
OBJECTS = $(patsubst src/%.cc, obj/%.o, $(SOURCES))
STATIC  = lib/liblinalg.a
DYNAMIC = lib/linlinalg.so

.PHONY: clean doc

# Main targets
all: $(STATIC) $(DYNAMIC)

doc:
	doxygen doc/doxygen.conf

$(STATIC): $(OBJECTS)
	ar rcs lib/liblinalg.a \
	    $(OBJECTS)

$(DYNAMIC): $(OBJECTS)
	$(CXX) -shared -o lib/liblinalg.so \
	    $(OBJECTS) \
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

clean:
	rm -rf obj/*.o obj/*/*.o lib/liblinalg.a lib/liblinalg.so doc/html/doxygen/* doc/latex/*
