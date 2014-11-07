# System specific configuration is in make.inc
include ../make.inc

# General parts:
SOURCES = $(wildcard test_*.cc)
TESTS = $(patsubst test_%.cc, test_%, $(SOURCES))

.PHONY: clean doc

# Main targets
all: $(TESTS)

$(TESTS): test_% : test_%.cc
	$(CXX) $< -o $@ -I../include $(LINALG_FLAGS) \
	    $(CXX_FLAGS) $(CXX_LINKER_FLAGS) $(BLAS_LIBS) \
	    $(MPI_FLAGS) $(MPI_INCLUDE) $(MPI_LIBS) \
	    $(MKL_FLAGS) $(MKL_INCLUDE) $(MKL_LIBS) \
	    $(CUDA_FLAGS) $(CUDA_INCLUDE) $(CUDA_LIBS) \
	    $(MAGMA_FLAGS) $(MAGMA_INCLUDE) $(MAGMA_LIBS) \
	    $(OTHER_FLAGS) $(OTHER_INCLUDE) $(OTHER_LIBS)

clean:
	rm -rf $(TESTS)
