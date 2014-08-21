LinAlg
======

A small, extensible matrix library for heterogeneous supercomputers

If you feel that you spend too much time adapting your algorithm to the
libraries typically found on supercomputers: LinAlg is a C++ library provides a
thin layer of abstraction for calls to BLAS, LAPACK, CUDA&copy;, MPI,
intel&copy; MKL, etc. written with the goal of increasing your productivity
without incurring overhead.

What it tries to be
-------------------

- A library that increases your productivity when writing algorithms for
  massively parallel and heterogeneous systems
- Fast and incurring virtually no overhead
- Small and easy to extend/adapt for your specific needs
- Suitable for use on top of the software ecosystem typically found on a
  supercomputer
- Expressive and clear instead of elegant and obscure

Example
-------

This example illustrates a few features:

- templated matrix type (support for all BLAS/LAPACK types)
- support for different computing engines
- support for MPI
- support for asynchronous execution
- some helper functions like reading from files and timing operations

```C++

// This code
//    - allocates a matrix on a GPU
//    - creates a vector of submatrices containing the diagonal
//      blocks
//    - reads the content of a file with string formatted filename
//      into the matrix on the GPU
//    - does two asynchronous multiplications on the subblocks,
//      one without prefactors and one with prefactors. Since the
//      matrices are stored on the GPU, CUBLAS is used automatically.
//    - asynchronously sends another block of the matrix via MPI
//    - measures the time the main thread waits for the asynchronous
//      multiplication and transfer to be finished

using namespace LinAlg;

Location engine = Location::GPU; // or Location::host, Location::MIC
std::vector<Dense<double>> blocks(5);
Dense<double> my_matrix;
Utilities::Timer wait_time("Time waited for asynchronous calls");


my_matrix.reallocate(100, 100, engine)

for (int start = 0, stop = 10; start < 100; start += 10, stop += 10) {
    blocks[i].clone_from(my_matrix(start, stop, start, stop));
}

Utilities::read_CSR(my_matrix, "somefile_rank%d.csr", rank);

auto gpu_stream = BLAS::xGEMM_async(blocks[1], blocks[2], blocks[4]);
BLAS::xGEMM_async(2, blocks[3], blocks[4], 0.5, blocks[2], gpu_stream);

auto mpi_stream = MPI::send_matrix_async(blocks[3], comm, 2, 0);

wait_time.tic();
gpu_stream.sync();
mpi_stream.sync();
wait_time.toc();  // prints "Time waited for asynchronous calls: x sec"
      

```

What it is *not*
----------------

- An adaptation of MATLAB&reg; for C++:\n
  as can be seen from the above code example, you still need to know your way
  with the above mentioned libraries, their use should just become less tedious
  and error prone

- A turn key solution:\n
  even for fairly mainstream codes, chances are you'll have to add to it to
  suit your needs. Doing so should be very simple, though.


Documentation
-------------

The automatically generated documentation can be found
[here](http://maurocalderara.github.io/LinAlg/doxygen/)
