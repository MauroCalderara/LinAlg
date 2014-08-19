LinAlg
======

A small, extensible matrix library for heterogeneous supercomputers

If you feel that you spend too much time adapting your algorithm to the
libraries typically found on supercomputers: LinAlg is a C++ library provides a
thin layer of abstraction for calls to BLAS, LAPACK, CUDA&copy;, MPI,
intel&copy; MKL, etc. written with the goal of increasing your productivity
without incurring any overhead.

What it tries to be
-------------------

- a library that boosts your productivity writing algorithms for massively
  parallel and heterogeneous systems
- fast and incurring virtually no overhead
- small and easy to extend/adapt for your specific needs
- suitable for use on top of the software ecosystem typically found on a
  supercomputer

What it is *not*
----------------

- an adaptation of MATLAB&copy; for C++: you still need to know your way with
  the above mentioned libraries, their use should just become less tedious and
  error prone
- a turn key solution: even for fairly mainstream codes, chances are you'll
  have to extend it to suit your needs. Doing so should be very simple, though.


