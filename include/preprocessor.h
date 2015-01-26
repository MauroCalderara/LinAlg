/** \file             preprocessor.h
 *
 *  \brief            Preprocessor macros
 *
 *  \date             Created:  Nov  27, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          camauro <camauro@domain.tld>
 *
 *  \version          $Revision$
 */
#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

// TODO:
//
//  document all preprocessor macros and their effect here
//

// LINALG_NO_CHECKS
//
//    Disable all argument checking (also disables all exceptions)

// Add_ / NoChange / UpCase
//
//    Govern FORTRAN naming conventions (for BLAS/LAPACK). When using Magma,  
//    you have to use Add_

// HAVE_CUDA
//
//    Enable support for CUDA

// HAVE_MAGMA
//
//    Enable support for MAGMA
#ifdef HAVE_MAGMA
# ifdef NoChange
#   error "When setting -DHAVE_MAGMA you must set -DAdd_, not -DNoChange"
# elif UpCase
#   error "When setting -DHAVE_MAGMA you must set -DAdd_, not -DNoChange"
# endif
# ifndef Add_
#   define Add_
# endif
#endif

// HAVE_MKL
//
//    Enable support for intel MKL

// HAVE_MPI
//
//    Enable support for Message Passing Interface (MPI)

// HAVE_CUDA_AWARE_MPI
//
//    Have an MPI implementation that can deal with data on the GPU directly

// EXCEPTION_STACK_TRACE
//
//    Enable exception stack traces (not yet implemented)

// USE_POSIX_THREADS
//
//    Use pthreads instead of C++11 threads (which are faster) for the general 
//    Stream class

// USE_LOCAL_STREAMS
//
//    Create a new stream for each _synchronous_ operation. This prevents 
//    blocking of other operations in non-default streams. If not set, LinAlg 
//    reuses a set of global streams (for each GPU a separate stream for 
//    transfers into, transfers out of, transfers within, and computations on 
//    the device). LinAlg uses other streams than the default whenever 
//    possible such that operations in other streams do not block during the 
//    execution. However, all MAGMA LAPACK calls use the default stream.

// USE_LOCAL_CUDA_HANDLES
//
//    Create a new cuBLAS / cuSPARSE handle for each operation. If not set, 
//    LinAlg reuses a set of global handles (a separate one for each GPU).  
//    This is not recommended but it might improve thread safety.

// CUDA_BLOCK_SIZE
//
//    Number of tasks for each CUDA thread
#ifndef CUDA_BLOCK_SIZE
# define CUDA_BLOCK_SIZE 16
#endif


// USE_MAGMA_GESV
//
//    Use Magma's GESV routine instead of one 'built' by cuBLAS parts
//    TODO: change the default here
//#ifndef HAVE_MAGMA
//# define USE_CUBLAS_GESV
//#endif

// USE_MAGMA_GETRF
//
//    Use Magma's GETRF routine instead of the cuBLAS version
//    TODO: change the default here
//#ifndef HAVE_MAGMA
//# define USE_CUBLAS_GETRF
//#endif

// USE_MAGMA_GETRI
//
//    Use Magma's GETRI routine instead of the cuBLAS version
//    TODO: change the default here
//#ifndef HAVE_MAGMA
//# define USE_CUBLAS_GETRI
//#endif

// USE_MAGMA_TRSM
//
//    Use Magma's TRSM routine instead of the cuBLAS version
//    TODO: change the default here
//#ifndef HAVE_MAGMA
//# define USE_CUBLAS_TRSM
//#endif

// BUFFER_HELPER_DISPLAY
//
//    Have the BufferHelper class print its status during operations

// BUFFER_HELPER_VERBOSE
//
//    Have the BufferHelper class print information about adding tasks to the 
//    queue and syncing with them

// SIMPLE_PROFILER
//
//    Enable the simple profiler

// SIMPLE_PROFILER_PREALLOCATED_RECORDS
//
//    How many records to preallocate for each function that is profiled
#ifndef SIMPLE_PROFILER_PREALLOCATED_RECORDS
# define SIMPLE_PROFILER_PREALLOCATED_RECORDS 1000
#endif

#endif /* PREPROCESSOR_H_ */
