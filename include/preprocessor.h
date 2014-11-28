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

// USE_GLOBAL_TRANSFER_STREAMS
//
//    Use a global stream for each of the following tasks:
//
//      transfer data into a GPU
//      transfer data out of a GPU
//      transfer data within one GPU
//
//    and each GPU in the system.
//
//    The streams will be stored in LinAlg::CUDA::
//      in_stream[device_id]
//      out_stream[device_id]
//      on_stream[device_id]
//

// USE_POSIX_THREADS
//
//    Use pthreads instead of C++11 threads (which are faster) for the general 
//    Stream class

// USE_MAGMA_GESV
//
//    Use Magma's GESV routine instead of one 'built' by CUBLAS parts

// USE_MAGMA_GETRF
//
//    Use Magma's GETRF routine instead of the CUBLAS version

// USE_MAGMA_GETRI
//
//    Use Magma's GETRI routine instead of the CUBLAS version

// USE_MAGMA_TRSM
//
//    Use Magma's TRSM routine instead of the CUBLAS version

// BUFFER_HELPER_DISPLAY
//
//    Have the BufferHelper class print its status during operations

// BUFFER_HELPER_VERBOSE
//
//    Have the BufferHelper class print information about adding tasks to the 
//    queue and syncing with them

#endif /* PREPROCESSOR_H_ */
