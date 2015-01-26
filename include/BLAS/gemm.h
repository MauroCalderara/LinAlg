/** \file
 *
 *  \brief            xGEMM (BLAS-3)
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_BLAS_GEMM_H_
#define LINALG_BLAS_GEMM_H_

/* Organization of the namespace:
 *
 *    LinAlg::BLAS
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::BLAS::<NAME>
 *        bindings to the <NAME> BLAS backend
 */

#include "../preprocessor.h"

#ifdef HAVE_CUDA
# include <cuda_runtime.h>
# include <cublas_v2.h>
# include "../CUDA/cuda_checks.h"
# include "../CUDA/cuda_cublas.h"
#endif

#include "../types.h"

#ifdef HAVE_MKL
# include <mkl.h>    // need to have types.h before mkl.h
#endif

#include "../profiling.h"
#include "../exceptions.h"
#include "../utilities/checks.h"
#include "../streams.h"
#include "../dense.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  void fortran_name(sgemm, SGEMM)(const char* transa, const char* transb, 
                                  const I_t* m, const I_t* n, const I_t* k, 
                                  const S_t* alpha, const S_t* A, 
                                  const I_t* lda, const S_t* B, const I_t* ldb, 
                                  const S_t* beta, S_t* C, const I_t* ldc);
  void fortran_name(dgemm, DGEMM)(const char* transa, const char* transb, 
                                  const I_t* m, const I_t* n, const I_t* k, 
                                  const D_t* alpha, const D_t* A, 
                                  const I_t* lda, const D_t* B, const I_t* ldb, 
                                  const D_t* beta, D_t* C, const I_t* ldc);
  void fortran_name(cgemm, CGEMM)(const char* transa, const char* transb, 
                                  const I_t* m, const I_t* n, const I_t* k, 
                                  const C_t* alpha, const C_t* A, 
                                  const I_t* lda, const C_t* B, const I_t* ldb, 
                                  const C_t* beta, C_t* C, const I_t* ldc);
  void fortran_name(zgemm, ZGEMM)(const char* transa, const char* transb, 
                                  const I_t* m, const I_t* n, const I_t* k, 
                                  const Z_t* alpha, const Z_t* A, 
                                  const I_t* lda, const Z_t* B, const I_t* ldb, 
                                  const Z_t* beta, Z_t* C, const I_t* ldc);
#ifdef HAVE_MKL
  void fortran_name(scgemm, SCGEMM)(const char* transa, const char* transb, 
                                    const I_t* m, const I_t* n, const I_t* k, 
                                    const C_t* alpha, const S_t* A, 
                                    const I_t* lda, const C_t* B, 
                                    const I_t* ldb, const C_t* beta, C_t* C, 
                                    const I_t* ldc);
  void fortran_name(dzgemm, DZGEMM)(const char* transa, const char* transb, 
                                    const I_t* m, const I_t* n, const I_t* k, 
                                    const Z_t* alpha, const D_t* A, 
                                    const I_t* lda, const Z_t* B, 
                                    const I_t* ldb, const Z_t* beta, Z_t* C, 
                                    const I_t* ldc);
#endif
}
#endif /* DOXYGEN_SKIP */

namespace LinAlg {

namespace BLAS {

namespace FORTRAN {

/** \brief            General matrix-matrix multiply
 *
 *  C = alpha * op(A) * op(B) + beta * C
 *
 *  \param[in]        transa
 *
 *  \param[in]        transb
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in]        k
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        lda
 *
 *  \param[in]        B
 *
 *  \param[in]        ldb
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    C
 *
 *  \param[in]        ldc
 *
 *  See [DGEMM](http://www.mathkeisan.com/usersguide/man/dgemm.html)
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, S_t alpha,
                  S_t* A, int lda, S_t* B, int ldb, S_t beta, S_t* C, int ldc) {

  PROFILING_FUNCTION_HEADER

  fortran_name(sgemm, SGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                             &ldb, &beta, C, &ldc);

}
/** \overload
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, D_t alpha,
                  D_t* A, int lda, D_t* B, int ldb, D_t beta, D_t* C, int ldc) {

  PROFILING_FUNCTION_HEADER

  fortran_name(dgemm, DGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                             &ldb, &beta, C, &ldc);
}
/** \overload
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, C_t alpha,
                  C_t* A, int lda, C_t* B, int ldb, C_t beta, C_t* C, int ldc) {

  PROFILING_FUNCTION_HEADER

  fortran_name(cgemm, CGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                             &ldb, &beta, C, &ldc);

}
/** \overload
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, Z_t alpha,
                  Z_t* A, int lda, Z_t* B, int ldb, Z_t beta, Z_t* C, int ldc) {

  PROFILING_FUNCTION_HEADER

  fortran_name(zgemm, ZGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                             &ldb, &beta, C, &ldc);

}

} /* namespace FORTRAN */

#ifdef HAVE_MKL
namespace MKL {

/** \brief            General real-matrix x complex-matrix multiply
 *
 *  C = alpha * op(A) * op(B) + beta * C      with A real; B, and, C complex
 *
 *  \param[in]        transa
 *
 *  \param[in]        transb
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in]        k
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        lda
 *
 *  \param[in]        B
 *
 *  \param[in]        ldb
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    C
 *
 *  \param[in]        ldc
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, C_t alpha,
                  S_t* A, int lda, C_t* B, int ldb, C_t beta, C_t* C, int ldc) {

  PROFILING_FUNCTION_HEADER

  fortran_name(scgemm, SCGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                               &ldb, &beta, C, &ldc);

}
inline void xGEMM(char transa, char transb, int m, int n, int k, Z_t alpha,
                  D_t* A, int lda, Z_t* B, int ldb, Z_t beta, Z_t* C, int ldc) {

  PROFILING_FUNCTION_HEADER

  fortran_name(dzgemm, DZGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                               &ldb, &beta, C, &ldc);

}

} /* namespace LinAlg::BLAS::MKL */
#endif /* HAVE_MKL */

#ifdef HAVE_CUDA
namespace cuBLAS {

/** \brief            General matrix-matrix multiply
 *
 *  C = alpha * op(A) * op(B) + beta * C
 *
 *  \param[in]        handle
 *
 *  \param[in]        transa
 *
 *  \param[in]        transb
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in]        k
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        lda
 *
 *  \param[in]        B
 *
 *  \param[in]        ldb
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    C
 *
 *  \param[in]        ldc
 *
 *  See [cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xGEMM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, I_t k,
                  const S_t* alpha, const S_t* A, I_t lda, const S_t* B,
                  I_t ldb, const S_t* beta, S_t* C, I_t ldc) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, \
                          ldb, beta, C, ldc));

}
/** \overload
 */
inline void xGEMM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, I_t k,
                  const D_t* alpha, const D_t* A, I_t lda, const D_t* B,
                  I_t ldb, const D_t* beta, D_t* C, I_t ldc) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, \
                          ldb, beta, C, ldc));

}
/** \overload
 */
inline void xGEMM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, I_t k,
                  const C_t* alpha, const C_t* A, I_t lda, const C_t* B,
                  I_t ldb, const C_t* beta, C_t* C, I_t ldc) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasCgemm(handle, transa, transb, m, n, k, \
                          (const cuComplex*)alpha, (const cuComplex*)A, lda, \
                          (const cuComplex*)B, ldb, (const cuComplex*)beta, \
                          (cuComplex*)C, ldc));

}
/** \overload
 */
inline void xGEMM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, I_t k,
                  const Z_t* alpha, const Z_t* A, I_t lda, const Z_t* B,
                  I_t ldb, const Z_t* beta, Z_t* C, I_t ldc) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasZgemm(handle, transa, transb, m, n, k, \
                          (const cuDoubleComplex*)alpha, \
                          (const cuDoubleComplex*)A, lda, \
                          (const cuDoubleComplex*)B, ldb, \
                          (const cuDoubleComplex*)beta, \
                          (cuDoubleComplex*)C, ldc));

}

} /* namespace LinAlg::BLAS::cuBLAS */
#endif /* HAVE_CUDA */

using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_output_transposed;
#ifdef HAVE_CUDA
using LinAlg::Utilities::check_gpu_structures;
using LinAlg::Utilities::check_stream_prefer_native;
using LinAlg::Utilities::check_stream_device_id;
using LinAlg::Utilities::check_stream_alive;
using LinAlg::CUDA::cuBLAS::prepare_cublas;
using LinAlg::CUDA::cuBLAS::finish_cublas;
#endif

// Convenience bindings (bindings for Dense<T>)
/** \brief            General matrix-matrix multiply
 *
 *  C = alpha * op(A) * op(B) + beta * C
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        B
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    C
 */
template <typename T>
inline void xGEMM(const T alpha, const Dense<T>& A, const Dense<T>& B,
                  const T beta, Dense<T>& C) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEMM(alpha, A, B, beta, C)");
  check_output_transposed(C, "xGEMM(alpha, A, B, beta, C)");

  if (A.rows() != C.rows() || A.cols() != B.rows() || B.cols() != C.cols()) {
    throw excBadArgument("xGEMM(alpha, A, B, beta, C), A, B, C: argument "
                         "matrix size mismatch (A:%dx%d B:%dx%d C:%dx%d)",
                         A.rows(), A.cols(), B.rows(), B.cols(), C.rows(),
                         C.cols());
  }
#endif

  auto location = A._location;
  auto device_id = A._device_id;
  auto m = A.rows();
  auto n = B.cols();
  auto k = B.rows();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  auto B_ptr = B._begin();
  auto ldb = B._leading_dimension;
  auto C_ptr = C._begin();
  auto ldc = C._leading_dimension;

  if (location == Location::host) {

    char transa = (A._transposed) ? 'T' : 'N';
    char transb = (B._transposed) ? 'T' : 'N';

    FORTRAN::xGEMM(transa, transb, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta,
                   C_ptr, ldc);

  }
#ifdef HAVE_CUDA
  else if (location == Location::GPU) {

    using BLAS::cuBLAS::xGEMM;

# ifndef LINALG_NO_CHECKS
    check_gpu_structures("xGEMM()");
# endif

    cublasOperation_t transa = (A._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (B._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;

    int               prev_device = 0;
    cudaStream_t      prev_cuda_stream;
    Stream*           stream_;

# ifndef USE_LOCAL_STREAMS
    stream_ = &(LinAlg::CUDA::compute_stream[device_id]);
# else
    Stream my_stream(device_id);
    stream_ = &my_stream;
# endif

    auto handle = prepare_cublas(*stream_, &prev_device, &prev_cuda_stream);
    xGEMM(*handle, transa, transb, m, n, k, &alpha, A_ptr, lda,  B_ptr, ldb, 
          &beta, C_ptr, ldc);
    finish_cublas(*stream_, &prev_device, &prev_cuda_stream, handle);

    stream_->sync_cuda();

  }
#endif

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("xGEMM(): BLAS-3 GEMM not supported on selected "
                           "location");

  }
#endif

}

/** \overload
 */
template <typename T, typename U, typename V>
inline void xGEMM(const T alpha, const Dense<U>& A, const Dense<V>& B,
                  const V beta, Dense<V>& C) {

  // This is the most general case that is only supported on the CPU if we 
  // have MKL support

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEMM(alpha, A, B, beta, C)");
  check_output_transposed(C, "xGEMM(alpha, A, B, beta, C)");

  if (A.rows() != C.rows() || A.cols() != B.rows() || B.cols() != C.cols()) {
    throw excBadArgument("xGEMM(alpha, A, B, beta, C), A, B, C: argument "
                         "matrix size mismatch (A:%dx%d B:%dx%d C:%dx%d)",
                         A.rows(), A.cols(), B.rows(), B.cols(), C.rows(),
                         C.cols());
  }

  if (A._location != Location::host) {
    throw excUnimplemented("xGEMM(alpha, A, B, beta, C): the most general case "
                           "of this operation is only supported for matrices "
                           "in main memory (see BLAS/gemm.h)");
  
  }
#ifndef HAVE_MKL
  throw excUnimplemented("xGEMM(alpha, A, B, beta, C): the most general case "
                         "of this operation (alpha complex, beta complex != 1) "
                         "is only supported when using MKL (see BLAS/gemm.h)");
#endif
#else /* LINALG_NO_CHECKS */
#ifndef HAVE_MKL
  std::printf("xGEMM(alpha, A, B, beta, C): real-complex requires MKL, "
              "returning\n");
  return;
#endif
#endif /* not LINALG_NO_CHECKS */

#ifdef HAVE_MKL
  auto location = A._location;
  auto device_id = A._device_id;
  auto m = A.rows();
  auto n = B.cols();
  auto k = B.rows();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  auto B_ptr = B._begin();
  auto ldb = B._leading_dimension;
  auto C_ptr = C._begin();
  auto ldc = C._leading_dimension;
  char transa = (A._transposed) ? 'T' : 'N';
  char transb = (B._transposed) ? 'T' : 'N';

  MKL::xGEMM(transa, transb, m, n, k, cast<V>(alpha), A_ptr, lda, B_ptr, ldb, 
             cast<V>(beta), C_ptr, ldc);
#endif

}

/** \brief            General asynchronous matrix-matrix multiply
 *
 *  C = alpha * op(A) * op(B) + beta * C
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        B
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    C
 *
 *  \param[in]        stream
 *
 *  \returns          The ticket number for the operation on the stream
 */
template <typename T>
inline I_t xGEMM_async(const T alpha, const Dense<T>& A, const Dense<T>& B,
                       const T beta, Dense<T>& C, Stream& stream) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEMM(alpha, A, B, beta, C)");
  check_output_transposed(C, "xGEMM(alpha, A, B, beta, C)");

  if (A.rows() != C.rows() || A.cols() != B.rows() || B.cols() != C.cols()) {
    throw excBadArgument("xGEMM(alpha, A, B, beta, C), A, B, C: argument "
                         "matrix size mismatch (A:%dx%d B:%dx%d C:%dx%d)",
                         A.rows(), A.cols(), B.rows(), B.cols(), C.rows(),
                         C.cols());
  }
#endif

  I_t ticket = 0;

  auto location = A._location;
  auto device_id = A._device_id;
  auto m = A.rows();
  auto n = B.cols();
  auto k = B.rows();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  auto B_ptr = B._begin();
  auto ldb = B._leading_dimension;
  auto C_ptr = C._begin();
  auto ldc = C._leading_dimension;

  if (location == Location::host) {

    if (stream.synchronous) {

      char transa = (A._transposed) ? 'T' : 'N';
      char transb = (B._transposed) ? 'T' : 'N';

      FORTRAN::xGEMM(transa, transb, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, 
                     beta, C_ptr, ldc);

    } else {
    
      // Create a task using the synchronous variant

#ifndef LINALG_NO_CHECKS
      check_stream_alive(stream, "xGEMM_async()");
#endif

      // Arguments passed by copy, ensures memory lifetime but callee can't 
      // modify the arguments anymore
      auto task = [=]() mutable { xGEMM(alpha, A, B, beta, C); };

      ticket = stream.add(task);

    }

  }
#ifdef HAVE_CUDA
  else if (location == Location::GPU) {

    using BLAS::cuBLAS::xGEMM;

# ifndef LINALG_NO_CHECKS
    check_gpu_structures("xGEMM_async()");
    check_stream_prefer_native(stream, "xGEMM_async()");
    check_stream_device_id(stream, device_id, "xGEMM_async()");
# endif

    cublasOperation_t transa = (A._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (B._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;

    int               prev_device = 0;
    cudaStream_t      prev_cuda_stream;

    auto handle = prepare_cublas(stream, &prev_device, &prev_cuda_stream);
    xGEMM(*handle, transa, transb, m, n, k, &alpha, A_ptr, lda,  B_ptr, ldb, 
          &beta, C_ptr, ldc);
    finish_cublas(stream, &prev_device, &prev_cuda_stream, handle);

  }
#endif

#ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("xGEMM(): BLAS-3 GEMM not supported on selected "
                           "location");

  }
#endif

  return ticket;

}

/** \overload
 */
template <typename T, typename U, typename V>
inline I_t xGEMM_async(const T alpha, const Dense<U>& A, const Dense<V>& B,
                       const V beta, Dense<V>& C, Stream& stream) {

  // This is the most general case that is only supported on the CPU if we 
  // have MKL support

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEMM_async(alpha, A, B, beta, C)");
  check_output_transposed(C, "xGEMM_async(alpha, A, B, beta, C)");

  if (A.rows() != C.rows() || A.cols() != B.rows() || B.cols() != C.cols()) {
    throw excBadArgument("xGEMM_async(alpha, A, B, beta, C), A, B, C: " 
                         "argument matrix size mismatch (A:%dx%d B:%dx%d "
                         "C:%dx%d)", A.rows(), A.cols(), B.rows(), B.cols(), 
                         C.rows(), C.cols());
  }

  if (A._location != Location::host) {
    throw excUnimplemented("xGEMM_async(alpha, A, B, beta, C): the most "
                           "general case of this operation is only supported "
                           "for matrices in main memory (see BLAS/gemm.h)");
  
  }
# ifndef HAVE_MKL
  throw excUnimplemented("xGEMM_async(alpha, A, B, beta, C): the most "
                         "general case of this operation (alpha complex, beta "
                         "complex != 1) is only supported when using MKL "
                         "(see BLAS/gemm.h)");
# endif
#else /* LINALG_NO_CHECKS */
# ifndef HAVE_MKL
  std::printf("xGEMM_async(alpha, A, B, beta, C): real-complex requires MKL, "
              "returning\n");
  return;
# endif
#endif /* not LINALG_NO_CHECKS */

  I_t ticket = 0;

#ifdef HAVE_MKL
  auto location = A._location;
  auto device_id = A._device_id;
  auto m = A.rows();
  auto n = B.cols();
  auto k = B.rows();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  auto B_ptr = B._begin();
  auto ldb = B._leading_dimension;
  auto C_ptr = C._begin();
  auto ldc = C._leading_dimension;

  if (location == Location::host) {

    char transa = (A._transposed) ? 'T' : 'N';
    char transb = (B._transposed) ? 'T' : 'N';

    if (stream.synchronous) {
      
      MKL::xGEMM(transa, transb, m, n, k, cast<V>(alpha), A_ptr, lda, B_ptr, 
                 ldb, beta, C_ptr, ldc);

    } else {

      // Create a task using the synchronous variant

# ifndef LINALG_NO_CHECKS
      check_stream_alive(stream, "xGEMM_async()");
# endif

      // Arguments passed by copy, ensures memory lifetime but callee can't 
      // modify the arguments anymore
      auto task = [=]() mutable { xGEMM(alpha, A, B, beta, C); };

      ticket = stream.add(task);

    }

  }
# ifndef LINALG_NO_CHECKS
  else {

    throw excUnimplemented("xGEMM_async(): BLAS-3 GEMM not supported on "
                           "selected location");

  }
# endif

#endif /* HAVE_MKL */

  return ticket;

}

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_GEMM_H_ */
