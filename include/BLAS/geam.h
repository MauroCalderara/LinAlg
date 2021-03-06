/** \file
 *
 *  \brief            xGEAM (cuBLAS BLAS-like)
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_BLAS_GEAM_H_
#define LINALG_BLAS_GEAM_H_

/* Organization of the namespace:
 *
 *    LinAlg::BLAS
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::BLAS::<NAME>
 *        bindings to the <NAME> BLAS backend
 */

#include <utility>    // std::move

#include "../preprocessor.h"

#ifdef HAVE_CUDA
# include <cuda_runtime.h>
# include <cublas_v2.h>
# include "../CUDA/cuda_checks.h"
# include "../CUDA/cuda_cublas.h"
#endif

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../utilities/checks.h"
#include "../streams.h"
#include "../dense.h"


namespace LinAlg {

namespace BLAS {

#ifdef HAVE_CUDA
namespace cuBLAS {

/** \brief            A cuBLAS routine to copy, transpose and add dense
 *                    matrices on the GPU
 *
 *  C = alpha * A + beta * B          where A and B can be transposed matrices
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
inline void xGEAM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const S_t alpha,
                  const S_t* A, I_t lda, const S_t beta, const S_t* B, I_t ldb,
                  S_t* C, I_t ldc) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasSgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, \
                          B, ldb, C, ldc));

}
/** \overload
 */
inline void xGEAM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const D_t alpha,
                  const D_t* A, I_t lda, const D_t beta, const D_t* B, I_t ldb,
                  D_t* C, I_t ldc) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasDgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, \
                          B, ldb, C, ldc));

}
/** \overload
 */
inline void xGEAM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const C_t alpha,
                  const C_t* A, I_t lda, const C_t beta, const C_t* B, I_t ldb,
                  C_t* C, I_t ldc) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasCgeam(handle, transa, transb, m, n, \
                          (cuComplex*)(&alpha), (cuComplex*)(A), lda, \
                          (cuComplex*)(&beta), (cuComplex*)(B), ldb, \
                          (cuComplex*)(C), ldc));

}
/** \overload
 */
inline void xGEAM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const Z_t alpha,
                  const Z_t* A, I_t lda, const Z_t beta, const Z_t* B, I_t ldb,
                  Z_t* C, I_t ldc) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasZgeam(handle, transa, transb, m, n, \
                          (cuDoubleComplex*)(&alpha), \
                          (cuDoubleComplex*)(A), lda, \
                          (cuDoubleComplex*)(&beta), \
                          (cuDoubleComplex*)(B), ldb, \
                          (cuDoubleComplex*)(C), ldc));

}
/** \overload
 */
template <typename T>
inline void xGEAM(int device_id, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const T alpha,
                  const T* A, I_t lda, const T beta, const T* B, I_t ldb,
                  T* C, I_t ldc) {
  xGEAM(LinAlg::CUDA::cuBLAS::handles[device_id], transa, transb, m, n, alpha,
        A, lda, beta, B, ldb, C, ldc);
}

} /* namespace LinAlg::BLAS::cuBLAS */

using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_output_transposed;
using LinAlg::Utilities::check_gpu_structures;
using LinAlg::Utilities::check_stream_no_prefer_native;
using LinAlg::Utilities::check_stream_device_id;
using LinAlg::CUDA::cuBLAS::prepare_cublas;
using LinAlg::CUDA::cuBLAS::finish_cublas;

/** \brief            A cuBLAS routine to copy, transpose and add dense
 *                    matrices on the GPU, asynchronous variant.
 *
 *  C = alpha * A + beta * B          where A and B can be transposed matrices
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        beta
 *
 *  \param[in]        B
 *
 *  \param[in]        C
 *
 *  \param[in]        stream
 *
 *  For more details see cublas documentation under BLAS-like functions
 */
template <typename T>
inline I_t xGEAM_async(const T alpha, const Dense<T>& A, const T beta,
                       const Dense<T>& B, Dense<T>& C, Stream& stream) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEAM_async()");
  check_output_transposed(C, "xGEAM_async()");
  check_gpu_structures("xGEAM_async()");

  if (A.rows() != B.rows() || A.rows() != C.rows() ||
      A.cols() != B.cols() || A.cols() != C.cols()   ) {

    throw excBadArgument("xGEAM_async(): argument matrix size mismatch");

  } else if (A._location != Location::GPU) {

    throw excBadArgument("xGEAM_async(): matrices must reside on the GPU");

  }

  check_stream_no_prefer_native(stream, "xGEAM_async()");
  check_stream_device_id(stream, A._device_id, "xGEAM_async()");
#endif

  auto device_id = A._device_id;
  cublasOperation_t transa = (A._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = (B._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto m = A.rows();
  auto n = A.cols();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  auto B_ptr = B._begin();
  auto ldb = B._leading_dimension;
  auto C_ptr = C._begin();
  auto ldc = C._leading_dimension;

  int          prev_device = 0;
  cudaStream_t prev_cuda_stream;

  auto handle = prepare_cublas(stream, &prev_device, &prev_cuda_stream);
  cuBLAS::xGEAM(*handle, transa, transb, m, n, alpha, A_ptr, lda, beta, B_ptr, 
                ldb, C_ptr, ldc);
  finish_cublas(stream, &prev_device, &prev_cuda_stream, handle);

  return 0;

}

/** \brief            A cuBLAS routine to copy, transpose and add dense
 *                    matrices on the GPU
 *
 *  C = alpha * A + beta * B          where A and B can be transposed matrices
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        beta
 *
 *  \param[in]        B
 *
 *  \param[in]        C
 *
 *  See [cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
template <typename T>
inline void xGEAM(const T alpha, const Dense<T>& A, const T beta,
                  const Dense<T>& B, Dense<T>& C) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEAM()");
  check_output_transposed(C, "xGEAM()");
  check_gpu_structures("xGEAM()");

  if (A.rows() != B.rows() || A.rows() != C.rows() ||
      A.cols() != B.cols() || A.cols() != C.cols()   ) {

    throw excBadArgument("xGEAM(): argument matrix size mismatch");

  } else if (A._location != Location::GPU) {

    throw excBadArgument("xGEAM(): matrices must reside on the GPU");

  }
#endif

  // As there is only little code, we duplicate the code from xGEAM_async() 
  // here

  auto device_id = A._device_id;
  cublasOperation_t transa = (A._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = (B._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto m = A.rows();
  auto n = A.cols();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  auto B_ptr = B._begin();
  auto ldb = B._leading_dimension;
  auto C_ptr = C._begin();
  auto ldc = C._leading_dimension;

  int          prev_device = 0;
  cudaStream_t prev_cuda_stream;
  Stream*      stream_;

#ifndef USE_LOCAL_STREAMS
  stream_ = &(LinAlg::CUDA::on_stream[device_id]);
#else
  Stream my_stream(device_id);
  stream_ = &my_stream;
#endif

  auto handle = prepare_cublas(*stream_, &prev_device, &prev_cuda_stream);
  cuBLAS::xGEAM(*handle, transa, transb, m, n, alpha, A_ptr, lda, beta, B_ptr, 
                ldb, C_ptr, ldc);
  finish_cublas(*stream_, &prev_device, &prev_cuda_stream, handle);

  stream_->sync_cuda();

}

#endif /* HAVE_CUDA */

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_GEAM_H_ */
