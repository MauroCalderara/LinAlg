/** \file
 *
 *  \brief            xGEAM (CUBLAS BLAS-like)
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

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../CUDA/cuda_checks.h"
#include "../CUDA/cuda_cublas.h"
#endif

#include <utility>    // std::move

#include "../types.h"
#include "../exceptions.h"
#include "../utilities/checks.h"
#include "../streams.h"
#include "../dense.h"


namespace LinAlg {

namespace BLAS {

#ifdef HAVE_CUDA
namespace CUBLAS {

/** \brief            A CUBLAS routine to copy, transpose and add dense
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
 *  See [CUBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xGEAM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const S_t alpha,
                  const S_t* A, I_t lda, const S_t beta, const S_t* B, I_t ldb,
                  S_t* C, I_t ldc) {
  checkCUBLAS(cublasSgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, \
                          B, ldb, C, ldc));
};
/** \overload
 */
inline void xGEAM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const D_t alpha,
                  const D_t* A, I_t lda, const D_t beta, const D_t* B, I_t ldb,
                  D_t* C, I_t ldc) {
  checkCUBLAS(cublasDgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, \
                          B, ldb, C, ldc));
};
/** \overload
 */
inline void xGEAM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const C_t alpha,
                  const C_t* A, I_t lda, const C_t beta, const C_t* B, I_t ldb,
                  C_t* C, I_t ldc) {

  checkCUBLAS(cublasCgeam(handle, transa, transb, m, n, \
                          (cuComplex*)(&alpha), (cuComplex*)(A), lda, \
                          (cuComplex*)(&beta), (cuComplex*)(B), ldb, \
                          (cuComplex*)(C), ldc));
};
/** \overload
 */
inline void xGEAM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const Z_t alpha,
                  const Z_t* A, I_t lda, const Z_t beta, const Z_t* B, I_t ldb,
                  Z_t* C, I_t ldc) {
  checkCUBLAS(cublasZgeam(handle, transa, transb, m, n, \
                          (cuDoubleComplex*)(&alpha), \
                          (cuDoubleComplex*)(A), lda, \
                          (cuDoubleComplex*)(&beta), \
                          (cuDoubleComplex*)(B), ldb, \
                          (cuDoubleComplex*)(C), ldc));
};
/** \overload
 */
template <typename T>
inline void xGEAM(int device_id, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, const T alpha,
                  const T* A, I_t lda, const T beta, const T* B, I_t ldb,
                  T* C, I_t ldc) {
  xGEAM(LinAlg::CUDA::CUBLAS::handles[device_id], transa, transb, m, n, alpha,
        A, lda, beta, B, ldb, C, ldc);
};

} /* namespace LinAlg::BLAS::CUBLAS */

using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_output_transposed;
using LinAlg::CUDA::CUBLAS::handles;

/** \brief            A CUBLAS routine to copy, transpose and add dense
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
 *  See [CUBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
template <typename T>
inline void xGEAM(const T alpha, const Dense<T>& A, const T beta,
                  const Dense<T>& B, Dense<T>& C) {

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEAM()");
  check_output_transposed(C, "xGEAM()");

  if (A.rows() != B.rows() || A.rows() != C.rows() ||
      A.cols() != B.cols() || A.cols() != C.cols()   ) {

    throw excBadArgument("xGEAM(): argument matrix size mismatch");

  } else if (A._location != Location::GPU) {

    throw excBadArgument("xGEAM(): matrices must reside on the GPU");

  }
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

  xGEAM(handles[device_id], transa, transb, m, n, alpha, A_ptr, lda,
        beta, B_ptr, ldb, C_ptr, ldc);

};

/** \brief            A CUBLAS routine to copy, transpose and add dense
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
inline void xGEAM_async(const T alpha, const Dense<T>& A, const T beta,
                        const Dense<T>& B, Dense<T>& C, CUDAStream& stream) {

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEAM()");
  check_output_transposed(C, "xGEAM()");

  if (stream.synchronous_operation) {

    xGEAM(alpha, A, beta, B, C);

    return;

  }

  if (A.rows() != B.rows() || A.rows() != C.rows() ||
      A.cols() != B.cols() || A.cols() != C.cols()   ) {

    throw excBadArgument("xGEAM(): argument matrix size mismatch");

  } else if (A._location != Location::GPU) {

    throw excBadArgument("xGEAM(): matrices must reside on the GPU");

  }
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

  xGEAM(stream.cublas_handle, transa, transb, m, n, alpha, A_ptr, lda, beta,
        B_ptr, ldb, C_ptr, ldc);

};
/** \overload
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
 *  \returns          A stream to synchronize the operation on
 */
template <typename T>
inline CUDAStream xGEAM_async(const T alpha, const Dense<T>& A, const T beta,
                              const Dense<T>& B, Dense<T>&C) {

  CUDAStream stream;

  xGEAM_async(alpha, A, beta, B, C, stream);

  return std::move(stream);

};

#endif /* HAVE_CUDA */

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_GEAM_H_ */
