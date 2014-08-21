/** \file             gemm.h
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

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_helper.h"
#endif

#include "types.h"
#include "exceptions.h"
#include "checks.h"
#include "dense.h"


namespace LinAlg {

namespace BLAS {

namespace FORTRAN {

#ifndef DOXYGEN_SKIP
extern "C" {
  void fortran_name(sgemm, SGEMM)(char* transa, char* transb, I_t* m, I_t* n,
                                  I_t* k, S_t* alpha, S_t* A, I_t* lda, S_t* B,
                                  I_t* ldb, S_t* beta, S_t* C, I_t* ldc);
  void fortran_name(dgemm, DGEMM)(char* transa, char* transb, I_t* m, I_t* n,
                                  I_t* k, D_t* alpha, D_t* A, I_t* lda, D_t* B,
                                  I_t* ldb, D_t* beta, D_t* C, I_t* ldc);
  void fortran_name(cgemm, CGEMM)(char* transa, char* transb, I_t* m, I_t* n,
                                  I_t* k, C_t* alpha, C_t* A, I_t* lda, C_t* B,
                                  I_t* ldb, C_t* beta, C_t* C, I_t* ldc);
  void fortran_name(zgemm, ZGEMM)(char* transa, char* transb, I_t* m, I_t* n,
                                  I_t* k, Z_t* alpha, Z_t* A, I_t* lda, Z_t* B,
                                  I_t* ldb, Z_t* beta, Z_t* C, I_t* ldc);
}
#endif

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
 *  \param[in|out]    C
 *
 *  \param[in]        ldc
 *
 *  See [DGEMM](http://www.mathkeisan.com/usersguide/man/dgemm.html)
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, S_t alpha,
                  S_t* A, int lda, S_t* B, int ldb, S_t beta, S_t* C,
                  int ldc) {
  fortran_name(sgemm, SGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                             &ldb, &beta, C, &ldc);
};
/** \overload
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, D_t alpha,
                  D_t* A, int lda, D_t* B, int ldb, D_t beta, D_t* C,
                  int ldc) {
  fortran_name(dgemm, DGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                             &ldb, &beta, C, &ldc);
};
/** \overload
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, C_t alpha,
                  C_t* A, int lda, C_t* B, int ldb, C_t beta, C_t* C,
                  int ldc) {
  fortran_name(cgemm, CGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                             &ldb, &beta, C, &ldc);
};
/** \overload
 */
inline void xGEMM(char transa, char transb, int m, int n, int k, Z_t alpha,
                  Z_t* A, int lda, Z_t* B, int ldb, Z_t beta, Z_t* C,
                  int ldc) {
  fortran_name(zgemm, ZGEMM)(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B,
                             &ldb, &beta, C, &ldc);
};

} /* namespace FORTRAN */

#ifdef HAVE_CUDA
namespace CUBLAS {

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
 *  \param[in|out]    C
 *
 *  \param[in]        ldc
 *
 *  See [CUBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xGEMM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, I_t k,
                  const S_t* alpha, const S_t* A, I_t lda, const S_t* B,
                  I_t ldb, const S_t* beta, S_t* C, I_t ldc) {
  checkCUBLAS(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, \
                          ldb, beta, C, ldc));
};
/** \overload
 */
inline void xGEMM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, I_t k,
                  const D_t* alpha, const D_t* A, I_t lda, const D_t* B,
                  I_t ldb, const D_t* beta, D_t* C, I_t ldc) {
  checkCUBLAS(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, \
                          ldb, beta, C, ldc));
};
/** \overload
 */
inline void xGEMM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, I_t k,
                  const C_t* alpha, const C_t* A, I_t lda, const C_t* B,
                  I_t ldb, const C_t* beta, C_t* C, I_t ldc) {
  checkCUBLAS(cublasCgemm(handle, transa, transb, m, n, k, \
                          (const cuComplex*)alpha, (const cuComplex*)A, lda, \
                          (const cuComplex*)B, ldb, (const cuComplex*)beta, \
                          (cuComplex*)C, ldc));
};
/** \overload
 */
inline void xGEMM(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, I_t m, I_t n, I_t k,
                  const Z_t* alpha, const Z_t* A, I_t lda, const Z_t* B,
                  I_t ldb, const Z_t* beta, Z_t* C, I_t ldc) {
  checkCUBLAS(cublasZgemm(handle, transa, transb, m, n, k, \
                          (const cuDoubleComplex*)alpha, \
                          (const cuDoubleComplex*)A, lda, \
                          (const cuDoubleComplex*)B, ldb, \
                          (const cuDoubleComplex*)beta, \
                          (cuDoubleComplex*)C, ldc));
};

} /* namespace CUBLAS */

using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_output_transposed;
#ifdef HAVE_CUDA
using LinAlg::CUDA::CUBLAS::handles;
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
 *  \param[in|out]    C
 */
template <typename T>
inline void xGEMM(const T alpha, const Dense<T>& A, const Dense<T>& B,
                  const T beta, Dense<T>& C) {

#ifndef LINALG_NO_CHECKS
  check_device(A, B, C, "xGEMM()");
  check_output_transposed(C, "xGEMM()");

  if (A.rows() != C.rows() || A.cols() != B.rows() || B.cols() != C.cols()) {
    throw excBadArgument("xGEMM(): argument matrix size mismatch");
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
    xGEMM(transa, transb, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr,
          ldc);
  }
#ifdef HAVE_CUDA
  else if (location == Location::GPU) {
    cublasOperation_t transa = (A._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (B._transposed) ? CUBLAS_OP_T : CUBLAS_OP_N;
    BLAS::CUBLAS::xGEMM(handles[device_id], transa, transb, m, n, k, &alpha,
                        A_ptr, lda,  B_ptr, ldb, &beta, C_ptr, ldc);
  }
#endif

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xGEMM(): BLAS-3 GEMM not supported on selected "
                           "location");
  }
#endif

}

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_GEMM_H_ */
