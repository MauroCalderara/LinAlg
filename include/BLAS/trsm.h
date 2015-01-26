/** \file
 *
 *  \brief            xTRSM (BLAS-3)
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_BLAS_TRSM_H_
#define LINALG_BLAS_TRSM_H_

/* Organization of the namespace:
 *
 *    LinAlg::BLAS
 *        convenience bindings supporting different locations for Dense<T>
 *
 *        'Abstract' functions like 'solve' and 'invert'
 *
 *    LinAlg::BLAS::<NAME>
 *        bindings to the <NAME> BLAS backend
 */

#include "../preprocessor.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../CUDA/cuda_checks.h"
#include "../CUDA/cuda_cublas.h"
#endif

#include "../types.h"
#include "../profiling.h"
#include "../utilities/checks.h"
#include "../dense.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  void fortran_name(strsm, STRSM)(const char* side, const char* uplo,
                                  const char* transa, const char* diag,
                                  const I_t* m, const I_t* n, const S_t* alpha,
                                  const S_t* A, const I_t* lda, S_t* B,
                                  const I_t* ldb);
  void fortran_name(dtrsm, DTRSM)(const char* side, const char* uplo,
                                  const char* transa, const char* diag,
                                  const I_t* m, const I_t* n, const D_t* alpha,
                                  const D_t* A, const I_t* lda, D_t* B,
                                  const I_t* ldb);
  void fortran_name(ctrsm, CTRSM)(const char* side, const char* uplo,
                                  const char* transa, const char* diag,
                                  const I_t* m, const I_t* n, const C_t* alpha,
                                  const C_t* A, const I_t* lda, C_t* B,
                                  const I_t* ldb);
  void fortran_name(ztrsm, ZTRSM)(const char* side, const char* uplo,
                                  const char* transa, const char* diag,
                                  const I_t* m, const I_t* n, const Z_t* alpha,
                                  const Z_t* A, const I_t* lda, Z_t* B,
                                  const I_t* ldb);
}
#endif

namespace LinAlg {

namespace BLAS {

namespace FORTRAN {

/** \brief            Triangular matrix solve
 *
 *  op(A) * X = alpha * B
 *  X * op(A) = alpha * B
 *
 *  \param[in]        side
 *
 *  \param[in]        uplo
 *
 *  \param[in]        transa
 *
 *  \param[in]        diag
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
 *  \param[in,out]    B
 *
 *  \param[in]        ldb
 *
 *
 *  See [DTRSM](http://www.mathkeisan.com/usersguide/man/dtrsm.html)
 */
inline void xTRSM(char side, char uplo, char transa, char diag, int m,
                  int n, S_t alpha, S_t* A, int lda, S_t* B, int ldb) {

  PROFILING_FUNCTION_HEADER

  fortran_name(strsm, STRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, A,
                             &lda, B, &ldb);

}
/** \overload
 */
inline void xTRSM(char side, char uplo, char transa, char diag, int m,
                  int n, D_t alpha, D_t* A, int lda, D_t* B, int ldb) {

  PROFILING_FUNCTION_HEADER

  fortran_name(dtrsm, DTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, A,
                             &lda, B, &ldb);

}
/** \overload
 */
inline void xTRSM(char side, char uplo, char transa, char diag, int m,
                  int n, C_t alpha, C_t* A, int lda, C_t* B, int ldb) {

  PROFILING_FUNCTION_HEADER

  fortran_name(ctrsm, CTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, A,
                             &lda, B, &ldb);

}
/** \overload
 */
inline void xTRSM(char side, char uplo, char transa, char diag, int m,
                  int n, Z_t alpha, Z_t* A, int lda, Z_t* B, int ldb) {

  PROFILING_FUNCTION_HEADER

  fortran_name(ztrsm, ZTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, A,
                             &lda, B, &ldb);

}

} /* namespace LinAlg::BLAS::FORTRAN */

#ifdef HAVE_CUDA
namespace cuBLAS {

/** \brief            Triangular matrix solve
 *
 *  op(A) * X = alpha * B
 *  X * op(A) = alpha * B
 *
 *  \param[in]        handle
 *
 *  \param[in]        side
 *
 *  \param[in]        uplo
 *
 *  \param[in]        trans
 *
 *  \param[in]        diag
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
 *  \param[in,out]    B
 *
 *  \param[in]        ldb
 *
 *  See [cuBLAS Documentation](http://docs.nvidia.com/cuda/cublas/)
 */
inline void xTRSM(cublasHandle_t handle, cublasSideMode_t side,
                  cublasFillMode_t uplo, cublasOperation_t trans,
                  cublasDiagType_t diag, I_t m, I_t n, const S_t alpha,
                  const S_t* A, I_t lda, S_t* B, I_t ldb) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, \
                          lda, B, ldb));

}
/** \overload
 */
inline void xTRSM(cublasHandle_t handle, cublasSideMode_t side,
                  cublasFillMode_t uplo, cublasOperation_t trans,
                  cublasDiagType_t diag, I_t m, I_t n, const D_t alpha,
                  const D_t* A, I_t lda, D_t* B, I_t ldb) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, \
                          lda, B, ldb));

}
/** \overload
 */
inline void xTRSM(cublasHandle_t handle, cublasSideMode_t side,
                  cublasFillMode_t uplo, cublasOperation_t trans,
                  cublasDiagType_t diag, I_t m, I_t n, const C_t alpha,
                  const C_t* A, I_t lda, C_t* B, I_t ldb) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasCtrsm(handle, side, uplo, trans, diag, m, n, \
                          (const cuComplex*)&alpha, (const cuComplex*)A, lda, \
                          (cuComplex*)B, ldb));

}
/** \overload
 */
inline void xTRSM(cublasHandle_t handle, cublasSideMode_t side,
                  cublasFillMode_t uplo, cublasOperation_t trans,
                  cublasDiagType_t diag, I_t m, I_t n, const Z_t alpha,
                  const Z_t* A, I_t lda, Z_t* B, I_t ldb) {

  PROFILING_FUNCTION_HEADER

  checkCUBLAS(cublasZtrsm(handle, side, uplo, trans, diag, m, n, \
                          (const cuDoubleComplex*)&alpha, \
                          (const cuDoubleComplex*)A, lda, \
                          (cuDoubleComplex*)B, ldb));

}

} /* namespace LinAlg::BLAS::cuBLAS */

# ifdef HAVE_MAGMA

namespace MAGMA {

/** \brief            Triangular matrix solve
 *
 *  op(A) * X = alpha * B
 *  X * op(A) = alpha * B
 *
 *  \param[in]        side
 *
 *  \param[in]        uplo
 *
 *  \param[in]        trans
 *
 *  \param[in]        diag
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A (on device)
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    B (on device)
 *
 *  \param[in]        ldb
 *
 *  See
 *  [DTRSM](http://www.math.utah.edu/software/lapack/lapack-blas/dtrsm.html)
 */
inline void xTRSM(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                  magma_diag_t diag, magma_int_t m, magma_int_t n, S_t alpha,
                  const S_t* A, magma_int_t lda, S_t* B, magma_int_t ldb) {

  PROFILING_FUNCTION_HEADER

  magmablas_strsm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);

}
/** \overload
 */
inline void xTRSM(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                  magma_diag_t diag, magma_int_t m, magma_int_t n, D_t alpha,
                  const D_t* A, magma_int_t lda, D_t* B, magma_int_t ldb) {

  PROFILING_FUNCTION_HEADER

  magmablas_dtrsm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);

}
/** \overload
 */
inline void xTRSM(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                  magma_diag_t diag, magma_int_t m, magma_int_t n, C_t alpha,
                  const C_t* A, magma_int_t lda, C_t* B, magma_int_t ldb) {

  PROFILING_FUNCTION_HEADER

  magmablas_ctrsm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);

}
/** \overload
 */
inline void xTRSM(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                  magma_diag_t diag, magma_int_t m, magma_int_t n, Z_t alpha,
                  const Z_t* A, magma_int_t lda, Z_t* B, magma_int_t ldb) {

  PROFILING_FUNCTION_HEADER

  magmablas_ztrsm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);

}

} /* namespace LinAlg::BLAS::MAGMA */
# endif /* HAVE_MAGMA */

#endif /* HAVE_CUDA */

using LinAlg::Utilities::check_dimensions;
using LinAlg::Utilities::check_device;
#ifdef HAVE_CUDA
using LinAlg::Utilities::check_gpu_structures;
#endif

/** \brief            xTRSM
 *
 *  Triangular system solver
 *
 *  A * X = alpha * B
 *  X * A = alpha * B
 *
 *  where B gets overwritten with X
 *
 *  \param[in]        side
 *
 *  \param[in]        uplo
 *
 *  \param[in]        diag
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        B
 */
template <typename T>
inline void xTRSM(Side side, UPLO uplo, Diag diag, const T alpha, 
                  const Dense<T>& A, Dense<T>& B) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_format(Format::ColMajor, A, "xTRSM(), A");
  check_format(Format::ColMajor, B, "xTRSM(), B");
  check_input_transposed(B, "xTRSM()");
  check_device(A, B, "xTRSM()");
  if (side == Side::left) {
    check_dimensions(B.rows(), B.rows(), A, "xTRSM(), A");
  } else {
    check_dimensions(B.cols(), B.cols(), A, "xTRSM(), A");
  }
#endif

  auto m_         = B.rows();
  auto n_         = B.cols();
  auto alpha_     = alpha;
  auto A_         = A.begin();
  auto lda_       = A._leading_dimension;
  auto B_         = B.begin();
  auto ldb_       = B._leading_dimension;

  if (A._location == Location::host) {

    auto side_      = (side == Side::left)  ? 'l' : 'r';
    auto uplo_      = (uplo == UPLO::lower) ? 'l' : 'u';
    auto transa_    = (A._transposed)       ? 't' : 'n';
    //   transa_    = (A._ctransposed)      ? 'c' : transa_;
    auto diag_      = (diag == Diag::unit)  ? 'u' : 'n';
  
    FORTRAN::xTRSM(side_, uplo_, transa_, diag_, m_, n_, alpha_, A_, lda_, B_,
                   ldb_);
  
  }

#ifdef HAVE_CUDA
  else if (A._location == Location::GPU) {

# ifndef LINALG_NO_CHECKS
    check_gpu_structures("xTRSM()");
# endif

# ifndef USE_MAGMA_TRSM

    auto handle_    = CUDA::cuBLAS::handles[A._device_id];
    auto side_      = (side == Side::left)  ? CUBLAS_SIDE_LEFT  :
                                              CUBLAS_SIDE_RIGHT ;
    auto uplo_      = (uplo == UPLO::lower) ? CUBLAS_FILL_MODE_LOWER :
                                              CUBLAS_FILL_MODE_UPPER ;
    auto transa_    = (A._transposed)       ? CUBLAS_OP_T :
                                              CUBLAS_OP_N ;
    //   transa_    = (A._ctransposed)      ? CUBLAS_OP_C :
    //                                        transa_     ;
    auto diag_      = (diag == Diag::unit)  ? CUBLAS_DIAG_UNIT     :
                                              CUBLAS_DIAG_NON_UNIT ;

    BLAS::cuBLAS::xTRSM(handle_, side_, uplo_, transa_, diag_, m_, n_, alpha_, 
                        A_, lda_, B_, ldb_);

# else  /* USE_MAGMA_TRSM */

    auto side_      = (side == Side::left)  ? MagmaLeft  :
                                              MagmaRight ;
    auto uplo_      = (uplo == UPLO::lower) ? MagmaLower :
                                              MagmaUpper ;
    auto transa_    = (A._transposed)       ? MagmaTrans :
                                              MagmaNoTrans ;
    //   transa_    = (A._ctransposed)      ? MagmaConjTrans :
    //                                        transa_        ;
    auto diag_      = (diag == Diag::unit)  ? MagmaUnit    :
                                              MagmaNonUnit ;

    BLAS::MAGMA::xTRSM(side_, uplo_, transa_, diag_, m_, n_, 
                         alpha_, A_, lda_, B_, ldb_);
    
# endif
  
  }
#endif /* HAVE_CUDA */

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xTRSM(): BLAS TRSM not supported on selected "
                           "location");
  }
#endif

}

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_TRSM_H_ */
