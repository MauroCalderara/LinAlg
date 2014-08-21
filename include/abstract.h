/** \file
 *
 *  \brief            Convenience bindings like multiply(), inv() and solve()
 *
 *  Organization of the namespace:
 *
 *    LinAlg
 *        functions like 'solve' and 'multiply' (abstract.h)
 *
 *    LinAlg::BLAS
 *
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::BLAS::\<backend\>
 *
 *        bindings for the backend
 *
 *    LinAlg::LAPACK
 *
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::LAPACK::\<backend\>
 *
 *        bindings for the backend
 *
 *
 *  \date             Created:  Jul 23, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_ABSTRACT_H_
#define LINALG_ABSTRACT_H_

#include "types.h"
#include "exceptions.h"
#include "utilities/utilities.h"
#include "BLAS/blas.h"      // the bindings to the various BLAS libraries
#include "LAPACK/lapack.h"  // same for LAPACK
#include "dense.h"
#include "sparse.h"

namespace LinAlg {

///////////////////////
// Convenience bindings
//
// These are the bindings for Dense<T> matrices, optionally with streams.
// Argument and error checking is done on lower levels.

/** \brief            Matrix-matrix multiply
 *
 *  C = A * B
 *
 *  \param[in]        A
 *
 *  \param[in]        B
 *
 *  \param[in,out]    C
 */
template <typename T>
inline void multiply(const Dense<T>& A, const Dense<T>& B, Dense<T>& C) {
  BLAS::xGEMM(cast<T>(1.0), A, B, cast<T>(0.0), C);
}

/** \brief            Matrix-matrix multiply with prefactors
 *
 *  C = alpha * A * B + beta * C
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
inline void multiply(const T alpha, const Dense<T>& A, const Dense<T>& B,
                     const T beta, Dense<T>& C) {
  BLAS::xGEMM(alpha, A, B, beta, C);
}


/** \brief            Solve a linear system
 *
 *  A * X = B     (B is overwritten with X, A with its own LU decomposition)
 *
 *  \param[in,out]    A
 *
 *  \param[in,out]    B
 *
 *  \param[in,out]    ipiv
 *                    OPTIONAL: pivoting vector. If unspecified or empty, the
 *                    routine allocates one with suitable size.
 *
 *  \todo             Actually one should check if B is a vector and call xTRSV
 *                    instead of xTRSM in the CUDA code.
 */
template <typename T>
inline void solve(Dense<T>& A, Dense<T>& B, Dense<int>& ipiv) {

  auto location = A._location;
  auto n        = A.rows();

  if (location == Location::host) {

    if (ipiv._rows == 0) {

      ipiv.reallocate(n, 1);

    }

    LAPACK::xGESV(A, ipiv, B);

  }
#ifdef HAVE_CUDA
  else if (location == Location::GPU) {

#ifndef USE_MAGMA_GESV
    // This is basically xGESV done by hand. We use the 'expert' interface
    // in LAPACK::CUBLAS::xGETRF directly instead of LAPACK::xGETRF

#ifndef LINALG_NO_CHECKS
    BLAS::check_input_transposed(A, "solve() (even though A^T could be "
                                 "implemented)");
    BLAS::check_input_transposed(B, "solve()");
#endif

    auto handle = CUDA::CUBLAS::handles[A._device_id];
    auto A_ptr = A._begin();
    auto lda = A._leading_dimension;
    auto B_ptr = B._begin();
    auto ldb = B._leading_dimension;
    auto ipiv_override = nullptr;
    int  info = 0;

    // LU decompose using xGETRF without pivoting (use ipiv_override == empty
    // vector to enforce no pivoting. cudaXtrsm doesn't support pivoting)
    LAPACK::CUBLAS::xGETRF(handle, n, A_ptr, lda, ipiv_override, &info);
#ifndef LINALG_NO_CHECKS
    if (info != 0) {
      throw excMath("solve(): unable to LU decompose A (xGETRF(): error = %d)",
                    info);
    }
#endif

    // Directly solve using xTRSM (no xLASWP since we didn't pivot):
    // 1: y = L\b
    auto nrhs = B.cols();
    BLAS::CUBLAS::xTRSM(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, nrhs, T(1.0), A_ptr,
                        lda, B_ptr, ldb);
    // 2: x = U\y
    BLAS::CUBLAS::xTRSM(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, nrhs, T(1.0),
                        A_ptr, lda, B_ptr, ldb);
#endif /* not USE_MAGMA_GESV */
  }
#endif /* HAVE_CUDA */

};
/** \overload
 *
 *  \param[in,out]    A
 *
 *  \param[in,out]    B
 */
template <typename T>
inline void solve(Dense<T>& A, Dense<T>& B) {
  Dense<int> ipiv;
  solve(A, B, ipiv);
}


/** \brief            Invert a matrix in-place or out-of-place
 *
 *  A = A**-1 (in-place)
 *  C = A**-1 (out-of-place)
 *
 *  \param[in,out]    A
 *
 *  \param[in,out]    C
 *                    OPTIONAL: Storage for out-of-place inversion. If left
 *                    unspecified or empty, an in-place inversion is performed.
 */
template <typename T>
inline void invert(Dense<T>& A, Dense<T>& C) {

  Dense<int> ipiv(A._rows, 1, A._location, A._device_id);
  Dense<T> work;

  LAPACK::xGETRF(A, ipiv);
  LAPACK::xGETRI(A, ipiv, work, C);

};
/** \overload
 *
 *  \param[in,out]    A
 */
template <typename T>
inline void invert(Dense<T>& A) {
  Dense<T> C;
  invert(A, C);
}


} /* namespace LinAlg */

#endif /* LINALG_ABSTRACT_H_ */
