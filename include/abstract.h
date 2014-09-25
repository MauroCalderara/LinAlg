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
 *  \note             This function is provided as a convenience for one-off
 *                    linear system solving. When calling solve() multiple 
 *                    times it is generally more efficient to avoid the 
 *                    internal allocation/deallocation of the pivoting vector 
 *                    by allocating it separately and directly calling xGESV() 
 *                    multiple times, reusing the same pivot vector.
 *
 *  \todo             This routine fails when using the GPU because somehow
 *                    the destructor of ipiv crashes.
 */
template <typename T>
inline void solve(Dense<T>& A, Dense<T>& B) {

  Dense<int> ipiv(A._rows, 1, A._location, A._device_id);
  LAPACK::xGESV(A, ipiv, B);

}


/** \brief            Invert a matrix in-place
 *
 *  A = A**-1 (in-place)
 *
 *  \param[in,out]    A
 *
 *  \note             - This function is provided as a convenience for one-off
 *                      inversions. When performing multiple inversions it is 
 *                      generally more efficient to avoid the internal 
 *                      allocation and deallocation of the pivot and work 
 *                      vectors by allocating them separately and directly 
 *                      calling xGETRF+xGETRI multiple times, reusing the same 
 *                      the vectors.
 *                    - In-place inversions are generally fast compared to
 *                      out-of-place inversion. The exception are inversions 
 *                      on the GPU, which are slightly slower than their 
 *                      in-place counterparts.
 */
template <typename T>
inline void invert(Dense<T>& A) {

  Dense<int> ipiv(A._rows, 1, A._location, A._device_id);
  Dense<T>   work;

  // We assume that pivoting factorization is faster for all backends 
  // (including those that support non-pivoting factorization like CUBLAS)
  LAPACK::xGETRF(A, ipiv);
  LAPACK::xGETRI(A, ipiv, work);

};

/** \brief            Invert a matrix out-of-place
 *
 *  C = A**-1 (out-of-place)
 *
 *  \param[in,out]    A
 *                    On return, A is replaced by its LU decomposition.
 *
 *  \param[in,out]    C
 *                    Target for out-of-place inversion. If empty, the routine 
 *                    allocates suitable memory.
 *
 *  \note             - This function is provided as a convenience for one-off
 *                      inversions. When performing multiple inversions it is 
 *                      generally more efficient to avoid the internal 
 *                      allocation and deallocation of the pivot and work 
 *                      vectors by allocating them separately and directly 
 *                      calling xGETRF+xGETRI multiple times, reusing the same 
 *                      the vectors.
 *                    - Out-of-place inversions are fast on the GPU and
 *                      generally slow otherwise.
 */
template <typename T>
inline void invert(Dense<T>& A, Dense<T>& C) {

  Dense<int> ipiv(A._rows, 1, A._location, A._device_id);
  Dense<T>   work;

  LAPACK::xGETRF(A, ipiv);
  LAPACK::xGETRI(A, ipiv, work, C);

};

} /* namespace LinAlg */

#endif /* LINALG_ABSTRACT_H_ */
