/** \file
 *
 *  \brief            Convenience for inversion
 *
 *  Organization of the namespace:
 *
 *    LinAlg
 *        functions like 'solve' and 'multiply' (abstract/\*.h)
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
 *  \date             Created:  Jan 08, 2015
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_ABSTRACT_INVERT_H_
#define LINALG_ABSTRACT_INVERT_H_

#include "../preprocessor.h"

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../utilities/utilities.h"
#include "../LAPACK/lapack.h"  // same for LAPACK
#include "../dense.h"
#include "../sparse.h"

namespace LinAlg {

///////////////////////
// Convenience bindings
//
// These are the bindings for Dense<T> matrices, optionally with streams.
// Argument and error checking is done on lower levels.

/** \brief            Invert a matrix in-place
 *
 *  A <- A**-1 (in-place)
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

  PROFILING_FUNCTION_HEADER

  // Note: MAGMA needs ipiv to be on the host
#ifdef USE_MAGMA_GETRF
  Dense<int> pivot(A._rows, 1, Location::host, 0);
#else
  Dense<int> pivot(A._rows, 1, A._location, A._device_id);
#endif
  Dense<T>   work;

  // We assume that pivoting factorization is faster for all backends 
  // (including those that support non-pivoting factorization like cuBLAS)
  LAPACK::xGETRF(A, pivot);
  LAPACK::xGETRI(A, pivot, work);

}

/** \brief            Invert a matrix in-place
 *
 *  A <- A**-1 (in-place)
 *
 *  \param[in,out]    A
 *
 *  \param[in]        pivot
 *                    The pivoting vector (not needed anymore after the 
 *                    function call)
 *
 *  \note             This function is more efficient than the one without the
 *                    pivoting argument when being invoked multiple times as 
 *                    the pivot vector is only allocated and deallocated once.
 */
template <typename T>
inline void invert(Dense<T>& A, Dense<int>& pivot) {

  PROFILING_FUNCTION_HEADER

  Dense<T>   work;

  // We assume that pivoting factorization is faster for all backends 
  // (including those that support non-pivoting factorization like cuBLAS)
  LAPACK::xGETRF(A, pivot);
  LAPACK::xGETRI(A, pivot, work);

}

/** \brief            Invert a matrix out-of-place
 *
 *  C <- A**-1 (out-of-place)
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

  PROFILING_FUNCTION_HEADER

  // Note: MAGMA needs ipiv to be on the host
#ifdef USE_MAGMA_GETRF
  Dense<int> pivot(A._rows, 1, Location::host, 0);
#else
  Dense<int> pivot(A._rows, 1, A._location, A._device_id);
#endif
  Dense<T>   work;

  LAPACK::xGETRF(A, pivot);
  LAPACK::xGETRI(A, pivot, work, C);

}

/** \brief            Invert a matrix out-of-place
 *
 *  C <- A**-1 (out-of-place)
 *
 *  \param[in,out]    A
 *                    On return, A is replaced by its LU decomposition.
 *
 *  \param[in]        pivot
 *                    The pivoting vector (not needed anymore after the 
 *                    function call)
 *
 *  \param[in,out]    C
 *                    Target for out-of-place inversion. If empty, the routine 
 *                    allocates suitable memory.
 *
 *  \note             This function is more efficient than the one without the
 *                    pivoting argument when being invoked multiple times as 
 *                    the pivot vector is only allocated and deallocated once.
 *                    Out-of-place inversions are fast when using the cuBLAS 
 *                    backend and generally slow otherwise.
 */
template <typename T>
inline void invert(Dense<T>& A, Dense<int>& pivot, Dense<T>& C) {

  PROFILING_FUNCTION_HEADER

  Dense<T>   work;

  LAPACK::xGETRF(A, pivot);
  LAPACK::xGETRI(A, pivot, work, C);

}

} /* namespace LinAlg */

#endif /* LINALG_ABSTRACT_INVERT_H_ */
