/** \file
 *
 *  \brief            Convenience bindings for solving systems
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
 *  \date             Created:  Jan 08, 2015
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_ABSTRACT_SOLVE_H_
#define LINALG_ABSTRACT_SOLVE_H_

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


/////////////////
// Linear solving

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
 *                    the destructor of pivot crashes.
 */
template <typename T>
inline void solve(Dense<T>& A, Dense<T>& B) {

  // Note:
  //  - for the LAPACK and MAGMA GESV backends we need a pivot vector that is
  //    allocated on the CPU
  //  - for the cuBLAS (GETRF + 2*TRSM) implementation the pivot vector is
  //    ignored

  PROFILING_FUNCTION_HEADER

  Dense<int> pivot(A._rows, 1, Location::host, 0);
  LAPACK::xGESV(A, pivot, B);

}

/** \brief            Solve a linear system
 *
 *  A * X = B     (B is overwritten with X, A with its own LU decomposition)
 *
 *  \param[in,out]    A
 *
 *  \param[in]        pivot
 *                    The pivoting vector (not needed anymore after the 
 *                    function call)
 *
 *  \param[in,out]    B
 *
 *  \note             This function is more efficient than the one without the
 *                    pivoting argument when being invoked multiple times as 
 *                    the pivot vector is only allocated and deallocated once.
 */
template <typename T>
inline void solve(Dense<T>& A, Dense<I_t>& pivot, Dense<T>& B) {

  PROFILING_FUNCTION_HEADER

  LAPACK::xGESV(A, pivot, B);

}

} /* namespace LinAlg */

#endif /* LINALG_ABSTRACT_SOLVE_H_ */
