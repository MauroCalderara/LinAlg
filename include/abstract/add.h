/** \file
 *
 *  \brief            Convenience bindings for addition
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
#ifndef LINALG_ABSTRACT_ADD_H_
#define LINALG_ABSTRACT_ADD_H_

#include "../preprocessor.h"

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../utilities/utilities.h"
#include "../BLAS/blas.h"      // the bindings to the various BLAS libraries
#include "../dense.h"
#include "../sparse.h"

namespace LinAlg {

///////////////////////
// Convenience bindings
//
// These are the bindings for Dense<T> matrices, optionally with streams.
// Argument and error checking is done on lower levels.

///////////
// Addition

/** \brief            Matrix-matrix addition with prefactor
 *
 *  B <- alpha * A + B
 * 
 *  \param[in]        alpha
 *                    OPTIONAL: default = T(1)
 *
 *  \param[in]        A
 *
 *  \param[in,out]    B
 */
template <typename T>
inline void add(T alpha, const Dense<T>& A, Dense<T>& B) {

  PROFILING_FUNCTION_HEADER

  using Utilities::check_format;
  using Utilities::check_input_transposed;
  using Utilities::check_output_transposed;
  using Utilities::check_same_dimensions;
  using Utilities::check_device;

#ifndef LINALG_NO_CHECKS
  // Currently only ColMajor is supported
  check_format(Format::ColMajor, A, "add(alpha, A, B), A [only ColMajor is "
               "supported]");
  check_format(Format::ColMajor, B, "add(alpha, A, B), B [only ColMajor is "
               "supported]");
  check_input_transposed(A, "add(alpha, A, B), A");
  check_output_transposed(B, "add(alpha, A, B), B");
  check_same_dimensions(A, B, "add(alpha, A, B), A, B");
  check_device(A, B, "add(alpha, A, B), A, B");
#endif

  if (A._location == Location::host) {

    using BLAS::FORTRAN::xAXPY;

    auto x_ptr = A._begin();
    auto incx = 1;
    auto y_ptr = B._begin();
    auto incy = 1;
  
    if (A._rows == A._leading_dimension && B._rows == B._leading_dimension) {

      // Matrices continuous in memory, use one xAXPY call:
      auto n = A._rows * A._cols;
      xAXPY(n, alpha, x_ptr, incx, y_ptr, incy);

    } else {

      // At least one matrix not continuous in memory, make one call per 
      // column
      auto rows = A._rows;
      auto lda = A._leading_dimension;
      auto ldb = B._leading_dimension;

      for (I_t col = 0; col < A._cols; ++col) {

        xAXPY(rows, alpha, x_ptr + col * lda, incx, y_ptr + col * ldb, incy);

      }
    
    }
  
  }

#ifdef HAVE_CUDA
  else if (A._location == Location::GPU) {
  
    // B = alpha * A + 1.0 * B (see cuBLAS GEAM documentation under 'in-place 
    // mode')
    BLAS::xGEAM(alpha, A, cast<T>(1.0), B, B);
  
  }
#endif /* HAVE_CUDA */

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("add(): matrix addition on engine not "
                           "implemented");
  }
#endif

}
/** \overload
 */
template <typename T>
inline void add(Dense<T>& A, Dense<T>& B) {

  add(cast<T>(1.0), A, B);

}

} /* namespace LinAlg */

#endif /* LINALG_ABSTRACT_ADD_H_ */
