/** \file
 *
 *  \brief            Convenience bindings for multiplication
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
#ifndef LINALG_ABSTRACT_MULTIPLY_H_
#define LINALG_ABSTRACT_MULTIPLY_H_

#include "../preprocessor.h"

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../utilities/utilities.h"
#include "../BLAS/blas.h"      // the bindings to the various BLAS libraries
#include "../dense.h"
#include "../sparse.h"
#include "add.h"

namespace LinAlg {

///////////////////////
// Convenience bindings
//
// These are the bindings for Dense<T> matrices, optionally with streams.
// Argument and error checking is done on lower levels.


/////////////////
// Multiplication

/** \brief            Matrix-matrix multiply with prefactors
 *
 *  C <- alpha * A * B + beta * C
 *
 *  \param[in]        alpha
 *                    OPTIONAL: default = T(1)
 *
 *  \param[in]        A
 *
 *  \param[in]        B
 *
 *  \param[in]        beta
 *                    OPTIONAL: default = T(0)
 *
 *  \param[in,out]    C
 */
template <typename T>
inline void multiply(const T alpha, const Dense<T>& A, const Dense<T>& B,
                     const T beta, Dense<T>& C) {
  BLAS::xGEMM(alpha, A, B, beta, C);
}
/** \overload
 */
template <typename T>
inline void multiply(const Dense<T>& A, const Dense<T>& B, Dense<T>& C) {
  BLAS::xGEMM(cast<T>(1.0), A, B, cast<T>(0.0), C);
}

/** \brief           Mixed type matrix-matrix multiply
 *
 * C <- alpha * A * B + beta * C
 *
 *  \param[in]        alpha
 *                    OPTIONAL: default = 1
 *
 *  \param[in]        A
 *
 *  \param[in]        B
 *
 *  \param[in]        beta
 *                    OPTIONAL: default = 0
 *
 *  \param[in,out]    C
 *
 *  \param[in,out]    work_T
 *                    OPTIONAL: work space of type T, will be overwritten. See 
 *                    below for size requirements. Default: the routine 
 *                    allocates a suitable work space.
 *
 *  \note             Size requirements for work_T:
 *
 *                      if A is of type T, and B and C are of type U:
 *                        size(work_T) >= 2 * size(B) + 2 * size(C)
 *
 *                      if A and C are of type U, and B is of type T:
 *                        size(work_T) >= 2 * size(A) + 2 * size(C)
 */
template <typename T, typename U>
inline void multiply(const T alpha, const Dense<T>& A, const Dense<U>& B,
                     const U beta, Dense<U>& C, Dense<T>& work_T) {

  // This is an awful hack and the same can be found in the next overload

  PROFILING_FUNCTION_HEADER

  using Utilities::check_format;
  using Utilities::check_output_transposed;
  using Utilities::check_input_transposed;

  if (alpha == cast<T>(0.0) && beta == cast<U>(1.0)) return;

#ifndef LINALG_NO_CHECKS
  if (C._is_complex() == false) {
    throw excBadArgument("multiply(mixed_types): C must always be complex");
  } else if (type<T>() == Type::S && type<U>() == Type::Z) {
    throw excBadArgument("multiply(mixed_types): cannot mix precision (A is "
                         "LinAlg::S_t, C is LinAlg::Z_t");
  } else if (type<T>() == Type::D && type<U>() == Type::C) {
    throw excBadArgument("multiply(mixed_types): cannot mix precision (A is "
                         "LinAlg::D_t, C is LinAlg::C_t");
  }
  check_format(Format::ColMajor, A, "multiply(mixed_types), A: matrix must " 
                                    "be Format::ColMajor");
  check_format(Format::ColMajor, B, "multiply(mixed_types), B: matrix must " 
                                    "be Format::ColMajor");
  check_format(Format::ColMajor, C, "multiply(mixed_types), C: matrix must " 
                                    "be Format::ColMajor");
  check_output_transposed(C, "multiply(mixed_types), C");
#endif

  auto B_size = B.rows() * B.cols();
  auto C_size = C.rows() * C.cols();

  auto work_T_size = 2 * B_size + 2 * C_size;
  if (work_T.is_empty()) work_T.reallocate(work_T_size, 1, C._location, 
                                           C._device_id);

#ifndef LINALG_NO_CHECKS
  auto work_T_actual_size = work_T.rows() * work_T.cols();
  if (work_T_actual_size < work_T_size) {
    throw excBadArgument("multiply(alpha, A, B, beta, B, work_T), "
                         "work_T: work_T is too small (is %d elements "
                         " < %d)", work_T_actual_size, work_T_size);
  }
#endif

  auto location  = C._location;
  auto device_id = C._device_id;

  auto work_T_ptr = work_T._begin();

  // Layout in work_T
  //        |B_real|B_imag|C_real|C_imag|
  int start = 0;
  Dense<T> B_real(work_T_ptr + start, B.rows(), B.rows(), B.cols(), location, 
                  device_id);
  start += B_size;
  Dense<T> B_imag(work_T_ptr + start, B.rows(), B.rows(), B.cols(), location, 
                  device_id);
  start += B_size;
  Dense<T> C_real(work_T_ptr + start, C.rows(), C.rows(), C.cols(), location, 
                  device_id);
  start += C_size;
  Dense<T> C_imag(work_T_ptr + start, C.rows(), C.rows(), C.cols(), location, 
                  device_id);

  // Strategy:
  //  split B -> B_real, B_imag
  //  C_real = alpha * A * B_real
  //  C_imag = alpha * A * B_imag
  //  C = C_real + i * C_imag + beta * C

  using Utilities::complex2realimag;
  using Utilities::realimag2complex;
  
  complex2realimag(B, B_real, B_imag);
  if (B._transposed) {
    B_real.transpose();
    B_imag.transpose();
  }
  multiply(A, B_real, C_real);
  multiply(A, B_imag, C_imag);
  realimag2complex(alpha, C_real, C_imag, beta, C);
  
}
/** \overload
 */
template <typename T, typename U>
inline void multiply(const T alpha, const Dense<T>& A, const Dense<U>& B,
                     const U beta, Dense<U>& C) {

  PROFILING_FUNCTION_HEADER

#ifdef HAVE_MKL
  if (C._location == Location::host) {
  
    // We can 'upcast' alpha from real to complex and use MKL's dzgemm instead 
    // of our awful hack
    LinAlg::BLAS::xGEMM(cast<U>(alpha), A, B, beta, C);

  } else {
#endif

    // Pass through to the hack, it will allocate as needed
    Dense<T> work_T;

    multiply(alpha, A, B, beta, C, work_T);

#ifdef HAVE_MKL
  }
#endif


}
/** \overload
 */
template <typename T, typename U>
inline void multiply(const T alpha, const Dense<T>& A, const Dense<U>& B,
                     const T beta, Dense<U>& C) {

  multiply(alpha, A, B, cast<U>(beta), C);

}
/** \overload
 */
template <typename T, typename U>
inline void multiply(const Dense<T>& A, const Dense<U>& B, Dense<U>& C) {

  multiply(cast<T>(1.0), A, B, cast<U>(0.0), C);

}

/** \overload
 */
template <typename T, typename U>
inline void multiply(const T alpha, const Dense<U>& A, const Dense<T>& B,
                     const U beta, Dense<U>& C, Dense<T>& work_T) {

  PROFILING_FUNCTION_HEADER

  using Utilities::check_format;
  using Utilities::check_output_transposed;
  using Utilities::check_input_transposed;

  if (alpha == cast<T>(0.0) && beta == cast<U>(1.0)) return;

#ifndef LINALG_NO_CHECKS
  if (C._is_complex() == false) {
    throw excBadArgument("multiply(mixed_types): C must always be complex");
  } else if (type<T>() == Type::S && type<U>() == Type::Z) {
    throw excBadArgument("multiply(mixed_types): cannot mix precision (A is "
                         "LinAlg::S_t, C is LinAlg::Z_t");
  } else if (type<T>() == Type::D && type<U>() == Type::C) {
    throw excBadArgument("multiply(mixed_types): cannot mix precision (A is "
                         "LinAlg::D_t, C is LinAlg::C_t");
  }
  check_format(Format::ColMajor, A, "multiply(mixed_types), A: matrix must " 
                                    "be Format::ColMajor");
  check_format(Format::ColMajor, B, "multiply(mixed_types), B: matrix must " 
                                    "be Format::ColMajor");
  check_format(Format::ColMajor, C, "multiply(mixed_types), C: matrix must " 
                                    "be Format::ColMajor");
  check_output_transposed(C, "multiply(mixed_types), C");
#endif

  auto A_size = A.rows() * A.cols();
  auto C_size = C.rows() * C.cols();

  auto work_T_size = 2 * A_size + 2 * C_size;
  if (work_T.is_empty()) work_T.reallocate(work_T_size, 1, C._location,
                                           C._device_id);

#ifndef LINALG_NO_CHECKS
  auto work_T_actual_size = work_T.rows() * work_T.cols();
  if (work_T_actual_size < work_T_size) {
    throw excBadArgument("multiply(alpha, A, B, beta, B, work_T), work_T: "
                         "work_T is too small (is %d elements < %d)", 
                         work_T_actual_size, work_T_size);
  }
#endif

  auto location  = C._location;
  auto device_id = C._device_id;

  auto work_T_ptr = work_T._begin();

  // Layout in work_T
  //        |A_real|A_imag|C_real|C_imag|
  int start = 0;
  Dense<T> A_real(work_T_ptr + start, A.rows(), A.rows(), A.cols(), location, 
                  device_id);
  start += A_size;
  Dense<T> A_imag(work_T_ptr + start, A.rows(), A.rows(), A.cols(), location, 
                  device_id);
  start += A_size;
  Dense<T> C_real(work_T_ptr + start, C.rows(), C.rows(), C.cols(), location, 
                  device_id);
  start += C_size;
  Dense<T> C_imag(work_T_ptr + start, C.rows(), C.rows(), C.cols(), location, 
                  device_id);

  // Strategy:
  //  split A -> A_real, A_imag
  //  C_real = alpha * A_real * B
  //  C_imag = alpha * A_imag * B
  //  C = C_real + i * C_imag
  
  using Utilities::complex2realimag;
  using Utilities::realimag2complex;

  complex2realimag(A, A_real, A_imag);
  if (A._transposed) {
    A_real.transpose();
    A_imag.transpose();
  }
  multiply(A_real, B, C_real);
  multiply(A_imag, B, C_imag);
  realimag2complex(alpha, C_real, C_imag, beta, C);

}
/** \overload
 */
template <typename T, typename U>
inline void multiply(const T alpha, const Dense<U>& A, const Dense<T>& B,
                     const U beta, Dense<U>& C) {

  PROFILING_FUNCTION_HEADER

  // Pass through to the hack, it will allocate as needed
  Dense<T> work_T;

  multiply(alpha, A, B, beta, C, work_T);

}
/** \overload
 */
template <typename T, typename U>
inline void multiply(const Dense<U>& A, const Dense<T>& B, Dense<U>& C) {

  multiply(cast<T>(1.0), A, B, cast<U>(0.0), C);

}

} /* namespace LinAlg */

#endif /* LINALG_ABSTRACT_MULTIPLY_H_ */
