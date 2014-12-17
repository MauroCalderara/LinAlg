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

#include "preprocessor.h"
#include "types.h"
#include "profiling.h"
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

/** \brief            Matrix-matrix addition with prefactor
 *
 *  B <- alpha * A + B
 * 
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in,out]    B
 */
template <typename T>
inline void add(T alpha, Dense<T>& A, Dense<T>& B) {

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

/** \brief            Matrix-matrix addition
 *
 *  B <- A + B
 * 
 *  \param[in]        A
 *
 *  \param[in,out]    B
 */
template <typename T>
inline void add(Dense<T>& A, Dense<T>& B) {

  add(cast<T>(1.0), A, B);

}

/** \brief            Matrix-matrix multiply
 *
 *  C <- A * B
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
 *  C <- alpha * A * B + beta * C
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

  PROFILING_FUNCTION_HEADER

  Utilities::Timer gemm_timer("xGEMM");

  BLAS::xGEMM(alpha, A, B, beta, C);

  gemm_timer.toc();

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

  Utilities::Timer gesv_timer("xGESV");

  LAPACK::xGESV(A, pivot, B);

  gesv_timer.toc();

}


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

#endif /* LINALG_ABSTRACT_H_ */
