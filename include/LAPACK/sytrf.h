/** \file
 *
 *  \brief            xSYTRF
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LAPACK_SYTRF_H_
#define LINALG_LAPACK_SYTRF_H_

/* Organization of the namespace:
 *
 *    LinAlg::LAPACK
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::LAPACK::<NAME>
 *        bindings to the <NAME> LAPACK backend
 */

MARKER: this is work in progress

#include "../preprocessor.h"

#ifdef HAVE_CUDA
# ifdef HAVE_MAGMA
#   include <magma.h>
# endif
#endif

#include "../types.h"
#include "../profiling.h"
#include "../exceptions.h"
#include "../utilities/checks.h"
#include "../dense.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  void fortran_name(ssytrf, SSYTRF)(const I_t* m, const I_t* n, S_t* A,
                                    const I_t* lda, I_t* ipiv, int* info);
  void fortran_name(dsytrf, DSYTRF)(const I_t* m, const I_t* n, D_t* A,
                                    const I_t* lda, I_t* ipiv, int* info);
  void fortran_name(csytrf, CSYTRF)(const I_t* m, const I_t* n, C_t* A,
                                    const I_t* lda, I_t* ipiv, int* info);
  void fortran_name(zsytrf, ZSYTRF)(const I_t* m, const I_t* n, Z_t* A,
                                    const I_t* lda, I_t* ipiv, int* info);
}
#endif

namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

/** \brief            Compute LU factorization
 *
 *  A <- P * L * U
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    ipiv
 *
 *  \param[in,out]    info
 *
 *  See
 *  [DSYTRF](http://www.math.utah.edu/software/lapack/lapack-d/dsytrf.html)
 */
inline void xSYTRF(I_t m, I_t n, S_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  fortran_name(ssytrf, SSYTRF)(&m, &n, A, &lda, ipiv, info);

}
/** \overload
 */
inline void xSYTRF(I_t m, I_t n, D_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  fortran_name(dsytrf, DSYTRF)(&m, &n, A, &lda, ipiv, info);

}
/** \overload
 */
inline void xSYTRF(I_t m, I_t n, C_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  fortran_name(csytrf, CSYTRF)(&m, &n, A, &lda, ipiv, info);

}
/** \overload
 */
inline void xSYTRF(I_t m, I_t n, Z_t* A, I_t lda, I_t* ipiv, int* info) {

  PROFILING_FUNCTION_HEADER

  fortran_name(zsytrf, ZSYTRF)(&m, &n, A, &lda, ipiv, info);

}

} /* namespace FORTRAN */


#ifdef HAVE_CUDA

# ifdef HAVE_MAGMA
namespace MAGMA {

/** \brief            Compute LDL factorization for symmetric matrices 
 *                    (non-pivoting version)
 *
 *  A <- L * D * Lt
 *
 *  \param[in]        uplo
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A (resides on the GPU)
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    info
 *
 *  See MAGMA testing/testing_dsytrf.cpp
 */
inline void xSYTRF_nopiv(magma_uplo_t uplo, I_t n, S_t* A, I_t lda, int* info) {

  PROFILING_FUNCTION_HEADER

  magma_ssytrf_nopiv_gpu(uplo, n, A, lda, info);

}
/** \overload
 */
inline void xSYTRF_nopiv(magma_uplo_t uplo, I_t n, D_t* A, I_t lda, int* info) {

  PROFILING_FUNCTION_HEADER

  magma_dsytrf_nopiv_gpu(uplo, n, A, lda, info);

}
/* C_t and Z_t functions not available in Magma at the time of this writing */

/** \brief            Compute LDL factorization for symmetric matrices 
 *
 *  A <- P * L * D * Lt
 *
 *  \param[in]        uplo
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A (resides on the GPU)
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    ipiv (resides on the CPU)
 *
 *  \param[in,out]    info
 *
 *  See MAGMA testing/testing_dsytrf.cpp
 */
inline void xSYTRF(magma_uplo_t uplo, I_t n, S_t* A, I_t lda, I_t* ipiv, 
                   int* info) {

  PROFILING_FUNCTION_HEADER

  magma_ssytrf(m, n, A, lda, ipiv, info);

}
/** \overload
 */
inline void xSYTRF(magma_uplo_t uplo, I_t n, S_t* A, I_t lda, I_t* ipiv, 
                   int* info) {

  PROFILING_FUNCTION_HEADER

  magma_ssytrf(m, n, A, lda, ipiv, info);

}
/* C_t and Z_t functions not available in Magma at the time of this writing */

} /* namespace LinAlg::LAPACK::MAGMA */
# endif /* HAVE_MAGMA */

#endif /* HAVE_CUDA */


using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_input_transposed;
#ifdef HAVE_CUDA
using LinAlg::Utilities::check_gpu_structures;
#endif

/** \brief            Compute the LDLt decomposition of a symmetric matrix
 *
 *  A = P * L * U     A is overwritten with L and U
 *
 *  \param[in]        A
 *
 *  \param[in]        ipiv
 *                    ipiv is A.rows()*1 matrix (a vector). When using the
 *                    GPU (MAGMA) backend and specifying an empty matrix for 
 *                    ipiv, the routine performs a non-pivoting LU 
 *                    decomposition.
 *
 *  \note             The return value of the routine is checked and a
 *                    corresponding exception is thrown if the matrix is
 *                    singular.
 */
template <typename T>
inline void xSYTRF(Dense<T>& A, Dense<int>& ipiv) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS

  check_input_transposed(A, "xSYTRF(A, ipiv), A:");
  if (!ipiv.is_empty()) {
    check_input_transposed(ipiv, "xSYTRF(A, ipiv), ipiv:");
  }
  if (!A.is(Property::symmetric)) {
    throw excBadArgument("xSYTRF(A, ipiv), A: matrix A must be symmetric");
  }

# ifndef HAVE_CUDA
  if (A.rows() != ipiv.rows()) {
    throw excBadArgument("xSYTRF(A, ipiv), A, ipiv: must have same number of "
                         "rows");
  }
# else
  if (A.rows() != A.cols() && A._location == Location::GPU) {
    throw excBadArgument("xSYTRF(A, ipiv), A: matrix A must be a square matrix "
                         "(cuBLAS restriction)");
  }
  if (!ipiv.is_empty()) {
    if (A.rows() != ipiv.rows()) {
      throw excBadArgument("xSYTRF(A, ipiv): A, ipiv: must have same number "
                           "of rows");
    }
  }
# endif

#endif /* LINALG_NO_CHECKS */

  auto n = A.cols();
  auto A_ptr = A._begin();
  auto lda = A._leading_dimension;
  int  info = 0;

  if (A._location == Location::host) {

#ifndef LINALG_NO_CHECKS
    check_device(A, ipiv, "xSYTRF(A, ipiv)");
#endif

    auto m = A.cols();
    auto ipiv_ptr = ipiv._begin();
    FORTRAN::xSYTRF(m, n, A_ptr, lda, ipiv_ptr, &info);

  }

#ifdef HAVE_CUDA
  else if (A._location == Location::GPU) {

# ifndef LINALG_NO_CHECKS
    check_gpu_structures("xSYTRF()");
# endif

# ifndef USE_MAGMA_SYTRF

#   ifndef LINALG_NO_CHECKS
    if (!ipiv.is_empty()) {
      check_device(A, ipiv, "xSYTRF(A, ipiv)");
    }
#   endif

    using LinAlg::CUDA::cuBLAS::handles;
    auto device_id = A._device_id;
    int* ipiv_ptr = (ipiv.is_empty()) ? NULL : ipiv._begin();

    cuBLAS::xSYTRF(handles[device_id], n, A_ptr, lda, ipiv_ptr, info);

# else /* USE_MAGMA_SYTRF */

#   ifndef LINALG_NO_CHECKS
    if (ipiv.location() != Location::host) {
      throw excBadArgument("xSYTRF(A, ipiv): ipiv must be allocated in main "
                           "memory (using MAGMA's sytrf)");
    }
    if (ipiv.is_empty()) {
      throw excBadArgument("xSYTRF(A, ipiv): ipiv must not be empty (using "
                           "MAGMA's sytrf)");
    }
#   endif

    auto m = A.cols();
    auto ipiv_ptr = ipiv._begin();
    MAGMA::xSYTRF(m, n, A_ptr, lda, ipiv_ptr, &info);

# endif /* not USE_MAGMA_SYTRF */

  }
#endif /* HAVE_CUDA */

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xSYTRF(): LAPACK SYTRF not supported on selected "
                           "location");
  }

  if (info != 0) {
    throw excMath("xSYTRF(): error: info = %d", info);
  }
#endif

}

} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_SYTRF_H_ */
