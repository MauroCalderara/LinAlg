/** \file             laset.h
 *
 *  \brief            xLASET
 *
 *  \date             Created:  Sep 26, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LAPACK_LASET_H_
#define LINALG_LAPACK_LASET_H_

/* Organization of the namespace:
 *
 *    LinAlg::LAPACK
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::LAPACK::<NAME>
 *        bindings to the <NAME> LAPACK backend
 */

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../CUDA/cuda_checks.h"

#ifdef HAVE_MAGMA
#include <magma.h>
#endif

#endif


#include "../types.h"
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

  void fortran_name(slaset, SLASET)(const char* uplo, const I_t* m, 
                                    const I_t* n, const S_t* alpha, 
                                    const S_t* beta, S_t* A, const I_t* lda);
  void fortran_name(dlaset, DLASET)(const char* uplo, const I_t* m, 
                                    const I_t* n, const D_t* alpha, 
                                    const D_t* beta, D_t* A, const I_t* lda);
  void fortran_name(claset, CLASET)(const char* uplo, const I_t* m, 
                                    const I_t* n, const C_t* alpha, 
                                    const C_t* beta, C_t* A, const I_t* lda);
  void fortran_name(zlaset, ZLASET)(const char* uplo, const I_t* m, 
                                    const I_t* n, const Z_t* alpha, 
                                    const Z_t* beta, Z_t* A, const I_t* lda);
}
#endif

namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

/** \brief            xLASET
 *
 *  \param[in]        uplo
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in]        alpha
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    A
 *
 *  \param[in]        uplo
 *
 *  See [DLASET](http://www.mathkeisan.com/usersguide/man/dlaset.html)
 */
inline void xLASET(const char* uplo, const I_t m, const I_t n, const S_t alpha,
                   const S_t beta, S_t* A, const I_t lda) {
  fortran_name(slaset, SLASET)(uplo, &m, &n, &alpha, &beta, A, &lda);
}
/** \overload
 */
inline void xLASET(const char* uplo, const I_t m, const I_t n, const D_t alpha,
                   const D_t beta, D_t* A, const I_t lda) {
  fortran_name(dlaset, DLASET)(uplo, &m, &n, &alpha, &beta, A, &lda);
}
/** \overload
 */
inline void xLASET(const char* uplo, const I_t m, const I_t n, const C_t alpha,
                   const C_t beta, C_t* A, const I_t lda) {
  fortran_name(claset, CLASET)(uplo, &m, &n, &alpha, &beta, A, &lda);
}
/** \overload
 */
inline void xLASET(const char* uplo, const I_t m, const I_t n, const Z_t alpha,
                   const Z_t beta, Z_t* A, const I_t lda) {
  fortran_name(zlaset, ZLASET)(uplo, &m, &n, &alpha, &beta, A, &lda);
}

} /* namespace FORTRAN */


#ifdef HAVE_CUDA
#ifdef HAVE_MAGMA
namespace MAGMA {

/** \brief            xLASET
 *
 *  \param[in]        uplo
 *
 *  \param[in]        m
 *
 *  \param[in]        n
 *
 *  \param[in]        alpha
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    A
 *
 *  \param[in]        uplo
 *
 *  See [DLASET](http://www.mathkeisan.com/usersguide/man/dlaset.html)
 */
inline void xLASET(magma_uplo_t uplo, I_t m, I_t n, S_t alpha, S_t beta, S_t* A,
                   I_t lda) {
  magmablas_slaset(uplo, m, n, alpha, beta, A, lda);
}
/** \overload
 */
inline void xLASET(magma_uplo_t uplo, I_t m, I_t n, D_t alpha, D_t beta, D_t* A,
                   I_t lda) {
  magmablas_dlaset(uplo, m, n, alpha, beta, A, lda);
}
/** \overload
 */
inline void xLASET(magma_uplo_t uplo, I_t m, I_t n, C_t alpha, C_t beta, C_t* A,
                   I_t lda) {
  magmablas_claset(uplo, m, n, alpha, beta, A, lda);
}
/** \overload
 */
inline void xLASET(magma_uplo_t uplo, I_t m, I_t n, Z_t alpha, Z_t beta, Z_t* A,
                   I_t lda) {
  magmablas_zlaset(uplo, m, n, alpha, beta, A, lda);
}
    
} /* namespace LinAlg::LAPACK::MAGMA */
#endif /* HAVE_MAGMA */
#endif

using LinAlg::Utilities::check_device;
using LinAlg::Utilities::check_input_transposed;

/** \brief            xLASET
 *
 *  \param[in]        alpha
 *
 *  \param[in]        beta
 *
 *  \param[in,out]    A
 *
 */
template <typename T>
inline void xLASET(T alpha, T beta, Dense<T>& A) {

#ifndef LINALG_NO_CHECKS
  check_input_transposed(A, "xLASET(alpha, beta, A), A:");
#endif

  if (A._location == Location::host) {
    auto uplo = "A";
    auto m = A._rows;
    auto n = A._cols;
    auto A_ptr = A._begin();
    auto lda = A._leading_dimension;
    FORTRAN::xLASET(uplo, m, n, alpha, beta, A_ptr, lda);
  }

#ifdef HAVE_CUDA
#ifdef HAVE_MAGMA
  else if (A._location == Location::GPU) {
    // TODO:
    // Don't know how to set magma_uplo_t to 'all' so we do both upper and 
    // lower :-/
    auto m = A._rows;
    auto n = A._cols;
    auto A_ptr = A._begin();
    auto lda = A._leading_dimension;
    MAGMA::xLASET(MagmaUpper, m, n, alpha, beta, A_ptr, lda);
    MAGMA::xLASET(MagmaLower, m, n, alpha, beta, A_ptr, lda);
  }
#endif
#endif

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xLASET(): LAPACK LASET not supported on selected "
                           "location");
  }
#endif

}

} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_LASET_H_ */
