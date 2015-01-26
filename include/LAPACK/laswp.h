/** \file
 *
 *  \brief            xLASWP (LAPACK)
 *
 *  \date             Created:  Nov 26 , 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LAPACK_LASWP_H_
#define LINALG_LAPACK_LASWP_H_

/* Organization of the namespace:
 *
 *    LinAlg::LAPACK
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::LAPACK::<NAME>
 *        bindings to the <NAME> LAPACK backend
 */

#include "../preprocessor.h"

#ifdef HAVE_CUDA
# ifdef HAVE_MAGMA
#  include <magma.h>
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

  void fortran_name(slaswp, SLASWP)(const I_t* n, S_t* A, const I_t* lda,
                                    const I_t* k1, const I_t* k2, 
                                    const I_t* ipiv, const I_t* incx);
  void fortran_name(dlaswp, DLASWP)(const I_t* n, D_t* A, const I_t* lda,
                                    const I_t* k1, const I_t* k2, 
                                    const I_t* ipiv, const I_t* incx);
  void fortran_name(claswp, CLASWP)(const I_t* n, C_t* A, const I_t* lda,
                                    const I_t* k1, const I_t* k2, 
                                    const I_t* ipiv, const I_t* incx);
  void fortran_name(zlaswp, ZLASWP)(const I_t* n, Z_t* A, const I_t* lda,
                                    const I_t* k1, const I_t* k2, 
                                    const I_t* ipiv, const I_t* incx);
}
#endif


namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

/** \brief            Perform a series of row interchanges on a matrix
 *
 *  \param[in]        n
 *
 *  \param[in,out]    A
 *
 *  \param[in]        lda
 *
 *  \param[in]        k1
 *
 *  \param[in]        k2
 *
 *  \param[in]        ipiv
 *
 *  \param[in]        incx
 *
 *  See [DLASWP](http://www.math.utah.edu/software/lapack/lapack-d/dlaswp.html)
 */
inline void xLASWP(const I_t n, S_t* A, const I_t lda, const I_t k1, 
                   const I_t k2, const I_t* ipiv, const I_t incx) {

  PROFILING_FUNCTION_HEADER

  fortran_name(slaswp, SLASWP)(&n, A, &lda, &k1, &k2, ipiv, &incx);

}
/** \overload
 */
inline void xLASWP(const I_t n, D_t* A, const I_t lda, const I_t k1, 
                   const I_t k2, const I_t* ipiv, const I_t incx) {

  PROFILING_FUNCTION_HEADER

  fortran_name(dlaswp, DLASWP)(&n, A, &lda, &k1, &k2, ipiv, &incx);

}
/** \overload
 */
inline void xLASWP(const I_t n, C_t* A, const I_t lda, const I_t k1, 
                   const I_t k2, const I_t* ipiv, const I_t incx) {

  PROFILING_FUNCTION_HEADER

  fortran_name(claswp, CLASWP)(&n, A, &lda, &k1, &k2, ipiv, &incx);

}
/** \overload
 */
inline void xLASWP(const I_t n, Z_t* A, const I_t lda, const I_t k1, 
                   const I_t k2, const I_t* ipiv, const I_t incx) {

  PROFILING_FUNCTION_HEADER

  fortran_name(zlaswp, ZLASWP)(&n, A, &lda, &k1, &k2, ipiv, &incx);

}

} /* namespace FORTRAN */

#ifdef HAVE_CUDA

# ifdef HAVE_MAGMA
namespace MAGMA {

/** \brief            Perform a series of row interchanges on a matrix
 *
 *  \param[in]        n
 *
 *  \param[in,out]    Atrans (Row-wise storage for MAGMA. Resides on the GPU)
 *
 *  \param[in]        lda
 *
 *  \param[in]        k1
 *
 *  \param[in]        k2
 *
 *  \param[in]        ipiv (resides on the CPU)
 *
 *  \param[in]        incx
 *
 *  See [DLASWP](http://www.math.utah.edu/software/lapack/lapack-d/dlaswp.html)
 *  or MAGMA sources
 */
inline void xLASWP(const I_t n, S_t* Atrans, const I_t lda, const I_t k1, 
                   const I_t k2, const I_t* ipiv, const I_t incx) {

  PROFILING_FUNCTION_HEADER

  magmablas_slaswp(n, Atrans, lda, k1, k2, ipiv, incx);

}
/** \overload
 */
inline void xLASWP(const I_t n, D_t* Atrans, const I_t lda, const I_t k1, 
                   const I_t k2, const I_t* ipiv, const I_t incx) {

  PROFILING_FUNCTION_HEADER

  magmablas_dlaswp(n, Atrans, lda, k1, k2, ipiv, incx);

}
/** \overload
 */
inline void xLASWP(const I_t n, C_t* Atrans, const I_t lda, const I_t k1, 
                   const I_t k2, const I_t* ipiv, const I_t incx) {

  PROFILING_FUNCTION_HEADER

  magmablas_claswp(n, Atrans, lda, k1, k2, ipiv, incx);

}
/** \overload
 */
inline void xLASWP(const I_t n, Z_t* Atrans, const I_t lda, const I_t k1, 
                   const I_t k2, const I_t* ipiv, const I_t incx) {

  PROFILING_FUNCTION_HEADER

  magmablas_zlaswp(n, Atrans, lda, k1, k2, ipiv, incx);

}

# endif /* HAVE_MAGMA */

#endif /* HAVE_CUDA */

} /* namespace MAGMA */

using LinAlg::Utilities::check_input_transposed;
#ifdef HAVE_CUDA
# ifdef HAVE_MAGMA
using LinAlg::Utilities::check_gpu_structures;
# endif
#endif

// Convenience bindings (bindings for Dense<T>)
/** \brief            Perform a series of row interchanges on a matrix A 
 *                    according to the permutation stored in ipiv
 *
 *  \param[in,out]    A 
 *
 *  \param[in]        ipiv
 *
 *  \note             'partial' pivoting is not supported, that is ipiv.rows() 
 *                    must equal A.rows()
 */
template <typename T>
inline void xLASWP(Dense<T>& A, Dense<int>& ipiv) {

  PROFILING_FUNCTION_HEADER

#ifndef LINALG_NO_CHECKS
  check_format(Format::ColMajor, A, "xLASWP(A, ipiv), A");
  check_format(Format::ColMajor, ipiv, "xLASWP(A, ipiv), ipiv");
  check_input_transposed(A, "xLASWP(A, ipiv), A");
  check_input_transposed(ipiv, "xLASWP(A, ipiv), ipiv");
  check_dimensions(A.rows(), 1, ipiv, "xLASWP(A, ipiv), ipiv");
#endif

  auto location = A._location;
  auto device_id = A._device_id;

  auto n_         = A.cols();
  auto A_         = A._begin();
  auto lda_       = A._leading_dimension;
  auto k1_        = 1;              // Fortran indexing
  auto k2_        = ipiv.rows();    // dito
  auto ipiv_      = ipiv._begin();
  auto incx_      = 1;

  if (location == Location::host) {

    FORTRAN::xLASWP(n_, A_, lda_, k1_, k2_, ipiv_, incx_);

  }
#ifdef HAVE_CUDA
# ifdef HAVE_MAGMA
  else if (location == Location::GPU) {

# ifndef LINALG_NO_CHECKS
    check_gpu_structures("xLASWP()");
    if (ipiv._location != Location::host) {
    
      throw excBadArgument("xLASWP(A, ipiv), ipiv: pivot vector ipiv must "
                           "reside in main memory");

    }
# endif

    // Magma's GPU based laswp assumes RowMajor storage so we create a  
    // temporary and stream back afterwards
    
    Dense<T> A_tmp;
    A_tmp.clone_from(A);

    A_tmp.format(Format::RowMajor);

    /*
    auto A_tmp_   = A_tmp._begin();
    n_            = A.rows();

    MAGMA::xLASWP(n_, A_tmp_, lda_, k1_, k2_, ipiv_, incx_);
    */

    A << A_tmp;

  }
# endif /* HAVE_MAGMA */
#endif /* HAVE_CUDA */

#ifndef LINALG_NO_CHECKS
  else {
    throw excUnimplemented("xLASWP(): LAPACK LASWP not supported on selected "
                           "location");
  }
#endif

}

} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_LASWP_H_ */
