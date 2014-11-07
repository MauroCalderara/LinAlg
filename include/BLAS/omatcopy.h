/** \file
 *
 *  \brief            xomatcopy (MKL BLAS-like)
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_BLAS_OMATCOPY_H_
#define LINALG_BLAS_OMATCOPY_H_

/* Organization of the namespace:
 *
 *    LinAlg::BLAS
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::BLAS::<NAME>
 *        bindings to the <NAME> BLAS backend
 */

#ifdef HAVE_MKL

#include "../types.h"      // need LinAlg::size_t before MKL header
#include <mkl.h>
#include "../profiling.h"

namespace LinAlg {

namespace BLAS {

namespace MKL {

// mkl_?omatcopy
/** \brief            Out-of-place matrix copy/transpose
 *
 *  B <- alpha * op(A)
 *
 *  \param[in]        ordering
 *
 *  \param[in]        trans
 *
 *  \param[in]        rows
 *
 *  \param[in]        cols
 *
 *  \param[in]        alpha
 *
 *  \param[in]        A
 *
 *  \param[in]        lda
 *
 *  \param[in,out]    B
 *
 *  \param[in]        ldb
 *
 *  See MKL Documentation for BLAS-like functions
 */
inline void xomatcopy(char ordering, char trans, const I_t rows, const I_t cols,
                      const S_t alpha, const S_t* A, I_t lda, S_t* B, I_t ldb) {

  PROFILING_FUNCTION_HEADER

  mkl_somatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);

}
/** \overload
 */
inline void xomatcopy(char ordering, char trans, const I_t rows, const I_t cols,
                      const D_t alpha, const D_t* A, I_t lda, D_t* B, I_t ldb) {

  PROFILING_FUNCTION_HEADER

  mkl_domatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);

}
/** \overload
 */
inline void xomatcopy(char ordering, char trans, const I_t rows, const I_t cols,
                      const C_t alpha, const C_t* A, I_t lda, C_t* B, I_t ldb) {

  PROFILING_FUNCTION_HEADER

  mkl_comatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);

}
/** \overload
 */
inline void xomatcopy(char ordering, char trans, const I_t rows, const I_t cols,
                      const Z_t alpha, const Z_t* A, I_t lda, Z_t* B, I_t ldb) {

  PROFILING_FUNCTION_HEADER

  mkl_zomatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);

}

} /* namepsace LinAlg::BLAS::MKL */

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* HAVE_MKL */

#endif /* LINALG_BLAS_OMATCOPY_H_ */
