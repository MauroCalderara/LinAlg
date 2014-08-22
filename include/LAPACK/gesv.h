/** \file
 *
 *  \brief            xGESV
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LAPACK_GESV_H_
#define LINALG_LAPACK_GESV_H_

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
#include <magma_lapack.h>
#endif

#endif


#include "../types.h"
#include "../exceptions.h"
#include "../utilities/checks.h"
#include "../dense.h"

namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

#ifndef DOXYGEN_SKIP
extern C {
  void fortran_name(sgesv, SGESV)(const I_t* n, const I_t* nrhs, S_t* A,
                                  const I_t* lda, I_t* ipiv, S_t* B,
                                  const I_t* ldb, int* info);
  void fortran_name(dgesv, DGESV)(const I_t* n, const I_t* nrhs, D_t* A,
                                  const I_t* lda, I_t* ipiv, D_t* B,
                                  const I_t* ldb, int* info);
  void fortran_name(cgesv, CGESV)(const I_t* n, const I_t* nrhs, C_t* A,
                                  const I_t* lda, I_t* ipiv, C_t* B,
                                  const I_t* ldb, int* info);
  void fortran_name(zgesv, ZGESV)(const I_t* n, const I_t* nrhs, Z_t* A,
                                  const I_t* lda, I_t* ipiv, Z_t* B,
                                  const I_t* ldb, int* info);
}
#endif

/** \brief            GESV
 *
 *  X = A^(-1) * B
 *
 *  \param[in]        n
 *
 *  \param[in]        nrhs
 *
 *  \param[in]        A
 *
 *  \param[in]        lda
 *
 *  \param[in]        ipiv
 *
 *  \param[in,out]    B
 *
 *  \param[in]        ldb
 *
 *  \param[in,out]    info
 *
 *  See [DGESV](http://www.mathkeisan.com/UsersGuide/man/dgesv.html)
 */
inline void xGESV(I_t n, I_t nrhs, S_t* A, I_t lda, int* ipiv, S_t* B, int ldb,
                  int* info) {
  fortran_name(sgesv, SGESV)(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
};
/** \overload
 */
inline void xGESV(I_t n, I_t nrhs, D_t* A, I_t lda, int* ipiv, D_t* B, int ldb,
                  int* info) {
  fortran_name(dgesv, DGESV)(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
};
/** \overload
 */
inline void xGESV(I_t n, I_t nrhs, C_t* A, I_t lda, int* ipiv, C_t* B, int ldb,
                  int* info) {
  fortran_name(cgesv, CGESV)(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
};
/** \overload
 */
inline void xGESV(I_t n, I_t nrhs, Z_t* A, I_t lda, int* ipiv, Z_t* B, int ldb,
                  int* info) {
  fortran_name(zgesv, ZGESV)(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
};

} /* namespace LinAlg::LAPACK::FORTRAN */

} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_GESV_H_ */
