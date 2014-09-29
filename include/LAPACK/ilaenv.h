/** \file
 *
 *  \brief            ILAENV
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_LAPACK_ILAENV_H_
#define LINALG_LAPACK_ILAENV_H_

/* Organization of the namespace:
 *
 *    LinAlg::LAPACK
 *        convenience bindings supporting different locations for Dense<T>
 *
 *        'Abstract' functions like 'solve' and 'invert'
 *
 *    LinAlg::LAPACK::<NAME>
 *        bindings to the <NAME> LAPACK backend
 */

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CUDA/cuda_checks.h"

#ifdef HAVE_MAGMA
#include <magma.h>
#endif

#endif


#include "../types.h"

#ifndef DOXYGEN_SKIP
extern "C" {

  using LinAlg::I_t;
  using LinAlg::S_t;
  using LinAlg::D_t;
  using LinAlg::C_t;
  using LinAlg::Z_t;

  I_t fortran_name(ilaenv, ILAENV)(const I_t* ispec, const char* name,
                                   const char* opts, const I_t* n1,
                                   const I_t* n2, const I_t* n3, const I_t* n4);
}
#endif

namespace LinAlg {

namespace LAPACK {

namespace FORTRAN {

/** \brief            ILAENV
 *
 *  \param[in]        ispec
 *
 *  \param[in]        name
 *
 *  \param[in]        opts
 *
 *  \param[in]        n1
 *
 *  \param[in]        n2
 *
 *  \param[in]        n3
 *
 *  \param[in]        n4
 *
 *  \returns          The result of the query
 *
 *  See [ILAENV](http://www.math.utah.edu/software/lapack/lapack-i.html#ilaenv)
 */
inline I_t ILAENV(I_t ispec, const char* name, const char* opts, I_t n1, I_t n2,
                  I_t n3, I_t n4) {
  return fortran_name(ilaenv, ILAENV)(&ispec, name, opts, &n1, &n2, &n3, &n4);
};

} /* namespace LinAlg::LAPACK::FORTRAN */

} /* namespace LinAlg::LAPACK */

} /* namespace LinAlg */

#endif /* LINALG_LAPACK_ILAENV_H_ */
