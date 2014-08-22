/** \file
 *
 *  \brief            xCOPY (BLAS-1)
 *
 *  \date             Created:  Jul 16, 2014
 *  \date             Modified: $Date$
 *
 *  \authors          mauro <mcalderara@iis.ee.ethz.ch>
 *
 *  \version          $Revision$
 */
#ifndef LINALG_BLAS_COPY_H_
#define LINALG_BLAS_COPY_H_

/* Organization of the namespace:
 *
 *    LinAlg::BLAS
 *        convenience bindings supporting different locations for Dense<T>
 *
 *    LinAlg::BLAS::<NAME>
 *        bindings to the <NAME> BLAS backend
 */

#include "../types.h"      // need LinAlg::size_t before MKL header

#include "../utilities/checks.h"
#include "../dense.h"

namespace LinAlg {

namespace BLAS {

namespace FORTRAN {

#ifndef DOXYGEN_SKIP
extern "C" {
  void fortran_name(scopy, SCOPY)(I_t* n, S_t* x, I_t* incx, S_t* y, I_t* incy);
  void fortran_name(dcopy, DCOPY)(I_t* n, D_t* x, I_t* incx, D_t* y, I_t* incy);
  void fortran_name(ccopy, CCOPY)(I_t* n, C_t* x, I_t* incx, C_t* y, I_t* incy);
  void fortran_name(zcopy, ZCOPY)(I_t* n, Z_t* x, I_t* incx, Z_t* y, I_t* incy);
}
#endif

/** \brief            General matrix-matrix multiply
 *
 *  y <- x
 *
 *  \param[in]        n
 *
 *  \param[in]        x
 *
 *  \param[in]        incx
 *
 *  \param[in]        y
 *
 *  \param[in]        incy
 *
 *  See [DCOPY](http://www.mathkeisan.com/usersguide/man/dcopy.html)
 */
inline void xCOPY(int n, S_t* x, int incx, S_t* y, int incy) {
  fortran_name(scopy, SCOPY)(&n, x, &incx, y, &incy);
};
/** \overload
 */
inline void xCOPY(int n, D_t* x, int incx, D_t* y, int incy) {
  fortran_name(dcopy, DCOPY)(&n, x, &incx, y, &incy);
};
/** \overload
 */
inline void xCOPY(int n, C_t* x, int incx, C_t* y, int incy) {
  fortran_name(ccopy, CCOPY)(&n, x, &incx, y, &incy);
};
/** \overload
 */
inline void xCOPY(int n, Z_t* x, int incx, Z_t* y, int incy) {
  fortran_name(zcopy, ZCOPY)(&n, x, &incx, y, &incy);
};

} /* namespace LinAlg::BLAS::FORTRAN */

} /* namespace LinAlg::BLAS */

} /* namespace LinAlg */

#endif /* LINALG_BLAS_COPY_H_ */
